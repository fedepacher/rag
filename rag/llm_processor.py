import logging
import os
import json
import requests
import sqlite3

from document_loader import Document
os.environ['ANONYMIZED_TELEMETRY'] = 'False' # Chroma spyware
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from chromadb import Chroma
from chromadb.config import Settings
from typing import List, Union


class BaseLLMProcessor:
    def __init__(self, llm, embedding, context_length):
        self.llm = llm
        self.embedding = embedding
        self.context_length = context_length

    def answer_json_parser(self, str_obj: str) -> list:
        pass

    def ask_question(self, question: str, context):
        pass

    def process_questionnaire(self, question: str, file_data: Document) -> str | None:
        context = file_data.get_chunked_text(self.context_length)

        answer = self.ask_question(question, context)

        return answer


class LLMProcessorOpenAI(BaseLLMProcessor):
    def __init__(self, llm, embedding, context_length):
        super().__init__(llm, embedding, context_length)

    def ask_question(self, question: str, context: List[str]) -> Union[str, dict]:
        """
        Answer a question using a Mistral model based on provided context.

        Creates a Chroma vector store from the context, retrieves relevant chunks,
        and generates an answer using a RetrievalQA chain. Answers are based solely
        on the provided context, with a fallback response if the answer is not found.

        Args:
            question (str): The question to answer.
            context (List[str]): List of text chunks to use as context.

        Returns:
            Union[str, dict]: The answer as a string or the full LLM response dict
                             (depending on RetrievalQA output). Returns an empty
                             string if an error occurs.

        Raises:
            None: Errors are caught, logged, and an empty string is returned.
        """
        try:
            logging.info(f"Processing question: {question}")
            logging.debug(f"Received {len(context)} context chunks")

            # Configure Chroma to use an in-memory store or a specific database path
            chroma_settings = Settings(
                persist_directory="./chroma_db",  # Specify a directory for persistence
                anonymized_telemetry=False
            )
            try:
                # Initialize Chroma with persistence and ensure database is accessible
                vectorstore = Chroma.from_texts(
                    texts=context,
                    embedding=self.embedding,
                    persist_directory=chroma_settings.persist_directory,
                    client_settings=chroma_settings
                )
                logging.debug("Chroma vector store created successfully")
                vectorstore.persist()  # Ensure data is saved if using persistence
            except sqlite3.OperationalError as sql_err:
                logging.error(f"SQLite database error: {sql_err}")
                # Attempt to recreate the database if the tenants table is missing
                if "no such table: tenants" in str(sql_err):
                    logging.warning("Attempting to recreate Chroma database")
                    try:
                        # Remove existing database file (if it exists)
                        db_path = os.path.join(chroma_settings.persist_directory, "chroma.sqlite3")
                        if os.path.exists(db_path):
                            os.remove(db_path)
                            logging.info(f"Removed corrupted database file: {db_path}")
                        # Recreate vector store
                        vectorstore = Chroma.from_texts(
                            texts=context,
                            embedding=self.embedding,
                            persist_directory=chroma_settings.persist_directory,
                            client_settings=chroma_settings
                        )
                        logging.debug("Chroma vector store recreated successfully")
                        vectorstore.persist()
                    except Exception as recreate_err:
                        logging.error(f"Failed to recreate Chroma database: {recreate_err}")
                        return ""
                else:
                    raise  # Re-raise other SQLite errors

            # Define the prompt template
            template = """Eres un asistente que debe responder preguntas basadas únicamente en la información proporcionada en el siguiente documento. No inventes respuestas ni utilices conocimientos externos. Si la respuesta no se encuentra en el documento, simplemente responde: "No sé la respuesta basada en la información proporcionada."

            [DOCUMENTO: {context}]

            Pregunta: {question}
            Respuesta:
            """
            qa_chain_prompt = PromptTemplate.from_template(template)
            logging.debug("Prompt template created")

            # Set up the RetrievalQA chain
            qachain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=vectorstore.as_retriever(),
                return_source_documents=False,
                chain_type_kwargs={"prompt": qa_chain_prompt}
            )
            logging.debug("RetrievalQA chain initialized")

            # Query the LLM
            logging.info(f"Querying LLM for question: {question}")
            llm_response = qachain({"query": question})
            logging.debug(f"LLM response: {llm_response}")

            return llm_response

        except Exception as err:
            logging.error(f"Error processing question '{question}': {err}", exc_info=True)
            return ""


class LLMProcessorOpenLLM(BaseLLMProcessor):
    def __init__(self, llm):
        super().__init__(llm, None, None)

    def ask_question(self, question: str, context):
        try:
            template = """Eres un asistente que debe responder preguntas basadas únicamente en la información proporcionada en el siguiente documento. No inventes respuestas ni utilices conocimientos externos. Si la respuesta no se encuentra en el documento, simplemente responde: "No sé la respuesta basada en la información proporcionada."
    
                        [DOCUMENTO: {context}]
                        
                        Pregunta: {question}
                        Respuesta:
                       """

            qa_chain_prompt = PromptTemplate(template=template, input_variables=["question", "context"])

            # Create the LLMChain
            chain = LLMChain(llm=self.llm, prompt=qa_chain_prompt)

            logging.debug(f"Waiting answer for question")
            llm_response = ''

            llm_response: str = chain.run({"question": question, "context": context})
        except Exception as err:
            logging.error(f'Exception occurred on text: {err}')

        return llm_response


class LLMProcessorHuggingFace(BaseLLMProcessor):
    def __init__(self, server_url: str, output_tokens: int):
        super().__init__(None, None, None)
        self.server_url = server_url
        self.output_tokens = output_tokens

    def ask_question(self, question: str, context):

        template = """Eres un asistente que debe responder preguntas basadas únicamente en la información proporcionada en el siguiente documento. No inventes respuestas ni utilices conocimientos externos. Si la respuesta no se encuentra en el documento, simplemente responde: "No sé la respuesta basada en la información proporcionada."

                    [DOCUMENTO: {context}]
                    
                    Pregunta: {question}
                    Respuesta:
                   """

        qa_prompt = template.format(question=question, context=context)

        payload = {
            "prompt": qa_prompt,
            "tokens": self.output_tokens
        }

        logging.debug(f"Waiting answer for question")
        response = requests.post(self.server_url, json=payload)
        llm_response = ''
        if response.status_code == 200:
            llm_response = response.json()
        else:
            logging.info(f'Request failed with status code: {response.status_code}')

        return llm_response
