import logging
import os
import requests
from typing import List

os.environ['ANONYMIZED_TELEMETRY'] = 'False'  # Chroma spyware
from langchain_community.vectorstores import Chroma, FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from document_loader import Document


INITIAL_PROMPT = """Eres un asistente que debe responder preguntas basadas únicamente en la información proporcionada en el siguiente documento. No inventes respuestas ni utilices conocimientos externos. Si la respuesta no se encuentra en el documento, simplemente responde: "No sé la respuesta basada en la información proporcionada."

                    [DOCUMENTO: {context}]
    
                    Pregunta: {question}
                    Respuesta:
                   """

class BaseLLMProcessor:
    def __init__(self, llm, embedding, context_length):
        self.llm = llm
        self.embedding = embedding
        self.context_length = context_length

    def answer_json_parser(self, str_obj: str) -> list:
        pass

    def ask_question(self, question: str, context: List[str]):
        pass

    def process_questionnaire(self, question: str, context: List[str]) -> str | None:
        answer = self.ask_question(question, context)
        return answer


class LLMProcessorOllama(BaseLLMProcessor):
    def __init__(self, llm, embedding, context_length):
        super().__init__(llm, embedding, context_length)

    def ask_question(self, question: str, context: List[str]) -> str:
        """
        Answer a question using the Mixtral-8x7B model with FAISS in-memory vector store.

        Args:
            question (str): The question to answer.
            context (List[str]): List of text chunks to use as context.

        Returns:
            str: The answer or an error message if processing fails.
        """
        try:
            logging.info(f"Processing question: {question} with {len(context)} context chunks")
            if not question.strip():
                logging.error("Question is empty")
                return "Error: La pregunta no puede estar vacía."
            if not context or not all(isinstance(c, str) for c in context):
                logging.error("Invalid or empty context")
                return "Error: El contexto proporcionado es inválido o está vacío."

            # Create FAISS in-memory vector store
            vectorstore = FAISS.from_texts(texts=context, embedding=self.embedding)
            logging.debug("FAISS vector store created successfully")

            # Define prompt
            qa_chain_prompt = PromptTemplate.from_template(INITIAL_PROMPT)

            # Set up RetrievalQA chain
            qachain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
                return_source_documents=False,
                chain_type_kwargs={"prompt": qa_chain_prompt}
            )
            logging.debug("RetrievalQA chain initialized")

            # Query LLM
            logging.debug("Waiting for LLM response")
            llm_response = qachain({"query": question})
            answer = llm_response.get("result", "") if isinstance(llm_response, dict) else str(llm_response)
            logging.info(f"Answer generated: {answer[:100]}...")
            return answer

        except Exception as err:
            logging.error(f"Error processing question '{question}': {err}", exc_info=True)
            return f"Error: No se pudo procesar la pregunta debido a: {str(err)}"


class LLMProcessorOpenAI(BaseLLMProcessor):
    def __init__(self, llm, embedding, context_length):
        super().__init__(llm, embedding, context_length)

    def ask_question(self, question: str, context):
        try:
            logging.debug(f"Processing {len(context)} chunks of text for question")
            vectorstore = Chroma.from_texts(texts=context, embedding=self.embedding)

            template = INITIAL_PROMPT
            qa_chain_prompt = PromptTemplate.from_template(template)  # Run chain
            qachain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=vectorstore.as_retriever(),
                return_source_documents=False,
                chain_type_kwargs={"prompt": qa_chain_prompt}
            )
            llm_response = ''

            logging.debug("Waiting answer for question")
            llm_response = qachain({"query": question})
        except Exception as err:
            logging.error(f'Exception occurred on chunk: {err}')
        return llm_response


class LLMProcessorOpenLLM(BaseLLMProcessor):
    def __init__(self, llm):
        super().__init__(llm, None, None)

    def ask_question(self, question: str, context):
        try:
            template = INITIAL_PROMPT

            qa_chain_prompt = PromptTemplate(template=template, input_variables=["question", "context"])

            # Create the LLMChain
            chain = LLMChain(llm=self.llm, prompt=qa_chain_prompt)

            logging.debug("Waiting answer for question")
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

        template = INITIAL_PROMPT

        qa_prompt = template.format(question=question, context=context)

        payload = {
            "prompt": qa_prompt,
            "tokens": self.output_tokens
        }

        logging.debug("Waiting answer for question")
        response = requests.post(self.server_url, json=payload)
        llm_response = ''
        if response.status_code == 200:
            llm_response = response.json()
        else:
            logging.info(f'Request failed with status code: {response.status_code}')

        return llm_response
