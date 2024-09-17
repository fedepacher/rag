import logging
import os
import json
import requests

from document_loader import Document
os.environ['ANONYMIZED_TELEMETRY'] = 'False' # Chroma spyware
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate


class BaseLLMProcessor:
    def __init__(self, llm, embedding, context_length):
        self.llm = llm
        self.embedding = embedding
        self.context_length = context_length

    def answer_json_parser(self, str_obj: str) -> list:
        pass

    def ask_question(self, question: str, context: str):
        pass

    def process_questionnaire(self, question: str, file_data: Document) -> str:
        context = file_data.get_file_content()

        answer = self.ask_question(question, context)

        return answer


class LLMProcessorOpenAI(BaseLLMProcessor):
    def __init__(self, llm, embedding, context_length):
        super().__init__(llm, embedding, context_length)

    def ask_question(self, question: str, context: str):
        logging.debug(f"Processing {len(context)} chunks of text for question")
        vectorstore = Chroma.from_texts(texts=[context], embedding=self.embedding)

        template = """Eres un asistente que debe responder preguntas basadas únicamente en la información proporcionada en el siguiente documento. No inventes respuestas ni utilices conocimientos externos. Si la respuesta no se encuentra en el documento, simplemente responde: "No sé la respuesta basada en la información proporcionada."

                    [DOCUMENTO: {context}]
                    
                    Pregunta: {question}
                    Respuesta:
                   """
        qa_chain_prompt = PromptTemplate.from_template(template)  # Run chain
        qachain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=False,
            chain_type_kwargs={"prompt": qa_chain_prompt}
        )
        llm_response = ''
        try:
            logging.debug(f"Waiting answer for question")
            llm_response = qachain({"query": question})
        except Exception as err:
            logging.info(f'Exception occurred on chunk: {err}')
        return llm_response


class LLMProcessorOpenLLM(BaseLLMProcessor):
    def __init__(self, llm):
        super().__init__(llm, None, None)

    def ask_question(self, question: str, context: str):
        template = """Responda la pregunta basandose unicamente en el contexto del PDF. Si no encunetra la respuesta no alucine, responda que no sabe. 
                      Pregunta: {question}
                      Contexto: {context}
                      Respuesta:
                   """

        qa_chain_prompt = PromptTemplate(template=template, input_variables=["question", "context"])

        # Create the LLMChain
        chain = LLMChain(llm=self.llm, prompt=qa_chain_prompt)

        logging.debug(f"Waiting answer for question")
        llm_response = ''
        try:
            llm_response: str = chain.run({"question": question, "context": context})
        except Exception as err:
            logging.info(f'Exception occurred on text: {err}')

        return llm_response


class LLMProcessorHuggingFace(BaseLLMProcessor):
    def __init__(self, server_url: str, output_tokens: int):
        super().__init__(None, None, None)
        self.server_url = server_url
        self.output_tokens = output_tokens

    def ask_question(self, question: str, context: str):

        template = """Responda la pregunta basandose unicamente en el contexto del PDF. Si no encunetra la respuesta no alucine, responda que no sabe. 
                      Pregunta: {question}
                      Contexto: {context}
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
