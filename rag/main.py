import os
import logging
import time
import requests
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import OpenLLM

from rag.message_clients import APIClient, MessageClient
from rag.document_loader import LocalDocumentLoader
from llm_processor import (BaseLLMProcessor, LLMProcessorOllama, LLMProcessorOpenLLM, LLMProcessorHuggingFace,
                           LLMProcessorOpenAI)


RESTART_AFTER_EXCEPTION_DELAY_SEC = 60




from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from llama_index.core import Document

# para ver el funcionamiento del RAG
# import logging
# import sys
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Descargar los recursos necesarios de nltk.
#nltk.download('punkt')
nltk.download('punkt_tab')

def chunk_text(text, chunk_size=4000):
    # Divide el texto en oraciones  
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def chunk_documents(documents):
    chunked_documents = []
    for doc in documents:
        # Acceder al texto de Document con doc.text
        chunks = chunk_text(doc.text)
        for chunk in chunks:
            # Crear objetos Document en lugar de diccionarios
            chunked_documents.append(Document(text=chunk))
    return chunked_documents

# Cargar documentos
documents = SimpleDirectoryReader("/app/resources/files").load_data()

# Aplicar chunking
chunked_documents = chunk_documents(documents)

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm = Ollama(base_url="http://localhost:11434", model="mistral", request_timeout=1000.0)

# Crear índice con los objetos Document
index = VectorStoreIndex.from_documents(chunked_documents)

query_engine = index.as_query_engine(similarity_top_k=5)    # "top_k" documents
#query_engine = index.as_query_engine()    # "top_k" documents

class MessageProcessor:
    def __init__(self, api: MessageClient, document_loader, mongo_collection, llm: BaseLLMProcessor):
        self.api = api
        self.document_loader = document_loader
        self.mongo_collection = mongo_collection
        self.llm = llm

    def run(self):
        logging.info("Start processing messages")
        # Load the document
        # logging.debug(f"Loading files document")
        # try:
        #     file_data = self.document_loader.load_document()
        # except Exception as e:
        #     logging.error(f"No document in folder resources/files: {e}")
        #     time.sleep(1)
        #     raise e
        try:
            for i, message in enumerate(self.api.messages()):
                logging.info(f"{message.input}: Processing prompt...")
                # Load the document
                # logging.debug(f"Loading file from document: {message.context}")
                # try:
                #     file_data = self.document_loader.load_document(message)
                # except Exception as e:
                #     logging.info(f"{message.document_id}: No document in the S3 storage: {e}")
                #     time.sleep(1)
                #     raise e

                start_time = time.time()

                logging.info("Getting answers based on question")
                #response = self.llm.process_questionnaire(message.input, file_data)
                logging.info(f"type message.input: {type(message.input)} // message.input: {message.input}")
                prompt = f"""Eres un asistente que debe responder preguntas basadas únicamente en la información proporcionada en el contexto pasado. No inventes respuestas ni utilices conocimientos externos, pero si intenta buscar, inferir o deducir la respuesta a partir de la información en el contexto. Si la respuesta no puede obtenerse de esa forma, simplemente responde: "No sé la respuesta basada en la información proporcionada."
                Pregunta: {message.input}
                """
                response = query_engine.query(prompt)

                print("\n\n--- Documentos usados como contexto ---")
                for node in response.source_nodes:
                    print(f"Score: {node.score}")
                    print(f"{node.node.get_content()[:2000]}\n")  # Mostramos primeros 500 caracteres



                end_time = time.time()

                execution_time = end_time - start_time
                logging.info(f"Execution time of LLM: {execution_time/60} minutes")

                # Process the input and generate the output
                output = response.response # self.process_input(message_data.input)
                logging.info(f"\n\n---------------------------------------\ntype output: {type(output)}\n-------------------\noutput: {output}")

                # TODO move this to a function or class
                # Update the MongoDB document with the new output value
                self.mongo_collection.update_one(
                    {"_id": ObjectId(message.id)},
                    {
                        "$set": {
                            "output": output,
                            "date_out": datetime.now()
                        }
                    }
                )
                logging.info(f"Message {message.id} updated with output: {output}")


        except Exception as e:
            logging.error(f"Error processing message: {e}")

    def run_continuous(self):
        while True:
            self.run()
            time.sleep(RESTART_AFTER_EXCEPTION_DELAY_SEC)


def main(api_url, document_location, mongo_host, mongo_port, mongo_user, mongo_pass, mongo_db_name,
         openai_api_key, openllm_server, huggingface_server, ollama_server_url):
    logging.info("Starting system")
    api_client = APIClient(api_url, token=None)
    logging.info(f"Connecting to local database {mongo_host}")
    client = MongoClient(f"mongodb://{mongo_user}:{mongo_pass}@{mongo_host}:{mongo_port}/")
    mongo_db = client[mongo_db_name]

    mongo_collection = mongo_db['prompts']  # Collection name
    logging.debug(f"Loading documents from local location: {document_location}")
    document_loader = LocalDocumentLoader(document_location)

    if openai_api_key is not None:
        logging.debug("LLM source set to: OpenAI API")
        # OpenAI model, be wary of the cost
        from langchain.embeddings.openai import OpenAIEmbeddings
        GPT4_Turbo = "gpt-4-1106-preview"
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=GPT4_Turbo, temperature=0.2)
        context_length = 128000
        embedding = OpenAIEmbeddings()
        llm_processor = LLMProcessorOpenAI(llm, embedding, context_length)
    elif openllm_server is not None:
        logging.debug("LLM source set to: OpenLLM")
        # Local LLM hosted with OpenLLM
        llm = OpenLLM(server_url=openllm_server, timeout=6000)
        llm_processor = LLMProcessorOpenLLM(llm)
    elif huggingface_server is not None:
        logging.debug("LLM source set to: HuggingFace")
        output_tokens = 1024
        llm_processor = LLMProcessorHuggingFace(huggingface_server, output_tokens)
    else:
        logging.debug("LLM source set to: Ollama")
        # Local LLM hosted with Ollama
        from langchain.llms import Ollama
        from langchain.embeddings import GPT4AllEmbeddings
        # from langchain.embeddings import OllamaEmbeddings
        model = "mistral"
        host = "http://localhost:11434"
        llm = Ollama(model=model,
            base_url=host,
            temperature=0.1,
            top_p=0.9,
            num_ctx=6000)
        context_length = 5000
        # embedding = OllamaEmbeddings(
        #     model=model,
        #     base_url=host
        # )
        embedding = GPT4AllEmbeddings()
        llm_processor = LLMProcessorOllama(llm, embedding, context_length)
        assert requests.get(url=ollama_server_url).ok, "Ollama is not running"

    message_processor = MessageProcessor(api_client, document_loader, mongo_collection, llm_processor)
    message_processor.run_continuous()


if __name__ == '__main__':
    API_URL = os.environ.get('API_URL', None)
    DOCUMENT_LOCATION = os.getenv("DOCUMENT_LOCATION", None)
    MONGO_HOST = os.getenv('MONGO_HOST')
    MONGO_PORT = int(os.getenv('MONGO_PORT'))
    MONGO_USER = os.getenv('MONGO_USER')
    MONGO_PASS = os.getenv('MONGO_PASS')
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')  # Replace with your MongoDB database name
    OPENLLM_SERVER = os.getenv('OPENLLM_SERVER', None)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', None)
    HUGGINGFACE_SERVER = os.getenv('HUGGINGFACE_SERVER', None)
    OLLAMA_SERVER_URL = os.getenv('OLLAMA_SERVER_URL', None)

    main(API_URL, DOCUMENT_LOCATION, MONGO_HOST, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_DB_NAME,
         OPENAI_API_KEY, OPENLLM_SERVER, HUGGINGFACE_SERVER, OLLAMA_SERVER_URL)
