from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize

# Descargar los recursos necesarios de nltk
nltk.download('punkt')

def chunk_text(text, chunk_size=200):
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
        # Aplicar chunking a cada documento
        chunks = chunk_text(doc['text'])
        for chunk in chunks:
            chunked_documents.append({'text': chunk})
    return chunked_documents

# Cargar documentos
documents = SimpleDirectoryReader("C:\\cmv-new\\Python\\custom_chat\\doc_pp").load_data()

# Aplicar chunking
chunked_documents = chunk_documents(documents)

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm = Ollama(model="llama2", request_timeout=1000.0)

# Crear índice con chunks
index = VectorStoreIndex.from_documents(chunked_documents)

query_engine = index.as_query_engine()

while True:
    user_input = input("Usuario: ")
    rta = query_engine.query(user_input)
    print(f"rta: {rta}")
