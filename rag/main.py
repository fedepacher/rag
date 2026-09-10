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
from rag.llm_processor import (BaseLLMProcessor, LLMProcessorOllama, LLMProcessorOpenLLM, LLMProcessorHuggingFace,
                               LLMProcessorOpenAI, RETRIEVAL_K)


RESTART_AFTER_EXCEPTION_DELAY_SEC = 60

# Ollama backend configuration. Kept as module-level constants so that offline tooling
# (e.g. resources/eval/run_baseline.py) measures exactly the same pipeline that runs in
# production instead of re-declaring these values and silently drifting away from it.
OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.0
OLLAMA_TOP_P = 0.9
OLLAMA_NUM_CTX = 6000
# Secondary guard against the repetition loop that num_predict caps.
#
# The llama-server parameter dump for the runaway run read "repeat_last_n = 64,
# repeat_penalty = 1.000": no penalty was in effect at all. At temperature 0 nothing
# else breaks a loop once the model enters one, because the same argmax keeps being the
# argmax -- so an uncapped generation had no mechanism to stop itself.
#
# num_predict is the hard stop and this is only prophylaxis: it makes entering the loop
# less likely, it does not bound anything. Kept mild (1.1) because a high repetition
# penalty is actively harmful here -- answers about electronics legitimately repeat
# terminology and formula symbols, and penalising that degrades the answer.
OLLAMA_REPEAT_PENALTY = 1.1
# num_ctx bounds prompt + completion together, so the prompt may only use what is left
# after the answer we intend to generate.
#
# This value is ENFORCED as num_predict on the generator, not merely subtracted when
# sizing chunks. Both halves of num_ctx have to be bounded or neither is: with the
# completion left open, a degenerate generation runs past num_ctx and llama.cpp starts
# context-shifting (n_keep=5, n_discard=3069 per shift) rather than stopping. The
# retrieved chunks and the instruction header are evicted, the model generates from an
# empty context and never emits EOS. Observed 2026-09-10: a GROUNDED_RETRY_PROMPT
# regeneration reached 18673 tokens in 2h39m at 2.05 tok/s and stalled the queue, which
# is processed serially. A truncated answer is a bounded, visible failure; an unbounded
# one is not a failure the service can even detect.
OLLAMA_ANSWER_TOKEN_BUDGET = 900
# Tokens the prompt spends on things that are not retrieved text: the longest generation
# template (GROUNDED_RETRY_PROMPT, ~200 tokens) plus a student question (~60).
PROMPT_OVERHEAD_TOKENS = 300
# Chunk size handed to TokenTextSplitter, in TOKENS -- the splitter counts cl100k_base
# tokens, not characters. DERIVED from the budget above rather than hand-picked, so it
# cannot drift out of step with num_ctx or RETRIEVAL_K again.
#
# It replaces a hand-picked OLLAMA_CONTEXT_LENGTH = 5000, whose name suggested a model
# context window while its value was a chunk size in tokens. One such chunk nearly filled
# num_ctx on its own, so a RETRIEVAL_K=4 prompt measured 20137 tokens against a 6000-token
# window. Ollama truncates rather than failing, and llama.cpp keeps the tail (n_keep=4):
# the instruction header -- "responde utilizando unicamente la informacion contenida en el
# documento" and the "No se la respuesta..." escape hatch -- was the first thing dropped,
# which is how the generator ended up answering from parametric memory.
CHUNK_SIZE_TOKENS = (OLLAMA_NUM_CTX - OLLAMA_ANSWER_TOKEN_BUDGET - PROMPT_OVERHEAD_TOKENS) // RETRIEVAL_K

# CRAG control model, pulled by entrypoint.sh alongside the generation model. Phi-3.5-mini
# grades retrieval relevance and rewrites queries; both are classification-style jobs, so
# temperature is 0 for determinism instead of the 0.9 top_p sampling the generator uses.
# It stays resident next to Llama for the lifetime of the process (~3.2 GiB on top of the
# generator's ~5.6 GiB, so ~8.5-9 GiB combined on the 16 GB target box). One instance is
# built here and shared by every control node — never one per node.
PHI_MODEL = "phi3.5:3.8b-mini-instruct-q4_K_M"
PHI_TEMPERATURE = 0.0
PHI_NUM_CTX = 6000
# Hard cap on what a control node may emit, for the same reason the generator has one.
# Sized by the longest thing any control prompt legitimately asks for: a reformulated
# query, itself bounded by REFORMULATED_QUERY_MAX_CHARS (500 chars, ~160 tokens at the
# ~3.1 chars/token this corpus measures). RELEVANCE_PROMPT and GROUNDING_PROMPT ask for
# a single word. 200 leaves room without leaving the loop open.
#
# Truncation cannot corrupt a verdict: parse_relevance and parse_grounding keyword-match
# free text and fall back to the permissive value, so a cut-off answer degrades the
# pipeline towards classic RAG rather than towards a refusal.
PHI_NUM_PREDICT = 200

# Embedding model behind the FAISS index, pulled by entrypoint.sh and served by the same
# local Ollama instance as the generator and the control model.
#
# Replaces GPT4AllEmbeddings, whose weakness on Spanish technical prose was measured
# rather than assumed. For the question "¿Por qué los FET presentan una elevada
# impedancia de entrada?" scored against a gate-insulation passage and a power-amplifier
# distractor, GPT4AllEmbeddings separated them by 0.0346 cosine (0.4030 vs 0.3684) while
# bge-m3 separated them by 0.1249 (0.5669 vs 0.4421) -- a 3.6x wider margin. Both rank
# the right passage first in a three-way test; the MARGIN is what decides whether it
# survives into the top RETRIEVAL_K of 163 real chunks, and at 0.03 it did not. That
# question retrieved three chunks of "AMPLIFICADORES DE POTENCIA - GALIANO" and none of
# "TransistoresdeEfectoDeCampo.pdf", and the answer was withheld.
#
# Cost: ~1.2 GB of weights, a third model on a box already holding ~8.8 GiB of Llama plus
# Phi. Ollama unloads idle models so this is not a permanent +1.2 GB, but the 16 GB target
# is tighter than before, and this dev box has already had the container SIGKILLed once
# under memory pressure.
#
# Dimensions change 384 -> 1024, so a stored FAISS index is invalid rather than merely
# stale. LLMProcessorOllama.get_index_hash covers the embedding model for that reason.
EMBEDDING_MODEL = "bge-m3"


def build_phi_control_llm():
    """Build the Phi-3.5-mini instance backing the CRAG control nodes.

    Returns:
        Ollama: LLM client pointed at the same local Ollama server as the generator.
    """
    from langchain.llms import Ollama

    return Ollama(model=PHI_MODEL,
                  base_url=OLLAMA_HOST,
                  temperature=PHI_TEMPERATURE,
                  num_ctx=PHI_NUM_CTX,
                  num_predict=PHI_NUM_PREDICT)


def build_ollama_embeddings():
    """Build the embedding model used to index and query the course bibliography.

    Served by the Ollama instance already running in this container, so it costs no new
    Python dependency and, unlike GPT4AllEmbeddings, reaches nothing outside the box.
    That removes an entire class of startup failure: GPT4AllEmbeddings' constructor did
    a bare requests.get on gpt4all.io's model index, and an intermittent CDN 503 took
    the service down three times in a row on a machine meant to run unattended. The
    bounded retry that worked around it is gone with the dependency it protected.

    Returns:
        OllamaEmbeddings: Client for EMBEDDING_MODEL on the local Ollama server. Nothing
        is contacted at construction time; the first embed call is what needs the model,
        and entrypoint.sh has already pulled it by then.
    """
    from langchain_community.embeddings import OllamaEmbeddings

    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)


def build_ollama_processor(enable_crag: bool = True) -> LLMProcessorOllama:
    """Build the Ollama-backed processor used by the default pipeline.

    Args:
        enable_crag (bool): Wire the Phi-3.5-mini control model in so the graph runs the
            CRAG correction loop (relevance grading, bounded reformulation, out-of-scope
            short-circuit). Pass False to get the classic linear retrieve -> generate
            pipeline and to avoid loading the control model at all; that is what the
            baseline harness measures, so the pre-CRAG reference point stays reachable.

    Returns:
        LLMProcessorOllama: Processor wired to the local Ollama server with
        bge-m3 embeddings over Ollama and a FAISS vector store.
    """
    from langchain.llms import Ollama
    # from langchain.embeddings import OllamaEmbeddings

    llm = Ollama(model=OLLAMA_MODEL,
                 base_url=OLLAMA_HOST,
                 temperature=OLLAMA_TEMPERATURE,
                 top_p=OLLAMA_TOP_P,
                 num_ctx=OLLAMA_NUM_CTX,
                 num_predict=OLLAMA_ANSWER_TOKEN_BUDGET,
                 repeat_penalty=OLLAMA_REPEAT_PENALTY)
    # embedding = OllamaEmbeddings(
    #     model=OLLAMA_MODEL,
    #     base_url=OLLAMA_HOST
    # )
    embedding = build_ollama_embeddings()
    grader_llm = build_phi_control_llm() if enable_crag else None
    return LLMProcessorOllama(llm, embedding, CHUNK_SIZE_TOKENS, grader_llm=grader_llm)


class MessageProcessor:
    def __init__(self, api: MessageClient, document_loader, mongo_collection, llm: BaseLLMProcessor):
        self.api = api
        self.document_loader = document_loader
        self.mongo_collection = mongo_collection
        self.llm = llm

    def run(self):
        logging.info("Start processing messages")
        # Load the document
        logging.debug("Loading files document")
        try:
            file_data = self.document_loader.load_document()
            context_chunked = file_data.get_chunked_text(self.llm.context_length)
            # Build/load FAISS index once
            self.llm.build_or_load_vectorstore(context_chunked)
        except Exception as e:
            logging.error(f"No document in folder resources/files: {e}")
            time.sleep(1)
            raise e
        try:
            for i, message in enumerate(self.api.messages()):
                logging.info(f"{message.input}: Processing prompt...")

                start_time = time.time()

                logging.info("Getting answers based on question")
                response = self.llm.process_questionnaire(message.input, context_chunked)

                end_time = time.time()

                execution_time = end_time - start_time
                logging.info(f"Execution time of LLM: {execution_time/60} minutes")

                # Process the input and generate the output
                output = response # self.process_input(message_data.input)

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
        # Local LLM hosted with Ollama, with the CRAG correction loop enabled
        llm_processor = build_ollama_processor()
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
