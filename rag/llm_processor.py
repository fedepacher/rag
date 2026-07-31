import logging
import os
import requests
import hashlib
import shutil
from typing import Any, Dict, List, Optional

os.environ['ANONYMIZED_TELEMETRY'] = 'False'  # Chroma spyware
from typing_extensions import TypedDict
from langchain_community.vectorstores import Chroma, FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph


FAISS_INDEX_PATH = "resources/faiss_index"
DOC_HASH_PATH = os.path.join(FAISS_INDEX_PATH, "doc_hash.txt")
# Number of chunks pulled from FAISS per question. Raised from 2 to 4 to give the
# planned CRAG/Self-RAG grading nodes more candidates to filter.
RETRIEVAL_K = 4
# Separator used to stuff the retrieved chunks into the prompt. Matches the default
# document_separator of the StuffDocumentsChain that RetrievalQA used before the
# LangGraph migration, so the generated prompt stays byte-for-byte identical.
DOCUMENT_SEPARATOR = "\n\n"
INITIAL_PROMPT = """
Eres un asistente experto que solo puede responder preguntas utilizando **únicamente** la información contenida en el documento a continuación. 
No debes usar conocimientos previos, hacer suposiciones, inferencias externas ni inventar respuestas. 
Si la información no está presente explícitamente en el documento, responde únicamente: 
**"No sé la respuesta basada en la información proporcionada."**

DOCUMENTO:
{context}

Pregunta: {question}
Respuesta:
"""

class RAGGraphState(TypedDict):
    """State carried through the LangGraph RAG pipeline.

    Attributes:
        question (str): The question being answered, unchanged across the run.
        retrieved_docs (List[Document]): Chunks returned by the FAISS retriever.
        iteration_count (int): Number of retrieval attempts made so far. Always 0
            today; the CRAG reformulation loop is the one that will increment it.
        answer (Optional[str]): Text produced by the generation node, None until then.
    """

    question: str
    retrieved_docs: List[Document]
    iteration_count: int
    answer: Optional[str]


class BaseLLMProcessor:
    def __init__(self, llm, embedding, context_length):
        self.llm = llm
        self.embedding = embedding
        self.context_length = context_length

    def answer_json_parser(self, str_obj: str) -> list:
        pass

    def build_or_load_vectorstore(self, context: List[str]):
        pass

    def ask_question(self, question: str, context: List[str]):
        pass

    def process_questionnaire(self, question: str, context: List[str]) -> str | None:
        answer = self.ask_question(question, context)
        return answer


class LLMProcessorOllama(BaseLLMProcessor):
    def __init__(self, llm, embedding, context_length):
        super().__init__(llm, embedding, context_length)
        self.vectorstore = None
        self.qa_chain_prompt = PromptTemplate.from_template(INITIAL_PROMPT)
        self.graph = None

    @staticmethod
    def get_text_hash(context: list[str]) -> str:
        joined = ''.join(context)
        return hashlib.sha256(joined.encode('utf-8')).hexdigest()

    def document_has_changed(self, context: list[str]) -> bool:
        new_hash = self.get_text_hash(context)
        if os.path.exists(DOC_HASH_PATH):
            with open(DOC_HASH_PATH, 'r') as f:
                old_hash = f.read().strip()
            return new_hash != old_hash
        return True

    def save_hash(self, context: list[str]):
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        with open(DOC_HASH_PATH, 'w') as f:
            f.write(self.get_text_hash(context))

    def build_or_load_vectorstore(self, context: list[str]):
        if os.path.exists(FAISS_INDEX_PATH) and not self.document_has_changed(context):
            self.vectorstore = FAISS.load_local(FAISS_INDEX_PATH, self.embedding)
        else:
            if os.path.exists(FAISS_INDEX_PATH):
                shutil.rmtree(FAISS_INDEX_PATH, ignore_errors=True)
            self.vectorstore = FAISS.from_texts(texts=context, embedding=self.embedding)
            self.vectorstore.save_local(FAISS_INDEX_PATH)
            self.save_hash(context)

    def retrieve_node(self, state: RAGGraphState) -> Dict[str, Any]:
        """Graph node: pull the most similar chunks for the question out of FAISS.

        Args:
            state (RAGGraphState): Current graph state, read for its question.

        Returns:
            Dict[str, Any]: State update carrying the retrieved documents.
        """
        question = state["question"]
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        retrieved_docs = retriever.invoke(question)
        logging.debug(f"Retrieved {len(retrieved_docs)} chunk(s) from FAISS (k={RETRIEVAL_K})")
        return {"retrieved_docs": retrieved_docs}

    def generate_node(self, state: RAGGraphState) -> Dict[str, Any]:
        """Graph node: stuff the retrieved chunks into the prompt and call the LLM.

        Args:
            state (RAGGraphState): Current graph state, read for its question and
                retrieved documents.

        Returns:
            Dict[str, Any]: State update carrying the generated answer.
        """
        question = state["question"]
        retrieved_docs = state["retrieved_docs"]
        context = DOCUMENT_SEPARATOR.join(document.page_content for document in retrieved_docs)
        prompt = self.qa_chain_prompt.format(context=context, question=question)

        logging.debug("Waiting for LLM response")
        llm_response = self.llm.invoke(prompt)
        answer = llm_response if isinstance(llm_response, str) else str(llm_response)
        return {"answer": answer}

    def build_graph(self):
        """Compile the RAG state graph: START -> retrieve -> generate -> END.

        The graph is intentionally linear. Conditional edges (CRAG relevance grading
        and its reformulation loop, Self-RAG grounding verification) are added by the
        follow-up issues; this one only replaces the former RetrievalQA chain.

        Returns:
            CompiledStateGraph: Graph ready to be invoked with a RAGGraphState.
        """
        workflow = StateGraph(RAGGraphState)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def ask_question(self, question: str, context: List[str]) -> str:
        """
        Answer a question using the configured Ollama model with FAISS in-memory vector store.

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

            # Build/load vectorstore if not already done
            if self.vectorstore is None:
                self.build_or_load_vectorstore(context)

            # Compile the graph once and reuse it across questions
            if self.graph is None:
                self.graph = self.build_graph()
                logging.debug("LangGraph RAG state graph compiled")

            # Run the graph
            final_state = self.graph.invoke({
                "question": question,
                "retrieved_docs": [],
                # Bumped by the CRAG reformulation loop once that node exists.
                "iteration_count": 0,
                "answer": None
            })
            answer = final_state.get("answer") or ""
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
