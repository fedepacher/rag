import logging
import os
import re
import unicodedata
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

# --- CRAG (Corrective RAG) ---------------------------------------------------

# The three verdicts the relevance grader can produce. Plain strings rather than an
# Enum because they travel inside the TypedDict LangGraph carries between nodes, and
# because the grader answers in Spanish with exactly these words.
RELEVANCE_RELEVANT = "relevante"
RELEVANCE_PARTIAL = "parcialmente_relevante"
RELEVANCE_IRRELEVANT = "irrelevante"

# Maximum number of iterations of the *correction loop*, i.e. of query reformulations.
#
# "Máximo 2 iteraciones" is genuinely ambiguous, so this is the reading we commit to:
# the first retrieval is not an iteration of the loop, it is what the loop corrects.
# Two iterations therefore means at most 2 reformulations, and the worst case for a
# single question is 3 retrievals + 3 relevance grades + 2 reformulations before the
# out-of-scope short-circuit fires. Sequence, with iteration_count starting at 0:
#   retrieve -> grade (irrelevante, 0 < 2) -> reformulate (count=1)
#            -> grade (irrelevante, 1 < 2) -> reformulate (count=2)
#            -> grade (irrelevante, 2 >= 2) -> out_of_scope
MAX_CRAG_ITERATIONS = 2

# Characters of each retrieved chunk handed to the grader.
#
# The previous value of 1500 was justified by a comment claiming chunks were "5000
# characters". They were 5000 TOKENS -- 15104 characters on average over the course
# bibliography -- so the grader was judging relevance, and Self-RAG was verifying
# grounding, against 9% of each chunk. That is why a fabricated claim could pass
# verification: the verifier had no sight of the material that would have contradicted it.
#
# Chunks are now CHUNK_SIZE_TOKENS (1200) tokens, ~3700 characters at the ~3.1 chars/token
# this corpus measures, so 3000 characters covers ~80% of a chunk. Budget against the
# grader's PHI_NUM_CTX of 6000: RETRIEVAL_K previews ~3900 tokens, plus the answer under
# verification (~500) and the instructions (~200), leaves headroom. Still a bound rather
# than the whole chunk, because three worst-case grading calls have to stay affordable on
# a CPU-only box.
GRADER_CHUNK_PREVIEW_CHARS = 3000

# Upper bound on a reformulated query. A rewrite longer than this is the model
# rambling rather than searching, and a bad query is worse than the original one.
REFORMULATED_QUERY_MAX_CHARS = 500

RELEVANCE_PROMPT = """
Eres un evaluador de relevancia de un sistema de búsqueda documental.
Tu única tarea es decidir si los FRAGMENTOS recuperados contienen información útil
para responder la PREGUNTA. No respondas la pregunta ni expliques tu decisión.

Responde con UNA sola palabra, exactamente una de estas tres:
- relevante: los fragmentos alcanzan para responder la pregunta.
- parcialmente_relevante: los fragmentos tratan el tema pero no alcanzan para responderla por completo.
- irrelevante: los fragmentos no tienen relación con la pregunta.

FRAGMENTOS:
{context}

PREGUNTA: {question}
Clasificación:
"""

REFORMULATION_PROMPT = """
Eres un asistente de búsqueda documental de un curso universitario de electrónica.
La búsqueda actual no recuperó fragmentos relevantes de la bibliografía del curso.
Reescribe la consulta para mejorar la búsqueda por similitud semántica: usa la
terminología técnica de la materia, explicita los conceptos clave y elimina el texto
conversacional. No respondas la pregunta.

Devuelve únicamente la consulta reescrita, en una sola línea, sin comillas ni explicaciones.

PREGUNTA ORIGINAL DEL ESTUDIANTE: {question}
BÚSQUEDA ACTUAL: {search_query}
CONSULTA REESCRITA:
"""

# Returned instead of a speculative answer once the correction loop is exhausted.
# Deliberately does not start with "Error:", which is the prefix the pipeline reserves
# for failures (resources/eval/run_baseline.py flags records by it): being out of scope
# is a valid, successful outcome, not a malfunction.
OUT_OF_SCOPE_ANSWER = (
    "No encontré información sobre esta consulta en la bibliografía del curso. "
    "La pregunta parece quedar fuera del alcance del material disponible, así que no "
    "puedo responderla sin especular. Te sugiero reformularla con los términos que usa "
    "la cátedra, o consultarla directamente con el docente."
)

# --- Self-RAG (grounding verification) ---------------------------------------

# The two verdicts the grounding verifier can produce. Plain strings for the same
# reasons as the relevance verdicts: they travel inside the TypedDict and the grader
# answers in Spanish with exactly these words.
GROUNDING_GROUNDED = "fundamentada"
GROUNDING_UNGROUNDED = "no_fundamentada"

# Total number of generation passes allowed per question, i.e. one first attempt plus
# MAX_GENERATION_ATTEMPTS - 1 grounding retries. Two is a deliberate compromise:
#
# * Going straight to the fallback on the first failed verification would let a single
#   3.8B grader veto an answer the classic pipeline would have sent. parse_grounding
#   already leans towards keeping the answer for that reason, and a false "no
#   fundamentada" on a correct answer is the expensive mistake here — the student gets
#   a refusal instead of the material.
# * Retrying blindly would be close to useless: the generator runs at
#   OLLAMA_TEMPERATURE = 0.0, so re-running the *same* prompt over the *same* chunks
#   reproduces the same text and burns a full Llama call on a CPU-only box for nothing.
#   The retry therefore uses GROUNDED_RETRY_PROMPT, a different prompt, which is what
#   makes a different answer possible at all at temperature 0.
# * More than one retry multiplies the slowest call in the pipeline. Llama is the
#   expensive model (Phi verification is cheap by comparison); if the stricter prompt
#   did not fix the grounding, a third pass is far more likely to cost another minute
#   than to succeed. Worst case per question stays at 2 Llama calls + 2 Phi checks.
MAX_GENERATION_ATTEMPTS = 2

# Characters of the generated answer handed to the verifier. Chunk previews already
# take GRADER_CHUNK_PREVIEW_CHARS * RETRIEVAL_K characters of the grader's num_ctx, and
# an answer that overflows it would push the *instructions* out of the window, which
# turns the verifier into a random verdict generator. Checking a truncated answer is
# strictly better than that.
GRADER_ANSWER_MAX_CHARS = 3000

# How much of a rejected draft is written to the log when Self-RAG withholds an answer.
#
# A refusal is the one outcome where the pipeline destroys its own evidence: the student
# gets UNGROUNDED_FALLBACK_ANSWER and the draft that was actually judged is discarded, so
# there is no way to tell afterwards whether the verifier caught a real fabrication or
# vetoed a good answer. Three of six questions refused on 2026-09-10 and none of them
# could be diagnosed from the logs. 1200 characters is enough to see which claim went
# beyond the sources without turning every refusal into a wall of log.
REJECTED_DRAFT_LOG_CHARS = 1200

# The escape hatch INITIAL_PROMPT and GROUNDED_RETRY_PROMPT force when the retrieved
# context does not answer the question. Matched normalised (lower case, punctuation and
# runs of whitespace collapsed) because the generator reproduces the sentence but not
# always its markdown or trailing punctuation.
NO_ANSWER_MARKER = "no se la respuesta basada en la informacion proporcionada"

# How much of a control model's raw reply is logged alongside the parsed verdict.
# parse_relevance and parse_grounding both fall back to a permissive value on anything
# they cannot read, so a silently unparseable reply looks exactly like a real verdict.
# Logging the raw text is what distinguishes "the model said parcialmente_relevante"
# from "the model rambled and the parser defaulted".
GRADER_RAW_LOG_CHARS = 200

GROUNDING_PROMPT = """
Eres un verificador de un sistema de preguntas y respuestas sobre documentos.
Tu única tarea es decidir si la RESPUESTA está respaldada por los FRAGMENTOS.
No respondas la pregunta, no la corrijas ni expliques tu decisión.

Criterios:
- fundamentada: todo lo que afirma la respuesta aparece en los fragmentos. Admitir que
  no se sabe la respuesta también cuenta como fundamentada.
- no_fundamentada: la respuesta agrega datos, cifras, fórmulas, nombres, definiciones o
  conclusiones que no aparecen en los fragmentos.

Responde con UNA sola palabra, exactamente una de estas dos:
fundamentada
no_fundamentada

FRAGMENTOS:
{context}

PREGUNTA: {question}

RESPUESTA A VERIFICAR:
{answer}

Clasificación:
"""

# Generation prompt used only on a grounding retry. Keeps INITIAL_PROMPT's contract
# (same variables, same "No sé la respuesta..." escape hatch) and adds the explicit
# grounding rules. The prompt has to differ from INITIAL_PROMPT for the retry to be
# worth its cost: see MAX_GENERATION_ATTEMPTS.
GROUNDED_RETRY_PROMPT = """
Eres un asistente experto que solo puede responder preguntas utilizando **únicamente** la información contenida en el documento a continuación.
Un intento anterior de responder esta pregunta fue descartado porque afirmaba cosas que no estaban en el documento.

Reglas estrictas:
- No uses conocimientos previos ni hagas inferencias que el documento no respalde.
- No agregues cifras, fórmulas, nombres, normas ni ejemplos que no aparezcan en el documento.
- Cada afirmación de tu respuesta debe poder señalarse en el documento.
- Prefiere una respuesta corta y verificable antes que una completa pero especulativa.
- Si la información no está presente explícitamente en el documento, responde únicamente:
**"No sé la respuesta basada en la información proporcionada."**

DOCUMENTO:
{context}

Pregunta: {question}
Respuesta:
"""

# Returned when the generated answer could not be verified against the retrieved chunks.
# Distinct from OUT_OF_SCOPE_ANSWER because the failure is a different one: out of scope
# means the bibliography does not cover the question, ungrounded means it may well cover
# it but what the generator wrote could not be traced back to it. Like OUT_OF_SCOPE_ANSWER
# it deliberately does not start with "Error:" — withholding an unverifiable answer is the
# safety mechanism working, not a pipeline failure.
UNGROUNDED_FALLBACK_ANSWER = (
    "No puedo responder esta consulta con el material disponible. Elaboré una respuesta "
    "pero no logré verificar que estuviera respaldada por la bibliografía del curso, y "
    "prefiero no enviarte información que podría ser incorrecta. Te sugiero reformular la "
    "pregunta siendo más específico, o consultarla directamente con el docente."
)

# --- Confidence level --------------------------------------------------------

# How much the pipeline itself trusts the answer it is about to send. Derived from the
# CRAG and Self-RAG signals already computed during the run — nothing new is measured
# and no extra model call is made, the levels are just a reading of the final graph
# state. Ordered from most to least trustworthy:
#
# * alta  - the first retrieval was graded `relevante` and the first generated answer
#           passed grounding verification. Nothing had to be corrected.
# * media - the answer is grounded, but the *retrieval* needed help: either the grader
#           only found `parcialmente_relevante` context (the "partial bibliography
#           coverage" case), or the student's own wording did not retrieve usable
#           chunks and a CRAG reformulation had to find them. Both mean the answer may
#           be incomplete even though everything it says is supported.
# * baja  - the answer is grounded, but only on a second generation pass: the verifier
#           rejected the first attempt and the strict retry prompt had to rescue it.
#           This is the only signal about the *answer text* rather than about the
#           retrieval, which is why it outranks the other two — on this exact question
#           the generator demonstrably drifted away from the sources once already.
CONFIDENCE_HIGH = "alta"
CONFIDENCE_MEDIUM = "media"
CONFIDENCE_LOW = "baja"

# Used for the two terminal fallbacks (OUT_OF_SCOPE_ANSWER, UNGROUNDED_FALLBACK_ANSWER).
# Those are not answers, they are refusals that already explain themselves, and labelling
# a refusal with a confidence level would read as if there were something to trust in it.
# No note is appended for this level, which also keeps both fallbacks byte-for-byte equal
# to their constants — resources/eval/run_baseline.py detects the out-of-scope
# short-circuit by comparing against OUT_OF_SCOPE_ANSWER verbatim.
CONFIDENCE_NOT_APPLICABLE = "no_aplica"

# Separator between the answer and its confidence note. A blank line plus a rule keeps
# the note visually detached in the plain-text email the student receives, and gives any
# caller that wants the bare answer back a single unambiguous split point.
CONFIDENCE_NOTE_SEPARATOR = "\n\n---\n"

# Student-facing wording per level. Written in the same register as OUT_OF_SCOPE_ANSWER
# and UNGROUNDED_FALLBACK_ANSWER: it states what the system did, not how sure it "feels".
# Each note says which signal produced the level, so a student who disagrees with it
# knows what to check.
# Carried by every level, including the highest. Issue #31: `alta` used to be the only
# note that gave no advice to check anything -- and it is the label both known-wrong
# answers carried. Reserving the advice for the levels where a check already reported
# trouble gets the logic exactly backwards: the checks report on retrieval and on
# provenance, and the failure that actually reached a student (a ratio stated inverted,
# perfectly traceable to the chunks) is invisible to all of them. There is no level at
# which this pipeline has established that an answer is right.
CONFIDENCE_ADVICE = (
    "Conviene contrastarla con el material de cátedra antes de darla por válida, o "
    "consultar al docente."
)

CONFIDENCE_NOTES = {
    CONFIDENCE_HIGH: (
        "Nivel de confianza: alta. La bibliografía del curso cubre la consulta y todo lo "
        "que afirma esta respuesta proviene del material recuperado. Eso verifica su "
        "procedencia, no su exactitud: una afirmación puede provenir del material y aun "
        f"así estar mal expresada o invertida. {CONFIDENCE_ADVICE}"
    ),
    CONFIDENCE_MEDIUM: (
        "Nivel de confianza: media. Lo que afirma esta respuesta proviene del material, "
        "pero la bibliografía cubre el tema solo parcialmente o hubo que reformular la "
        "búsqueda para encontrarlo, así que la respuesta puede estar incompleta. "
        f"{CONFIDENCE_ADVICE}"
    ),
    CONFIDENCE_LOW: (
        "Nivel de confianza: baja. Un primer intento de respuesta no pudo rastrearse "
        "hasta la bibliografía y hubo que reescribirlo con criterios más estrictos. "
        f"{CONFIDENCE_ADVICE}"
    )
}


class RAGGraphState(TypedDict):
    """State carried through the LangGraph RAG pipeline.

    Attributes:
        question (str): The student's question, verbatim and unchanged across the run.
            Generation and relevance grading always work against this, never against a
            machine-rewritten variant.
        search_query (str): Query actually sent to the retriever. Starts as a copy of
            ``question`` and is replaced by the CRAG reformulation node. Kept separate
            from ``question`` so a rewrite improves retrieval without changing what the
            student is answered.
        retrieved_docs (List[Document]): Chunks returned by the FAISS retriever.
        relevance (Optional[str]): Verdict of the CRAG grading node for the current
            chunks, one of RELEVANCE_RELEVANT, RELEVANCE_PARTIAL or
            RELEVANCE_IRRELEVANT. None before the first grade, and always None when the
            processor runs without a grader model.
        iteration_count (int): Number of query reformulations performed so far, capped
            at MAX_CRAG_ITERATIONS.
        answer (Optional[str]): Text produced by the generation, out-of-scope or
            ungrounded node, None until then.
        grounding (Optional[str]): Verdict of the Self-RAG verification node for the
            current answer, GROUNDING_GROUNDED or GROUNDING_UNGROUNDED. None before the
            first verification, and always None when the processor runs without a grader
            model. Also read by the generation node, which switches to
            GROUNDED_RETRY_PROMPT once a previous answer failed verification.
        generation_attempts (int): Number of generation passes performed so far, capped
            at MAX_GENERATION_ATTEMPTS. Deliberately *not* folded into
            ``iteration_count``: that one bounds the CRAG loop, which re-queries the
            retriever with a rewritten query, while this one bounds the Self-RAG loop,
            which regenerates from the *same* chunks. They are different loops with
            different costs, they can both run for the same question, and sharing a
            counter would silently let one starve the other.
    """

    question: str
    search_query: str
    retrieved_docs: List[Document]
    relevance: Optional[str]
    iteration_count: int
    answer: Optional[str]
    grounding: Optional[str]
    generation_attempts: int


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
    """Ollama-backed processor running its pipeline as a LangGraph state graph.

    The graph shape depends on whether a grader model was wired in:

    * ``grader_llm=None`` -> the linear pipeline ``retrieve -> generate``. This is the
      classic RAG behaviour the baseline harness measures, and it stays reachable on
      purpose so the pre-CRAG reference point does not disappear.
    * ``grader_llm`` set -> the CRAG correction loop, with relevance grading, bounded
      query reformulation and the out-of-scope short-circuit, followed by the Self-RAG
      grounding verification of the generated answer. Answers produced by this path
      also carry a confidence note derived from those same verdicts; see
      :meth:`derive_confidence`.

    Args:
        llm: Generation model (Llama 3.1 8B in production).
        embedding: Embedding model backing the FAISS vector store.
        context_length: Token budget used to chunk the course documents. Counted in
            cl100k_base tokens by TokenTextSplitter, not in characters.
        grader_llm: Small control model (Phi-3.5-mini in production) used by the CRAG
            and Self-RAG nodes to grade relevance, rewrite queries and verify that the
            answer is grounded. A single instance is shared by every control node; do
            not create one per node, both models stay resident.
    """

    def __init__(self, llm, embedding, context_length, grader_llm=None):
        super().__init__(llm, embedding, context_length)
        self.vectorstore = None
        self.qa_chain_prompt = PromptTemplate.from_template(INITIAL_PROMPT)
        self.grounded_retry_prompt = PromptTemplate.from_template(GROUNDED_RETRY_PROMPT)
        self.grader_llm = grader_llm
        self.relevance_prompt = PromptTemplate.from_template(RELEVANCE_PROMPT)
        self.reformulation_prompt = PromptTemplate.from_template(REFORMULATION_PROMPT)
        self.grounding_prompt = PromptTemplate.from_template(GROUNDING_PROMPT)
        self.graph = None

    def embedding_fingerprint(self) -> str:
        """Identify the embedding model that produced the stored vectors.

        Returns:
            str: Class name and model identifier, e.g. "OllamaEmbeddings:bge-m3". The
            attribute is read defensively because the embedding classes do not agree on
            a name for it (``model`` on OllamaEmbeddings, ``model_name`` elsewhere) and
            some expose neither; an unidentifiable embedding hashes as "?" and simply
            stops distinguishing that case, which is no worse than not hashing it.
        """
        model = getattr(self.embedding, "model", None) or getattr(self.embedding, "model_name", None)
        return f"{type(self.embedding).__name__}:{model or '?'}"

    def get_index_hash(self, context: list[str]) -> str:
        """Fingerprint everything the stored index depends on.

        Covers the chunk text AND the embedding model, because a FAISS index is only
        reusable when both are unchanged. Hashing the text alone -- which is what this
        did originally -- means swapping the embedding model silently loads vectors
        from the previous one: at best retrieval degrades to nonsense while looking
        healthy, at worst the dimensions disagree outright (GPT4AllEmbeddings produces
        384, bge-m3 produces 1024) and the failure surfaces far from its cause. The
        index is currently unmounted so a rebuilt image wipes it anyway, but that is an
        accident of the missing volume, not a guarantee -- and the day the volume is
        added, this is the bug that would appear.

        Args:
            context (list[str]): The chunks that would be indexed.

        Returns:
            str: Hex SHA-256 over the embedding fingerprint and the joined chunks.
        """
        joined = ''.join(context)
        return hashlib.sha256(f"{self.embedding_fingerprint()}\n{joined}".encode('utf-8')).hexdigest()

    def document_has_changed(self, context: list[str]) -> bool:
        new_hash = self.get_index_hash(context)
        if os.path.exists(DOC_HASH_PATH):
            with open(DOC_HASH_PATH, 'r') as f:
                old_hash = f.read().strip()
            return new_hash != old_hash
        return True

    def save_hash(self, context: list[str]):
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        with open(DOC_HASH_PATH, 'w') as f:
            f.write(self.get_index_hash(context))

    def build_or_load_vectorstore(self, context: list[str]):
        if os.path.exists(FAISS_INDEX_PATH) and not self.document_has_changed(context):
            # allow_dangerous_deserialization is required because a FAISS index is a
            # pickle, and langchain-community >= 0.0.27 refuses to unpickle one unless
            # the caller vouches for its origin. This index is never downloaded: the
            # else-branch below writes it with save_local from the course documents in
            # DOCUMENT_LOCATION, and document_has_changed gates this load path on a
            # SHA-256 of that same content. The file we are trusting is the one this
            # process produced.
            self.vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH, self.embedding, allow_dangerous_deserialization=True
            )
        else:
            if os.path.exists(FAISS_INDEX_PATH):
                shutil.rmtree(FAISS_INDEX_PATH, ignore_errors=True)
            self.vectorstore = FAISS.from_texts(texts=context, embedding=self.embedding)
            self.vectorstore.save_local(FAISS_INDEX_PATH)
            self.save_hash(context)

    def retrieve_node(self, state: RAGGraphState) -> Dict[str, Any]:
        """Graph node: pull the most similar chunks for the current query out of FAISS.

        Re-entered by the CRAG correction loop, in which case ``search_query`` holds the
        reformulated query rather than the student's original wording.

        Args:
            state (RAGGraphState): Current graph state, read for its search query.

        Returns:
            Dict[str, Any]: State update carrying the retrieved documents.
        """
        search_query = state.get("search_query") or state["question"]
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        retrieved_docs = retriever.invoke(search_query)
        logging.debug(f"Retrieved {len(retrieved_docs)} chunk(s) from FAISS (k={RETRIEVAL_K}) "
                      f"for query: {search_query}")
        # Which documents were reached is the single most useful retrieval signal, and it
        # is free: the loader already prefixes every chunk with its source. Logged at INFO
        # rather than DEBUG because a refusal cannot be diagnosed without it -- knowing the
        # verifier rejected an answer says nothing until you know it was written from the
        # noise chapter instead of the FET chapter.
        logging.info(f"Retrieved from: {', '.join(self.chunk_source(document) for document in retrieved_docs)}")
        return {"retrieved_docs": retrieved_docs}

    @staticmethod
    def chunk_source(document: Document) -> str:
        """Recover the source filename a chunk was tagged with by the loader.

        The vector store holds plain strings, so the provenance lives in the text itself:
        ``document_loader.SOURCE_HEADER`` prefixes every chunk with ``[Fuente: <file>]``.
        Reading it back from ``page_content`` rather than from ``metadata`` is not a
        preference, it is the only option until the loader is changed to build Document
        objects with metadata.

        Args:
            document (Document): Chunk returned by the retriever.

        Returns:
            str: The source filename, or "?" when the header is absent — which is what a
            chunk written by an older index build looks like, and is worth seeing in the
            log rather than crashing retrieval over.
        """
        match = re.match(r"\[Fuente: (.+?)\]", document.page_content)
        return match.group(1) if match else "?"

    @staticmethod
    def parse_relevance(grader_response: Any) -> str:
        """Map a free-form grader answer onto one of the three relevance verdicts.

        Small instruct models do not reliably answer with a bare label: they prepend
        "Clasificación:", wrap the word in markdown, or answer in a full sentence. The
        matching is therefore keyword based, and the order matters because "irrelevante"
        contains "relevante" as a substring and "no son relevantes" contains it too.

        Args:
            grader_response (Any): Raw value returned by the grader LLM.

        Returns:
            str: One of RELEVANCE_RELEVANT, RELEVANCE_PARTIAL or RELEVANCE_IRRELEVANT.
            An empty or unparseable answer falls back to RELEVANCE_PARTIAL, the verdict
            that keeps the pipeline generating: a confused grader must never be able to
            turn a question into an out-of-scope refusal that the classic pipeline would
            have answered.
        """
        text = re.sub(r"[\s_\-*`.:,;]+", " ", str(grader_response or "").lower()).strip()
        if not text:
            logging.warning("Relevance grader returned an empty answer; assuming partial relevance")
            return RELEVANCE_PARTIAL
        if "parcial" in text:
            return RELEVANCE_PARTIAL
        if "irrelevante" in text or re.search(r"\bno\s+(?:\w+\s+){0,3}relevante", text):
            return RELEVANCE_IRRELEVANT
        if "relevante" in text:
            return RELEVANCE_RELEVANT
        logging.warning(f"Unparseable relevance verdict '{text[:80]}'; assuming partial relevance")
        return RELEVANCE_PARTIAL

    @staticmethod
    def sanitize_reformulated_query(llm_response: Any) -> str:
        """Extract a usable search query from the reformulation model's answer.

        Args:
            llm_response (Any): Raw value returned by the grader LLM.

        Returns:
            str: First usable line, stripped of quoting and of the label the model
            sometimes echoes back, capped at REFORMULATED_QUERY_MAX_CHARS. Empty string
            when nothing usable came back, which the caller treats as "keep the previous
            query".
        """
        for line in str(llm_response or "").splitlines():
            candidate = line.strip().strip('"\'*` ')
            candidate = re.sub(r"^(?:consulta|pregunta|b[uú]squeda)[^:]{0,30}:\s*", "", candidate,
                               flags=re.IGNORECASE).strip()
            if candidate:
                return candidate[:REFORMULATED_QUERY_MAX_CHARS]
        return ""

    def grade_relevance_node(self, state: RAGGraphState) -> Dict[str, Any]:
        """Graph node: ask the control model whether the retrieved chunks are usable.

        Grades the chunks against the student's original question, not against the
        reformulated search query: the point is whether what came back can answer what
        was actually asked.

        Args:
            state (RAGGraphState): Current graph state, read for its question and
                retrieved documents.

        Returns:
            Dict[str, Any]: State update carrying the relevance verdict.
        """
        retrieved_docs = state["retrieved_docs"]
        if not retrieved_docs:
            logging.warning("Relevance grading skipped: retrieval returned no chunk")
            return {"relevance": RELEVANCE_IRRELEVANT}

        context = DOCUMENT_SEPARATOR.join(document.page_content[:GRADER_CHUNK_PREVIEW_CHARS]
                                          for document in retrieved_docs)
        prompt = self.relevance_prompt.format(context=context, question=state["question"])

        logging.debug("Waiting for the relevance grader")
        try:
            grader_response = self.grader_llm.invoke(prompt)
        except Exception as err:
            # The control model is an enhancement, not a dependency: if it is down the
            # pipeline degrades to classic RAG instead of failing the student's question.
            logging.error(f"Relevance grading failed, falling back to generation: {err}", exc_info=True)
            return {"relevance": RELEVANCE_PARTIAL}

        relevance = self.parse_relevance(grader_response)
        logging.info(f"CRAG relevance verdict: {relevance} (iteration {state.get('iteration_count', 0)})")
        logging.debug(f"Relevance grader said verbatim: "
                      f"{str(grader_response)[:GRADER_RAW_LOG_CHARS]!r}")
        return {"relevance": relevance}

    def reformulate_query_node(self, state: RAGGraphState) -> Dict[str, Any]:
        """Graph node: rewrite the search query after an irrelevant retrieval.

        Increments ``iteration_count`` unconditionally, including when the rewrite
        fails, so that the correction loop is bounded by construction and cannot spin.

        Resets ``generation_attempts`` for the same structural reason, and because the
        two counters bound different things: ``generation_attempts`` bounds regeneration
        *over one set of chunks*, which is why the Self-RAG retry swaps in a stricter
        prompt rather than re-running the same one. A reformulation replaces those
        chunks, so the next generation is a first attempt, not a retry. Carrying the
        count across would let one escape-hatch reformulation consume the grounding
        retry, leaving a genuinely ungrounded answer on the following round withheld
        with no second chance -- CRAG starving Self-RAG through a shared budget instead
        of a shared counter. It would also mislabel the result: ``derive_confidence``
        reads ``generation_attempts > 1`` as "the verifier rejected a draft", which on a
        merely reformulated question never happened.

        Args:
            state (RAGGraphState): Current graph state, read for its question, current
                search query and iteration count.

        Returns:
            Dict[str, Any]: State update carrying the new query, the iteration count and
            a reset generation budget.
        """
        search_query = state.get("search_query") or state["question"]
        iteration_count = state.get("iteration_count", 0) + 1
        prompt = self.reformulation_prompt.format(question=state["question"], search_query=search_query)

        logging.debug("Waiting for the query reformulation")
        try:
            reformulated_query = self.sanitize_reformulated_query(self.grader_llm.invoke(prompt))
        except Exception as err:
            logging.error(f"Query reformulation failed: {err}", exc_info=True)
            reformulated_query = ""

        if not reformulated_query:
            logging.warning(f"Reformulation {iteration_count}/{MAX_CRAG_ITERATIONS} produced nothing usable; "
                            f"keeping the previous query")
            return {"iteration_count": iteration_count, "generation_attempts": 0}

        logging.info(f"CRAG reformulation {iteration_count}/{MAX_CRAG_ITERATIONS}: "
                     f"'{search_query}' -> '{reformulated_query}'")
        return {"search_query": reformulated_query,
                "iteration_count": iteration_count,
                "generation_attempts": 0}

    def out_of_scope_node(self, state: RAGGraphState) -> Dict[str, Any]:
        """Graph node: answer that the question is not covered by the bibliography.

        Terminal alternative to generation. Reached only once the correction loop is
        exhausted, so that an exhausted question gets an honest refusal instead of a
        speculative answer built on chunks the grader already rejected.

        Args:
            state (RAGGraphState): Current graph state, read for its question.

        Returns:
            Dict[str, Any]: State update carrying the out-of-scope answer.
        """
        logging.info(f"CRAG exhausted {MAX_CRAG_ITERATIONS} reformulation(s) without relevant context; "
                     f"answering out of scope for: {state['question']}")
        # Pin the verdict this node's own existence implies. derive_confidence reads the
        # terminal state to tell a refusal from an answer, on the invariant "terminal
        # irrelevante <=> out_of_scope fired". Since route_after_generation can now reach
        # this node with a relevante verdict on the state, the invariant has to be
        # asserted by the node that ends the run rather than inferred from the last
        # grader verdict to survive. Reached from the grading loop this is a no-op.
        return {"answer": OUT_OF_SCOPE_ANSWER, "relevance": RELEVANCE_IRRELEVANT}

    @staticmethod
    def is_no_answer(answer: Any) -> bool:
        """Report whether the generator used the "I don't know" escape hatch.

        Both generation prompts instruct the model to answer *only*
        ``"No sé la respuesta basada en la información proporcionada."`` when the
        retrieved context does not contain the answer, so this sentence is not an
        answer at all: it is the generator reporting that retrieval failed.

        Args:
            answer (Any): Text produced by the generation node.

        Returns:
            bool: True when the answer opens with the escape hatch. Anchored at the
            start rather than searched anywhere in the text, because a real answer
            that happens to quote the phrase mid-paragraph is still a real answer.
        """
        normalised = unicodedata.normalize("NFKD", str(answer or "").lower())
        stripped = "".join(char for char in normalised if not unicodedata.combining(char))
        collapsed = re.sub(r"[\s*`_\"'.:,;¿?¡!-]+", " ", stripped).strip()
        return collapsed.startswith(NO_ANSWER_MARKER)

    def route_after_generation(self, state: RAGGraphState) -> str:
        """Conditional edge: decide whether the generated text is worth verifying.

        An answer that used the escape hatch does not need Self-RAG: the generator has
        already reported that the retrieved chunks do not answer the question, having
        read them in full rather than through the grader's
        ``GRADER_CHUNK_PREVIEW_CHARS`` window. Sending it to verification wastes the two
        most expensive calls in the pipeline to arrive at the refusal it already is.

        Measured on 2026-09-10 before this edge existed: Phi-3.5-mini graded that exact
        sentence ``no_fundamentada`` on both attempts of both refusing questions, in
        direct contradiction of GROUNDING_PROMPT, which states that admitting ignorance
        counts as grounded. Each of those questions then burned a second Llama
        generation and a second verification — R2 took 28 minutes and R5 took 30 — to
        reach a refusal that was already correct after the first pass. Routing on the
        generator's own words is deterministic and costs nothing, where relying on a
        3.8B model to honour a documented instruction demonstrably does not hold.

        What the escape hatch reports is a *retrieval* failure, so while the correction
        loop still has budget the honest response is to correct retrieval rather than to
        give up: route to ``reformulate_query`` and let a rewritten query try again.
        This is the only edge that actually exercises the CRAG loop in practice. The
        relevance grader has never once returned ``irrelevante`` on the recorded runs,
        so before this edge existed the loop was unreachable code -- and a question whose
        answer *was* in the corpus, reached by a differently phrased question minutes
        later, was refused outright (issue #29).

        Out of scope rather than ungrounded is the honest label once the budget is spent:
        "the material does not cover this" is precisely what the generator said, whereas
        ungrounded means an answer exists but could not be traced.

        Args:
            state (RAGGraphState): Current graph state, read for its generated answer
                and iteration count.

        Returns:
            str: "reformulate_query", "out_of_scope" or "verify_grounding".
        """
        if not self.is_no_answer(state.get("answer")):
            return "verify_grounding"
        iteration_count = state.get("iteration_count", 0)
        if iteration_count < MAX_CRAG_ITERATIONS:
            logging.info(f"Generator reported the context does not answer the question; "
                         f"correcting retrieval instead of refusing "
                         f"(reformulation {iteration_count + 1}/{MAX_CRAG_ITERATIONS})")
            return "reformulate_query"
        logging.info("Generator reported the context does not answer the question and the "
                     "reformulation budget is spent; skipping grounding verification and "
                     "answering out of scope")
        return "out_of_scope"

    def route_after_grading(self, state: RAGGraphState) -> str:
        """Conditional edge: pick the successor of the relevance grading node.

        Only ``irrelevante`` opens the correction loop. ``parcialmente_relevante`` goes
        straight to generation on purpose: partial relevance means the bibliography does
        cover the topic, INITIAL_PROMPT already forces the model to answer "No sé la
        respuesta basada en la información proporcionada." for whatever is missing, and
        reformulating from a partial hit risks trading a usable answer for an
        out-of-scope refusal the classic pipeline would never have produced. Each loop
        also costs a grade, a rewrite and a retrieval on a CPU-only box, which is only
        worth paying when the retrieved context is outright useless.

        Args:
            state (RAGGraphState): Current graph state, read for its relevance verdict
                and iteration count.

        Returns:
            str: Name of the next node: "generate", "reformulate_query" or "out_of_scope".
        """
        if (state.get("relevance") or RELEVANCE_PARTIAL) != RELEVANCE_IRRELEVANT:
            return "generate"
        if state.get("iteration_count", 0) < MAX_CRAG_ITERATIONS:
            return "reformulate_query"
        return "out_of_scope"

    def generate_node(self, state: RAGGraphState) -> Dict[str, Any]:
        """Graph node: stuff the retrieved chunks into the prompt and call the LLM.

        Re-entered by the Self-RAG loop when the previous answer failed grounding
        verification, in which case the stricter GROUNDED_RETRY_PROMPT is used instead of
        INITIAL_PROMPT: the generator runs at temperature 0, so re-sending the same
        prompt would reproduce the same answer and the retry would be pure cost.

        Increments ``generation_attempts`` unconditionally, so the Self-RAG loop is
        bounded by construction and cannot spin.

        Args:
            state (RAGGraphState): Current graph state, read for its question, retrieved
                documents and previous grounding verdict.

        Returns:
            Dict[str, Any]: State update carrying the generated answer and the attempt
            count.
        """
        question = state["question"]
        retrieved_docs = state["retrieved_docs"]
        generation_attempts = state.get("generation_attempts", 0) + 1
        context = DOCUMENT_SEPARATOR.join(document.page_content for document in retrieved_docs)

        if state.get("grounding") == GROUNDING_UNGROUNDED:
            logging.info(f"Self-RAG regeneration {generation_attempts}/{MAX_GENERATION_ATTEMPTS} "
                         f"with the strict grounding prompt")
            prompt = self.grounded_retry_prompt.format(context=context, question=question)
        else:
            prompt = self.qa_chain_prompt.format(context=context, question=question)

        logging.debug("Waiting for LLM response")
        llm_response = self.llm.invoke(prompt)
        answer = llm_response if isinstance(llm_response, str) else str(llm_response)
        # The only marker between the relevance verdict and the grounding verdict. Without
        # it those two timestamps bound `generate` and `verify_grounding` together, which
        # is what made the 817-1091 s block in Run E unsplittable and left #34 -- "does the
        # control model really cost more than the generator" -- unanswerable. Length rather
        # than the text: it makes a runaway visible in the log without dumping the answer,
        # and a runaway is what cost 2 h 39 min before num_predict was enforced.
        logging.info(f"Generation complete (attempt {generation_attempts}/{MAX_GENERATION_ATTEMPTS}), "
                     f"{len(answer)} chars")
        return {"answer": answer, "generation_attempts": generation_attempts}

    @staticmethod
    def parse_grounding(grader_response: Any) -> str:
        """Map a free-form verifier answer onto one of the two grounding verdicts.

        Same defensive treatment as :meth:`parse_relevance`, and the same substring trap:
        "no_fundamentada" contains "fundamentada", so the negative form has to be tested
        first. The normalisation turns underscores and hyphens into spaces, which is what
        lets a single pattern cover "no_fundamentada", "no fundamentada" and "no está
        fundamentada".

        Args:
            grader_response (Any): Raw value returned by the grader LLM.

        Returns:
            str: GROUNDING_GROUNDED or GROUNDING_UNGROUNDED. An empty or unparseable
            answer falls back to GROUNDING_GROUNDED, the verdict that keeps the answer:
            a confused verifier must never be able to withhold an answer the classic
            pipeline would have sent. Self-RAG is a safety net, and a torn net has to
            let the pipeline through rather than block it.
        """
        text = re.sub(r"[\s_\-*`.:,;]+", " ", str(grader_response or "").lower()).strip()
        if not text:
            logging.warning("Grounding verifier returned an empty answer; assuming the answer is grounded")
            return GROUNDING_GROUNDED
        if "infundad" in text or re.search(r"\bno\s+(?:\w+\s+){0,3}fundamentada", text):
            return GROUNDING_UNGROUNDED
        if "fundamentada" in text:
            return GROUNDING_GROUNDED
        logging.warning(f"Unparseable grounding verdict '{text[:80]}'; assuming the answer is grounded")
        return GROUNDING_GROUNDED

    def verify_grounding_node(self, state: RAGGraphState) -> Dict[str, Any]:
        """Graph node: ask the control model whether the answer is supported by the chunks.

        This is the Self-RAG component: it does not judge whether the answer is *good*,
        only whether every claim in it can be traced back to the retrieved context. An
        answer that admits it does not know is grounded by definition — INITIAL_PROMPT
        forces exactly that wording, and flagging it would spend two Llama calls to
        replace one honest refusal with another.

        Args:
            state (RAGGraphState): Current graph state, read for its question, retrieved
                documents and generated answer.

        Returns:
            Dict[str, Any]: State update carrying the grounding verdict.
        """
        answer = state.get("answer") or ""
        retrieved_docs = state["retrieved_docs"]
        if not answer.strip() or not retrieved_docs:
            # Unreachable in the CRAG graph (an empty retrieval is graded irrelevant and
            # short-circuits before generation), but a verifier call with nothing to
            # compare against would only produce noise.
            logging.warning("Grounding verification skipped: no answer or no retrieved chunk")
            return {"grounding": GROUNDING_GROUNDED}

        context = DOCUMENT_SEPARATOR.join(document.page_content[:GRADER_CHUNK_PREVIEW_CHARS]
                                          for document in retrieved_docs)
        prompt = self.grounding_prompt.format(context=context,
                                              question=state["question"],
                                              answer=answer[:GRADER_ANSWER_MAX_CHARS])

        logging.debug("Waiting for the grounding verifier")
        try:
            grader_response = self.grader_llm.invoke(prompt)
        except Exception as err:
            # Same contract as relevance grading: the control model is an enhancement,
            # not a dependency. If it is down the answer ships unverified rather than
            # the student getting a refusal for an infrastructure problem.
            logging.error(f"Grounding verification failed, keeping the answer: {err}", exc_info=True)
            return {"grounding": GROUNDING_GROUNDED}

        grounding = self.parse_grounding(grader_response)
        logging.info(f"Self-RAG grounding verdict: {grounding} "
                     f"(attempt {state.get('generation_attempts', 0)}/{MAX_GENERATION_ATTEMPTS})")
        if grounding == GROUNDING_UNGROUNDED:
            # The only place the rejected draft still exists. One more failed attempt and
            # ungrounded_node replaces it with UNGROUNDED_FALLBACK_ANSWER, after which
            # nothing records what was actually judged -- so a refusal caused by a real
            # fabrication and one caused by an over-strict 3.8B verifier look identical
            # from the outside. Logged at WARNING because a withheld answer is the outcome
            # most worth reviewing by hand, whatever the verdict turns out to be.
            logging.warning(f"Rejected draft (attempt {state.get('generation_attempts', 0)}), "
                            f"verifier said {str(grader_response)[:GRADER_RAW_LOG_CHARS]!r}, "
                            f"draft was: {answer[:REJECTED_DRAFT_LOG_CHARS]}")
        return {"grounding": grounding}

    def ungrounded_node(self, state: RAGGraphState) -> Dict[str, Any]:
        """Graph node: withhold an answer that could not be grounded in the material.

        Terminal alternative to shipping the generated text. Reached only after every
        allowed generation attempt failed verification, so the student gets an explicit
        "I cannot answer this with the available material" instead of content the system
        itself could not trace back to the bibliography.

        Args:
            state (RAGGraphState): Current graph state, read for its question.

        Returns:
            Dict[str, Any]: State update carrying the ungrounded fallback answer.
        """
        logging.warning(f"Self-RAG could not ground the answer after {MAX_GENERATION_ATTEMPTS} attempt(s); "
                        f"withholding it for: {state['question']}")
        return {"answer": UNGROUNDED_FALLBACK_ANSWER}

    def route_after_grounding(self, state: RAGGraphState) -> str:
        """Conditional edge: pick the successor of the grounding verification node.

        A grounded answer ends the run. An ungrounded one is regenerated once with the
        strict prompt, and if that fails too the fallback answer is sent: retrying more
        would multiply the slowest call in the pipeline for a case that already resisted
        the strictest prompt available.

        Args:
            state (RAGGraphState): Current graph state, read for its grounding verdict
                and generation attempt count.

        Returns:
            str: END, or the name of the next node: "generate" or "ungrounded".
        """
        if (state.get("grounding") or GROUNDING_GROUNDED) == GROUNDING_GROUNDED:
            return END
        if state.get("generation_attempts", 0) < MAX_GENERATION_ATTEMPTS:
            return "generate"
        return "ungrounded"

    @staticmethod
    def derive_confidence(state: RAGGraphState) -> str:
        """Read a confidence level out of the final graph state.

        Pure function of the CRAG and Self-RAG signals the run already produced: no
        extra model call, no extra measurement. It is evaluated once on the *final*
        state rather than inside a graph node, which is what makes each check exact —
        both loops overwrite their own verdict on every pass, so a verdict that
        survives to the end can only have come from the node that ended the run:

        * ``relevance == RELEVANCE_IRRELEVANT`` at the end means ``out_of_scope`` fired.
          Any earlier ``irrelevante`` is followed by a reformulation and a fresh grade.
        * ``grounding == GROUNDING_UNGROUNDED`` at the end means ``ungrounded`` fired.
          Any earlier ``no_fundamentada`` is followed by a regeneration and a fresh
          verification.

        Checking the state instead of comparing the answer against OUT_OF_SCOPE_ANSWER
        and UNGROUNDED_FALLBACK_ANSWER also means rewording either constant cannot
        silently start labelling refusals as answers.

        Args:
            state (RAGGraphState): Final state returned by ``graph.invoke``.

        Returns:
            str: CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW, or
            CONFIDENCE_NOT_APPLICABLE for the two terminal fallbacks. Anything the
            checks cannot place — a missing relevance verdict, which the CRAG graph
            cannot actually produce since grading always precedes generation — falls
            back to CONFIDENCE_MEDIUM. The unknown case must never be advertised as
            high confidence; only an explicit ``relevante`` verdict earns that.
        """
        relevance = state.get("relevance")
        grounding = state.get("grounding")
        if relevance == RELEVANCE_IRRELEVANT or grounding == GROUNDING_UNGROUNDED:
            return CONFIDENCE_NOT_APPLICABLE
        # More than one generation pass means the verifier rejected the first answer.
        # Compared against 1 rather than against MAX_GENERATION_ATTEMPTS because the
        # fact of interest is "the answer had to be rewritten", not "the cap was hit".
        if state.get("generation_attempts", 1) > 1:
            return CONFIDENCE_LOW
        if relevance == RELEVANCE_RELEVANT and state.get("iteration_count", 0) == 0:
            return CONFIDENCE_HIGH
        return CONFIDENCE_MEDIUM

    @staticmethod
    def annotate_with_confidence(answer: str, confidence: str) -> str:
        """Append the student-facing confidence note to an answer.

        Args:
            answer (str): Answer produced by the graph.
            confidence (str): Level returned by :meth:`derive_confidence`.

        Returns:
            str: The answer with its note appended, or the answer untouched when the
            level has no note — CONFIDENCE_NOT_APPLICABLE, an empty answer, or an
            unknown level. Returning the input unchanged is the safe degradation here:
            a missing note costs the student a hint, a wrong one costs them trust.
        """
        note = CONFIDENCE_NOTES.get(confidence)
        if not note or not answer.strip():
            return answer
        return f"{answer.rstrip()}{CONFIDENCE_NOTE_SEPARATOR}{note}"

    def build_graph(self):
        """Compile the RAG state graph, with or without the agentic nodes.

        Without a grader model the graph is the linear pipeline the LangGraph migration
        introduced, which is what the baseline harness measures. Generation is terminal:
        no relevance grading and no grounding verification, exactly as before::

            START -> retrieve -> generate -> END

        With one, both the CRAG correction loop and the Self-RAG grounding verification
        are wired in::

            START -> retrieve -> grade_relevance
            grade_relevance -[relevante | parcialmente_relevante]-> generate
            grade_relevance -[irrelevante, iteration_count <  MAX_CRAG]-> reformulate_query -> retrieve
            grade_relevance -[irrelevante, iteration_count >= MAX_CRAG]-> out_of_scope -> END
            generate -[escape hatch, iteration_count <  MAX_CRAG]-> reformulate_query -> retrieve
            generate -[escape hatch, iteration_count >= MAX_CRAG]-> out_of_scope -> END
            generate -[real answer]-> verify_grounding
            verify_grounding -[fundamentada]-> END
            verify_grounding -[no_fundamentada, generation_attempts <  MAX_GEN]-> generate
            verify_grounding -[no_fundamentada, generation_attempts >= MAX_GEN]-> ungrounded -> END

        Both loops are bounded by the node that closes them incrementing its own counter
        unconditionally: ``reformulate_query`` for ``iteration_count`` and ``generate``
        for ``generation_attempts``. The counters stay separate, and ``reformulate_query``
        additionally resets ``generation_attempts``, so a question that spent its
        reformulations still gets its grounding retry on the chunks it ended up with.

        ``iteration_count`` is what bounds the whole graph: it is incremented only by
        ``reformulate_query`` and never reset, so there are at most MAX_CRAG + 1 = 3
        retrieval rounds however the loop is entered, and at most MAX_GEN = 2 generations
        inside each -- an arithmetic ceiling of 20 nodes. Driving both loops to
        exhaustion on one question with an adversarial stub measures 18 super-steps, 6
        generations and 8 control calls (see
        ``test_the_worst_case_walk_stays_inside_the_default_recursion_limit``). That is
        inside LangGraph's default recursion limit of 25, but no longer by much: raising
        either cap needs ``recursion_limit`` raised with it, and the test re-measured.

        Returns:
            CompiledStateGraph: Graph ready to be invoked with a RAGGraphState.
        """
        workflow = StateGraph(RAGGraphState)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_edge(START, "retrieve")

        if self.grader_llm is None:
            workflow.add_edge("retrieve", "generate")
            workflow.add_edge("generate", END)
            return workflow.compile()

        workflow.add_node("grade_relevance", self.grade_relevance_node)
        workflow.add_node("reformulate_query", self.reformulate_query_node)
        workflow.add_node("out_of_scope", self.out_of_scope_node)
        workflow.add_node("verify_grounding", self.verify_grounding_node)
        workflow.add_node("ungrounded", self.ungrounded_node)
        workflow.add_edge("retrieve", "grade_relevance")
        workflow.add_conditional_edges("grade_relevance", self.route_after_grading, {
            "generate": "generate",
            "reformulate_query": "reformulate_query",
            "out_of_scope": "out_of_scope"
        })
        workflow.add_edge("reformulate_query", "retrieve")
        workflow.add_edge("out_of_scope", END)
        workflow.add_conditional_edges("generate", self.route_after_generation, {
            "verify_grounding": "verify_grounding",
            "reformulate_query": "reformulate_query",
            "out_of_scope": "out_of_scope"
        })
        workflow.add_conditional_edges("verify_grounding", self.route_after_grounding, {
            END: END,
            "generate": "generate",
            "ungrounded": "ungrounded"
        })
        workflow.add_edge("ungrounded", END)
        return workflow.compile()

    def ask_question(self, question: str, context: List[str]) -> str:
        """
        Answer a question using the configured Ollama model with FAISS in-memory vector store.

        Args:
            question (str): The question to answer.
            context (List[str]): List of text chunks to use as context.

        Returns:
            str: The answer or an error message if processing fails. When the CRAG/
            Self-RAG nodes are wired in, a confidence note is appended to the answer
            text; see :meth:`derive_confidence`. The return type stays ``str`` on
            purpose — the API, the email delivery loop and
            ``resources/eval/run_baseline.py`` all consume this value directly, and the
            confidence level is meant for the student reading the email, who never sees
            anything but this string.
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
                logging.debug(f"LangGraph RAG state graph compiled "
                              f"(CRAG/Self-RAG {'enabled' if self.grader_llm is not None else 'disabled'})")

            # Run the graph
            final_state = self.graph.invoke({
                "question": question,
                # Retrieval starts from the student's own wording; the CRAG
                # reformulation node is what replaces it on later iterations.
                "search_query": question,
                "retrieved_docs": [],
                "relevance": None,
                "iteration_count": 0,
                "answer": None,
                "grounding": None,
                "generation_attempts": 0
            })
            answer = final_state.get("answer") or ""

            # Confidence is a property of the CRAG/Self-RAG signals, so it only exists
            # on the agentic path. The classic graph has no relevance verdict and no
            # grounding verdict to derive one from, and it is what the baseline harness
            # measures, so it must keep returning the generator's text untouched.
            if self.grader_llm is not None:
                confidence = self.derive_confidence(final_state)
                logging.info(f"Answer confidence level: {confidence} "
                             f"(relevance={final_state.get('relevance')}, "
                             f"reformulations={final_state.get('iteration_count', 0)}, "
                             f"grounding={final_state.get('grounding')}, "
                             f"generations={final_state.get('generation_attempts', 0)})")
                answer = self.annotate_with_confidence(answer, confidence)

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
