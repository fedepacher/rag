"""Routing tests for the CRAG/Self-RAG state graph.

Every function exercised here is pure: it reads a ``RAGGraphState`` dict and returns a
node name, a state update or a confidence level. None of them calls a model, touches
FAISS or reads the corpus, so the whole module runs in milliseconds against the same
acceptance criteria that otherwise need a 1h47m pipeline run on the target box.

``LLMProcessorOllama.__init__`` only compiles PromptTemplates, so a processor can be
built with ``llm=None`` and a stub grader without reaching the network.
"""
import pytest

from rag.llm_processor import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_NOT_APPLICABLE,
    GROUNDING_GROUNDED,
    GROUNDING_UNGROUNDED,
    MAX_CRAG_ITERATIONS,
    MAX_GENERATION_ATTEMPTS,
    OUT_OF_SCOPE_ANSWER,
    UNGROUNDED_FALLBACK_ANSWER,
    RELEVANCE_IRRELEVANT,
    RELEVANCE_PARTIAL,
    RELEVANCE_RELEVANT,
    LLMProcessorOllama,
)

ESCAPE_HATCH = "No sé la respuesta basada en la información proporcionada."
REAL_ANSWER = "La puerta no maneja virtualmente corriente, salvo alguna corriente de fuga."


class StubGraderLLM:
    """Minimal stand-in for the Phi control model."""

    def __init__(self, response="impedancia de entrada de un FET", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return self.response


def build_processor(grader=None):
    """Build a processor with no LLM and no embedding — enough for the pure methods."""
    return LLMProcessorOllama(llm=None,
                              embedding=None,
                              context_length=1200,
                              grader_llm=grader if grader is not None else StubGraderLLM())


# --- is_no_answer: the trigger the whole feature hangs off -----------------------

@pytest.mark.parametrize("answer", [
    ESCAPE_HATCH,
    "no se la respuesta basada en la informacion proporcionada",
    '**"No sé la respuesta basada en la información proporcionada."**',
    "  No sé la respuesta basada en la información proporcionada",
])
def test_is_no_answer_recognises_the_escape_hatch(answer):
    assert LLMProcessorOllama.is_no_answer(answer) is True


@pytest.mark.parametrize("answer", [
    REAL_ANSWER,
    "",
    None,
    f"El FET tiene alta impedancia. {ESCAPE_HATCH}",
])
def test_is_no_answer_rejects_real_answers(answer):
    assert LLMProcessorOllama.is_no_answer(answer) is False


# --- route_after_generation: the #29 change --------------------------------------

@pytest.mark.parametrize("iteration_count", range(MAX_CRAG_ITERATIONS))
def test_escape_hatch_opens_the_correction_loop_while_budget_remains(iteration_count):
    """The generator read the chunks in full and reported retrieval failed.

    That is a stronger signal than the grader's verdict, which only ever saw
    GRADER_CHUNK_PREVIEW_CHARS of each chunk. It must open the loop, not end the run.
    """
    processor = build_processor()
    state = {"answer": ESCAPE_HATCH, "iteration_count": iteration_count}
    assert processor.route_after_generation(state) == "reformulate_query"


def test_escape_hatch_refuses_once_the_reformulation_budget_is_spent():
    processor = build_processor()
    state = {"answer": ESCAPE_HATCH, "iteration_count": MAX_CRAG_ITERATIONS}
    assert processor.route_after_generation(state) == "out_of_scope"


def test_escape_hatch_with_no_iteration_count_still_reformulates():
    """A state that never went through reformulation defaults to budget available."""
    processor = build_processor()
    assert processor.route_after_generation({"answer": ESCAPE_HATCH}) == "reformulate_query"


def test_real_answer_goes_to_grounding_verification():
    processor = build_processor()
    state = {"answer": REAL_ANSWER, "iteration_count": 0}
    assert processor.route_after_generation(state) == "verify_grounding"


# --- reformulate_query_node: the option-B counter reset ---------------------------

def test_reformulation_resets_the_generation_budget():
    """A rewritten query retrieves different chunks, so the Self-RAG retry budget —
    which bounds regeneration *over the same context* — starts fresh.

    Without this, one escape-hatch reformulation spends the grounding retry and a
    genuinely ungrounded answer on the next round is withheld with no second chance.
    """
    processor = build_processor()
    update = processor.reformulate_query_node({
        "question": "¿Por qué los FET presentan una elevada impedancia de entrada?",
        "search_query": "impedancia FET",
        "iteration_count": 0,
        "generation_attempts": 1,
    })
    assert update["generation_attempts"] == 0


def test_reformulation_resets_the_generation_budget_even_when_the_rewrite_fails():
    """Same structural guarantee as iteration_count: unconditional, so the reset
    cannot be skipped by a grader that returns nothing usable."""
    processor = build_processor(grader=StubGraderLLM(response="   "))
    update = processor.reformulate_query_node({
        "question": "¿Por qué los FET presentan una elevada impedancia de entrada?",
        "search_query": "impedancia FET",
        "iteration_count": 0,
        "generation_attempts": 2,
    })
    assert update["generation_attempts"] == 0
    assert update["iteration_count"] == 1


def test_reformulation_still_increments_iteration_count_on_grader_failure():
    """The loop stays bounded by construction even when the control model is down."""
    processor = build_processor(grader=StubGraderLLM(error=RuntimeError("phi is down")))
    update = processor.reformulate_query_node({
        "question": "¿Por qué los FET presentan una elevada impedancia de entrada?",
        "search_query": "impedancia FET",
        "iteration_count": 1,
        "generation_attempts": 1,
    })
    assert update["iteration_count"] == MAX_CRAG_ITERATIONS
    assert update["generation_attempts"] == 0


# --- the refusal contract run_baseline.py depends on ------------------------------

def test_out_of_scope_answer_is_byte_identical_and_pins_the_verdict():
    processor = build_processor()
    update = processor.out_of_scope_node({"question": "¿Cuál es la capital de Francia?"})
    assert update["answer"] == OUT_OF_SCOPE_ANSWER
    assert update["relevance"] == RELEVANCE_IRRELEVANT
    assert not OUT_OF_SCOPE_ANSWER.startswith("Error:")


# --- derive_confidence: the label the reset keeps honest --------------------------

def test_reformulated_then_answered_is_medium_not_low():
    """`baja` means "the verifier rejected a draft over this context". A question that
    only needed its query rewritten never had a draft rejected, so labelling it `baja`
    would report a generator failure that did not happen."""
    state = {"relevance": RELEVANCE_RELEVANT,
             "grounding": GROUNDING_GROUNDED,
             "iteration_count": 1,
             "generation_attempts": 1}
    assert LLMProcessorOllama.derive_confidence(state) == CONFIDENCE_MEDIUM


def test_regenerated_after_a_rejected_draft_is_low():
    state = {"relevance": RELEVANCE_RELEVANT,
             "grounding": GROUNDING_GROUNDED,
             "iteration_count": 0,
             "generation_attempts": 2}
    assert LLMProcessorOllama.derive_confidence(state) == CONFIDENCE_LOW


def test_clean_run_is_high():
    state = {"relevance": RELEVANCE_RELEVANT,
             "grounding": GROUNDING_GROUNDED,
             "iteration_count": 0,
             "generation_attempts": 1}
    assert LLMProcessorOllama.derive_confidence(state) == CONFIDENCE_HIGH


@pytest.mark.parametrize("state", [
    {"relevance": RELEVANCE_IRRELEVANT, "grounding": GROUNDING_GROUNDED},
    {"relevance": RELEVANCE_RELEVANT, "grounding": GROUNDING_UNGROUNDED},
])
def test_both_fallbacks_carry_no_confidence_note(state):
    assert LLMProcessorOllama.derive_confidence(state) == CONFIDENCE_NOT_APPLICABLE


# --- the graph actually wires the new edge ---------------------------------------

def test_generate_node_can_reach_reformulate_query_in_the_compiled_graph():
    processor = build_processor()
    graph = processor.build_graph()
    successors = {edge.target for edge in graph.get_graph().edges if edge.source == "generate"}
    assert "reformulate_query" in successors
    assert {"verify_grounding", "out_of_scope"} <= successors


def test_classic_graph_has_no_correction_nodes():
    """The baseline arm must stay the linear pipeline run_baseline.py measures."""
    processor = LLMProcessorOllama(llm=None, embedding=None, context_length=1200, grader_llm=None)
    nodes = set(processor.build_graph().get_graph().nodes)
    assert "reformulate_query" not in nodes
    assert "verify_grounding" not in nodes


# --- route_after_grading is unchanged --------------------------------------------

@pytest.mark.parametrize("relevance", [RELEVANCE_RELEVANT, RELEVANCE_PARTIAL])
def test_partial_relevance_still_goes_straight_to_generation(relevance):
    processor = build_processor()
    assert processor.route_after_grading({"relevance": relevance, "iteration_count": 0}) == "generate"


def test_irrelevant_verdict_refuses_once_the_budget_is_spent():
    processor = build_processor()
    state = {"relevance": RELEVANCE_IRRELEVANT, "iteration_count": MAX_CRAG_ITERATIONS}
    assert processor.route_after_grading(state) == "out_of_scope"


# --- route_after_grounding is unchanged ------------------------------------------

def test_ungrounded_retries_while_the_generation_budget_remains():
    processor = build_processor()
    state = {"grounding": GROUNDING_UNGROUNDED, "generation_attempts": MAX_GENERATION_ATTEMPTS - 1}
    assert processor.route_after_grounding(state) == "generate"


def test_ungrounded_falls_back_once_the_generation_budget_is_spent():
    processor = build_processor()
    state = {"grounding": GROUNDING_UNGROUNDED, "generation_attempts": MAX_GENERATION_ATTEMPTS}
    assert processor.route_after_grounding(state) == "ungrounded"


# --- the bound, verified by execution rather than by arithmetic -------------------

class StubDoc:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {}


class StubRetriever:
    def invoke(self, query):
        return [StubDoc(f"contenido recuperado para {query}") for _ in range(4)]


class StubVectorstore:
    def as_retriever(self, **kwargs):
        return StubRetriever()


class ScriptedLLM:
    """Generator stub that always uses the escape hatch — the worst case for #29."""

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return self.response


def build_running_processor(generator_response, grader_response):
    processor = LLMProcessorOllama(llm=ScriptedLLM(generator_response),
                                   embedding=None,
                                   context_length=1200,
                                   grader_llm=ScriptedLLM(grader_response))
    processor.vectorstore = StubVectorstore()
    return processor


def test_a_permanently_unanswerable_question_terminates_out_of_scope():
    """Every generation returns the escape hatch and every grade says relevante: the
    pathological input the new edge creates. It must terminate, and it must terminate
    on the refusal rather than on LangGraph's recursion limit."""
    processor = build_running_processor(ESCAPE_HATCH, RELEVANCE_RELEVANT)
    final = processor.build_graph().invoke({"question": "¿Cuál es la capital de Francia?"})
    assert final["answer"] == OUT_OF_SCOPE_ANSWER
    assert final["iteration_count"] == MAX_CRAG_ITERATIONS
    assert LLMProcessorOllama.derive_confidence(final) == CONFIDENCE_NOT_APPLICABLE


def test_a_permanently_ungrounded_answer_terminates_on_the_fallback():
    """Generation always produces a real answer and the verifier always rejects it.
    Reformulation never fires, so the Self-RAG retry budget is the only bound."""
    processor = build_running_processor(REAL_ANSWER, GROUNDING_UNGROUNDED)
    final = processor.build_graph().invoke({"question": "¿Qué es el CMRR?"})
    assert final["answer"] == UNGROUNDED_FALLBACK_ANSWER
    assert final["generation_attempts"] == MAX_GENERATION_ATTEMPTS
    assert LLMProcessorOllama.derive_confidence(final) == CONFIDENCE_NOT_APPLICABLE


class AlternatingLLM:
    """Real answer on odd calls, escape hatch on even ones.

    Drives both loops on the same question: the real answer reaches the verifier and
    burns a grounding retry, the escape hatch then opens the correction loop. A single
    fixed response can only ever exercise one loop, which is why this stub alternates.
    """

    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return REAL_ANSWER if self.calls % 2 == 1 else ESCAPE_HATCH


class AlwaysRejectingControl:
    """Grades every retrieval relevante and rejects every answer as ungrounded."""

    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return GROUNDING_UNGROUNDED if "fundament" in prompt.lower() else RELEVANCE_RELEVANT


def test_the_worst_case_walk_stays_inside_the_default_recursion_limit():
    """Both loops driven to exhaustion on one question.

    LangGraph raises GraphRecursionError past 25 super-steps, so reaching a terminal
    answer at all is the real assertion; the counts are pinned so that raising either
    cap fails here instead of in production, where one question costs hours.
    """
    processor = LLMProcessorOllama(llm=AlternatingLLM(),
                                   embedding=None,
                                   context_length=1200,
                                   grader_llm=AlwaysRejectingControl())
    processor.vectorstore = StubVectorstore()

    super_steps = 0
    for _ in processor.build_graph().stream({"question": "pregunta patológica"},
                                            stream_mode="updates"):
        super_steps += 1

    assert super_steps == 18, "worst-case walk changed; re-check recursion_limit"
    assert super_steps < 25
    assert processor.llm.calls == (MAX_CRAG_ITERATIONS + 1) * MAX_GENERATION_ATTEMPTS == 6
    assert processor.grader_llm.calls == 8
