"""Latency harness for the RAG pipeline, in either of its two shapes.

Runs every question of the teacher-validated evaluation dataset through the
Ollama + FAISS pipeline, records the generated answer and the wall-clock
latency of each call, and leaves the four quality criteria (pertinencia,
claridad, precision, lenguaje) empty for an instructor to score by hand.

Which pipeline gets measured is chosen with ``--no-crag`` (the default) or
``--crag``:

* ``--no-crag`` builds ``build_ollama_processor(enable_crag=False)``: a single
  FAISS retrieval followed by a single generation call. This is the classic,
  pre-agentic reference point, and it is the default on purpose. Production has
  had the CRAG correction loop on by default since issue #15, so without an
  explicit opt-out this harness would silently measure the agentic pipeline
  while still labelling its output ``classic-rag`` — and since no baseline has
  been recorded yet, the reference point CRAG is meant to be compared against
  would be lost before it was ever taken.
* ``--crag`` builds ``build_ollama_processor(enable_crag=True)`` and
  additionally records, per question, how many reformulations the correction
  loop performed and whether the out-of-scope short-circuit fired.

Both arms are measured by the same code on the same code path, which is the
whole point of running them from one script: a comparison between two harnesses
would be a comparison between two measurement methodologies. They write to
different files (``results/baseline_*`` vs ``results/crag_*``), and the harness
refuses to append a run of one pipeline onto a results file recorded with the
other. ``compare_runs.py`` diffs the two.

The script deliberately never scores answers automatically. Quality grading is
a human task here: a heuristic or an LLM-as-judge shortcut would invalidate the
very baseline this measurement exists to produce.

Run it from the repository root, on the target hardware, with the Ollama server
already up (see resources/eval/BASELINE.md and resources/eval/AB_TESTING.md).
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import requests

# The pipeline resolves resources/faiss_index and resources/files relative to the
# process working directory, so the repository root has to be both importable and
# the current directory before anything from rag/ is loaded.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# resources/ is not a package, so the sibling results-format module is imported by
# directory rather than by dotted path.
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from eval_io import (PIPELINE_ERROR_PREFIX, QUALITY_CRITERIA, SCHEMA_VERSION,  # noqa: E402
                     append_record, build_output_path, build_provenance, is_successful, load_records,
                     load_recorded_ids, pipeline_label, record_pipeline, summarize_latencies,
                     write_summary)
from rag.document_loader import LocalDocumentLoader  # noqa: E402
from rag.llm_processor import MAX_CRAG_ITERATIONS, OUT_OF_SCOPE_ANSWER, RETRIEVAL_K  # noqa: E402
from rag.main import (CHUNK_SIZE_TOKENS, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_NUM_CTX,  # noqa: E402
                      OLLAMA_TEMPERATURE, OLLAMA_TOP_P, PHI_MODEL, build_ollama_processor)

DEFAULT_DATASET_PATH = os.path.join("resources", "eval", "dataset.jsonl")
DEFAULT_DOCUMENT_LOCATION = os.path.join("resources", "files")
DEFAULT_OLLAMA_SERVER_URL = OLLAMA_HOST

REQUIRED_DATASET_FIELDS = ("id", "question")

WARMUP_QUESTION = "Pregunta de calentamiento del modelo, la respuesta no se registra."


class CragInstrumentation:
    """Counts CRAG node executions, one question at a time.

    ``ask_question`` returns a plain ``str``. The final graph state — where
    ``iteration_count`` and the relevance verdicts live — never leaves the processor,
    and widening that return type would change the contract of the method the API and
    email paths call in production, for the sake of a measurement-only concern. So
    instead of reaching into the pipeline, this class wraps the bound CRAG node methods
    on the processor *instance* and counts the calls from the outside.

    The wrappers only observe: each delegates to the original node and returns its state
    update untouched, so the pipeline being measured stays byte-for-byte the pipeline
    that runs in production. Without this, the A/B report could not answer either of the
    two questions issue #16 asks — "how slow is the worst case with 2 iterations" and
    "how often does the out-of-scope short-circuit fire" — because both are properties of
    the walk through the graph, not of the answer that comes out of it.

    ``install`` must run before the first question: ``build_graph`` captures the bound
    methods when it compiles the graph, and ``ask_question`` compiles it lazily on the
    first call.

    Args:
        processor: Processor returned by ``build_ollama_processor(enable_crag=True)``.
    """

    def __init__(self, processor: Any):
        self.processor = processor
        self.installed = False
        self.retrievals = 0
        self.relevance_grades = 0
        self.reformulations = 0
        self.relevance_verdicts: List[Optional[str]] = []

    def install(self) -> None:
        """Wrap the CRAG nodes of the processor with counting delegates.

        Raises:
            ValueError: If the processor has no grader model, i.e. CRAG is off and there
                is nothing to instrument.
        """
        if self.installed:
            return
        if self.processor.grader_llm is None:
            raise ValueError("Cannot instrument CRAG on a processor built with enable_crag=False")

        def counting(original: Callable[..., Dict[str, Any]],
                     observe: Callable[[Dict[str, Any]], None]) -> Callable[..., Dict[str, Any]]:
            def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
                update = original(state)
                observe(update)
                return update
            return wrapper

        def count_retrieval(_update: Dict[str, Any]) -> None:
            self.retrievals += 1

        def count_grade(update: Dict[str, Any]) -> None:
            self.relevance_grades += 1
            self.relevance_verdicts.append(update.get("relevance"))

        def count_reformulation(update: Dict[str, Any]) -> None:
            # reformulate_query_node owns iteration_count and increments it even when the
            # rewrite fails, so its own number is the authoritative one to record.
            self.reformulations = update.get("iteration_count", self.reformulations + 1)

        self.processor.retrieve_node = counting(self.processor.retrieve_node, count_retrieval)
        self.processor.grade_relevance_node = counting(self.processor.grade_relevance_node, count_grade)
        self.processor.reformulate_query_node = counting(self.processor.reformulate_query_node,
                                                         count_reformulation)
        # The graph binds these methods when it compiles. Drop any graph compiled before
        # the wrappers went in, otherwise the instrumented nodes would never be called.
        self.processor.graph = None
        self.installed = True
        logging.debug("CRAG instrumentation installed on the processor's graph nodes")

    def reset(self) -> None:
        """Zero the counters before measuring the next question."""
        self.retrievals = 0
        self.relevance_grades = 0
        self.reformulations = 0
        self.relevance_verdicts = []

    def snapshot(self, answer: str) -> Dict[str, Any]:
        """Freeze the counters of the question that just ran.

        Args:
            answer: Answer the pipeline returned, used to detect the short-circuit.

        Returns:
            Dict[str, Any]: The ``crag`` block of the result record.
        """
        return {
            "reformulations": self.reformulations,
            "retrievals": self.retrievals,
            "relevance_grades": self.relevance_grades,
            "relevance_verdicts": list(self.relevance_verdicts),
            "hit_iteration_cap": self.reformulations >= MAX_CRAG_ITERATIONS,
            "out_of_scope": is_out_of_scope(answer)
        }


def is_out_of_scope(answer: str) -> bool:
    """Report whether an answer is the CRAG out-of-scope short-circuit.

    Compared against the constant rather than pattern-matched, so a rewording of
    OUT_OF_SCOPE_ANSWER cannot quietly turn into a wrong short-circuit rate. Checked for
    both arms even though the classic pipeline has no node that can produce it: a
    non-zero count on the classic side would mean the two runs were not what they claim
    to be, which is worth catching.

    Args:
        answer: Answer returned by the pipeline.

    Returns:
        bool: True when the answer is the out-of-scope refusal, verbatim.
    """
    return answer.strip() == OUT_OF_SCOPE_ANSWER.strip()


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load and validate the evaluation dataset.

    Args:
        path: Path to the JSON Lines dataset described in resources/eval/README.md.

    Returns:
        List[Dict[str, Any]]: One entry per non-empty line, in file order.

    Raises:
        FileNotFoundError: If the dataset does not exist yet.
        ValueError: If a line is not valid JSON, misses a required field, or reuses an id.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Evaluation dataset not found at '{path}'. It is written and validated by the course "
            f"instructors (see resources/eval/README.md); do not generate synthetic questions to fill it."
        )

    entries: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    with open(path, 'r', encoding='utf-8') as dataset_file:
        for line_number, raw_line in enumerate(dataset_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {err}") from err
            if not isinstance(entry, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object, got {type(entry).__name__}")
            for field in REQUIRED_DATASET_FIELDS:
                if not str(entry.get(field, "")).strip():
                    raise ValueError(f"{path}:{line_number} is missing the required field '{field}'")
            entry_id = str(entry['id'])
            if entry_id in seen_ids:
                raise ValueError(f"{path}:{line_number} reuses id '{entry_id}'; ids must be unique and never reused")
            seen_ids.add(entry_id)
            entries.append(entry)

    if not entries:
        raise ValueError(f"Evaluation dataset '{path}' has no entries; nothing to measure.")
    return entries


def build_record(entry: Dict[str, Any], answer: str, latency_sec: float,
                 crag_stats: Optional[Dict[str, Any]],
                 provenance: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the result record for a single measured question.

    Args:
        entry: Dataset entry the question came from.
        answer: Answer returned by the pipeline, verbatim.
        latency_sec: Wall-clock seconds spent inside the pipeline call.
        crag_stats: Snapshot of the correction loop for this question, or None on a
            classic run. None rather than a block of zeroes on purpose, and for the same
            reason ``scores`` starts as null: "the loop was not there" and "the loop ran
            and did nothing" are different facts, and a comparison that confused the two
            would report a reformulation rate of zero for a pipeline that has no
            reformulation node at all.
        provenance: Commit, dirty flag and corpus hash of the system being measured.
            Stored per record, not only in the summary, because the harness resumes: a
            run interrupted and continued after a code change writes records from two
            systems into one file, and only a per-record block can expose that.

    Returns:
        Dict[str, Any]: Record ready to be appended to the results file, with the
        four quality criteria left as null for a human reviewer.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance,
        "id": str(entry['id']),
        "pipeline": pipeline_label(crag_stats is not None),
        "question": entry['question'],
        "expected_answer": entry.get('expected_answer'),
        "source_doc": entry.get('source_doc'),
        "topic": entry.get('topic'),
        "difficulty": entry.get('difficulty'),
        "generated_answer": answer,
        "latency_sec": round(latency_sec, 3),
        "pipeline_error": answer.startswith(PIPELINE_ERROR_PREFIX),
        "out_of_scope": is_out_of_scope(answer),
        "crag": crag_stats,
        "measured_at": datetime.now().isoformat(timespec='seconds'),
        "scores": {criterion: None for criterion in QUALITY_CRITERIA},
        "scored_by": None,
        "scored_at": None,
        "reviewer_notes": ""
    }


def summarize_crag(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the per-question CRAG counters of a run.

    Splitting latency by reformulation count is what turns the raw numbers into the
    typical-case and worst-case figures issue #16 asks for: bucket 0 is a question the
    correction loop did not touch, bucket ``MAX_CRAG_ITERATIONS`` is the worst case.

    Args:
        records: Every result record of a CRAG run.

    Returns:
        Dict[str, Any]: Reformulation histogram, verdict counts, out-of-scope tally and
        per-bucket latency statistics.
    """
    histogram: Dict[str, int] = {}
    verdict_counts: Dict[str, int] = {}
    latencies_by_bucket: Dict[str, List[float]] = {}
    out_of_scope_ids: List[str] = []

    for record in records:
        crag = record.get('crag') or {}
        bucket = str(crag.get('reformulations', 0))
        histogram[bucket] = histogram.get(bucket, 0) + 1
        for verdict in crag.get('relevance_verdicts') or []:
            key = str(verdict)
            verdict_counts[key] = verdict_counts.get(key, 0) + 1
        if record.get('out_of_scope'):
            out_of_scope_ids.append(str(record.get('id')))
        if is_successful(record):
            latencies_by_bucket.setdefault(bucket, []).append(record['latency_sec'])

    answered = [record for record in records if is_successful(record)]
    return {
        "max_iterations": MAX_CRAG_ITERATIONS,
        "reformulation_histogram": dict(sorted(histogram.items())),
        "total_reformulations": sum(int(bucket) * count for bucket, count in histogram.items()),
        "questions_reformulated": sum(count for bucket, count in histogram.items() if int(bucket) > 0),
        "questions_at_iteration_cap": sum(count for bucket, count in histogram.items()
                                          if int(bucket) >= MAX_CRAG_ITERATIONS),
        "relevance_verdict_counts": dict(sorted(verdict_counts.items())),
        "out_of_scope_answers": len(out_of_scope_ids),
        "out_of_scope_rate": round(len(out_of_scope_ids) / len(answered), 4) if answered else None,
        "out_of_scope_ids": out_of_scope_ids,
        "latency_by_reformulations": {bucket: summarize_latencies(latencies)
                                      for bucket, latencies in sorted(latencies_by_bucket.items())}
    }


def build_summary(records: List[Dict[str, Any]], args: argparse.Namespace,
                  warmup_latency_sec: Optional[float], index_build_sec: Optional[float],
                  chunk_count: Optional[int],
                  corpus_hash: Optional[str] = None) -> Dict[str, Any]:
    """Assemble the run summary written next to the results file.

    Args:
        records: Every result record of the run, including resumed ones.
        args: Parsed command line arguments.
        warmup_latency_sec: Latency of the discarded warm-up call, if any.
        index_build_sec: Seconds spent loading or rebuilding the FAISS index.
        chunk_count: Number of context chunks the index was built from.
        corpus_hash: Hash of the chunks this run retrieved from, or None when the
            summary is being recomputed from an existing results file and no index was
            built. Null means "not recorded here", which is not the same fact as a hash
            that differs -- the per-record blocks still carry the real one.

    Returns:
        Dict[str, Any]: Self-describing summary of configuration and latency.
    """
    successful = [record['latency_sec'] for record in records if is_successful(record)]
    failed = [record['id'] for record in records if not is_successful(record)]
    scored = [record for record in records
              if all(record.get('scores', {}).get(criterion) is not None for criterion in QUALITY_CRITERIA)]
    if args.crag:
        pipeline_note = ("CRAG pipeline: FAISS retrieval, Phi-3.5-mini relevance grading, up to "
                         f"{MAX_CRAG_ITERATIONS} query reformulations and the out-of-scope short-circuit "
                         "(enable_crag=True). Compare against the classic-rag results file of the same "
                         "model and k with resources/eval/compare_runs.py.")
    else:
        pipeline_note = ("Classic pipeline: FAISS retrieval + single Ollama generation call. The CRAG "
                         "correction loop is explicitly disabled (enable_crag=False), so these numbers "
                         "remain the pre-agentic reference point even though production runs CRAG.")
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": build_provenance(corpus_hash),
        "generated_at": datetime.now().isoformat(timespec='seconds'),
        "pipeline": pipeline_label(args.crag),
        "pipeline_note": pipeline_note,
        "configuration": {
            "model": OLLAMA_MODEL,
            "crag_enabled": args.crag,
            "grader_model": PHI_MODEL if args.crag else None,
            "max_crag_iterations": MAX_CRAG_ITERATIONS if args.crag else None,
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "num_ctx": OLLAMA_NUM_CTX,
            "chunk_size_tokens": CHUNK_SIZE_TOKENS,
            "retrieval_k": RETRIEVAL_K,
            "ollama_server_url": args.ollama_url,
            "document_location": args.document_location,
            "dataset": args.dataset,
            "context_chunks": chunk_count
        },
        "run": {
            "questions_recorded": len(records),
            "questions_failed": len(failed),
            "failed_ids": failed,
            "out_of_scope_answers": sum(1 for record in records if record.get('out_of_scope')),
            "warmup_latency_sec": round(warmup_latency_sec, 3) if warmup_latency_sec is not None else None,
            "index_build_or_load_sec": round(index_build_sec, 3) if index_build_sec is not None else None
        },
        "latency": summarize_latencies(successful),
        # Null rather than an empty block on a classic run: the classic graph has no
        # grading or reformulation node, so there is nothing that could have been counted.
        "crag": summarize_crag(records) if args.crag else None,
        "quality": {
            "criteria": list(QUALITY_CRITERIA),
            "scored_questions": len(scored),
            "pending_questions": len(records) - len(scored),
            "note": "Quality scores are filled in by hand by course instructors. This harness never auto-scores."
        }
    }


def check_ollama(ollama_url: str) -> None:
    """Fail fast if the Ollama server is not reachable.

    Args:
        ollama_url: Base URL of the Ollama server.

    Raises:
        RuntimeError: If the server does not answer or answers with an error.
    """
    try:
        response = requests.get(url=ollama_url, timeout=10)
    except requests.RequestException as err:
        raise RuntimeError(f"Ollama is not reachable at '{ollama_url}': {err}") from err
    if not response.ok:
        raise RuntimeError(f"Ollama answered {response.status_code} at '{ollama_url}'")
    logging.info(f"Ollama is reachable at '{ollama_url}'")


def prepare_context(document_location: str,
                    enable_crag: bool) -> Tuple[List[str], Any, Optional[CragInstrumentation], float, str]:
    """Load the course documents and build or load the FAISS index.

    Args:
        document_location: Directory holding the course PDFs/DOCXs.
        enable_crag: Build the processor with the CRAG correction loop wired in.
            False keeps this a classic-RAG measurement: linear retrieve -> generate, no
            relevance grading, no reformulation, control model never loaded.

    Returns:
        Tuple[List[str], Any, Optional[CragInstrumentation], float, str]: Context chunks,
        the ready processor, the node counters (None on a classic run), the seconds spent
        building or loading the vector store, and the corpus hash this run retrieved from.

        The corpus hash is ``get_index_hash`` over the same chunks the retriever sees, so
        it covers the chunk text *and* the embedding model. It is recorded because the
        course documents are untracked (#35): the corpus can change with no commit to
        show for it, which is exactly what ``dbed292`` did between two measured runs.
    """
    logging.info(f"Loading course documents from '{document_location}'")
    document = LocalDocumentLoader(document_location).load_document()
    llm = build_ollama_processor(enable_crag=enable_crag)
    context_chunked = document.get_chunked_text(llm.context_length)
    logging.info(f"Document split into {len(context_chunked)} chunks")

    instrumentation: Optional[CragInstrumentation] = None
    if enable_crag:
        # Installed here, before any question runs, because the graph binds its node
        # callables when ask_question lazily compiles it on the first call.
        instrumentation = CragInstrumentation(llm)
        instrumentation.install()

    logging.info("Building or loading the FAISS index")
    index_start = time.perf_counter()
    llm.build_or_load_vectorstore(context_chunked)
    index_build_sec = time.perf_counter() - index_start
    logging.info(f"FAISS index ready in {index_build_sec:.1f} s")
    corpus_hash = llm.get_index_hash(context_chunked)
    logging.info(f"Corpus hash: {corpus_hash[:12]}")
    return context_chunked, llm, instrumentation, index_build_sec, corpus_hash


def measure(llm: Any, question: str, context_chunked: List[str],
            instrumentation: Optional[CragInstrumentation]) -> Tuple[str, float, Optional[Dict[str, Any]]]:
    """Run one question through the pipeline and time it.

    Both arms go through ``process_questionnaire``, the same entry point production
    uses. A harness that called the graph directly to read its final state would be
    measuring a code path no student ever hits, and the A/B comparison would then be
    partly a comparison of measurement methods.

    Args:
        llm: Processor returned by ``build_ollama_processor``.
        question: Question text to send.
        context_chunked: Context chunks backing the vector store.
        instrumentation: CRAG node counters, or None on a classic run.

    Returns:
        Tuple[str, float, Optional[Dict[str, Any]]]: The answer, the wall-clock seconds
        it took, and the CRAG snapshot of this question (None on a classic run).
    """
    if instrumentation is not None:
        instrumentation.reset()
    start = time.perf_counter()
    answer = llm.process_questionnaire(question, context_chunked)
    latency_sec = time.perf_counter() - start
    answer = answer if isinstance(answer, str) else str(answer)
    crag_stats = instrumentation.snapshot(answer) if instrumentation is not None else None
    return answer, latency_sec, crag_stats


def print_summary(summary: Dict[str, Any]) -> None:
    """Print the run summary to stdout.

    Args:
        summary: Summary produced by ``build_summary``.
    """
    latency = summary['latency']
    run = summary['run']
    print("\n" + "=" * 72)
    print(f"{summary['pipeline']} run - {summary['configuration']['model']} - "
          f"k={summary['configuration']['retrieval_k']}")
    print("=" * 72)
    print(f"Questions recorded : {run['questions_recorded']} ({run['questions_failed']} failed)")
    if run['index_build_or_load_sec'] is not None:
        print(f"FAISS index ready  : {run['index_build_or_load_sec']:.1f} s")
    if run['warmup_latency_sec'] is not None:
        print(f"Warm-up call       : {run['warmup_latency_sec']:.1f} s (discarded, not in the statistics)")
    if latency['count']:
        print(f"Latency mean       : {latency['mean_sec']:.1f} s")
        print(f"Latency median     : {latency['median_sec']:.1f} s")
        print(f"Latency min / max  : {latency['min_sec']:.1f} s / {latency['max_sec']:.1f} s")
        print(f"Latency p95        : {latency['p95_sec']:.1f} s")
        if latency['stdev_sec'] is not None:
            print(f"Latency stdev      : {latency['stdev_sec']:.1f} s")
        print(f"Total generation   : {latency['total_sec'] / 60:.1f} min")
    else:
        print("Latency            : no successful question, nothing to summarize")
    if summary['crag'] is not None:
        crag = summary['crag']
        histogram = ", ".join(f"{bucket}x: {count}" for bucket, count in crag['reformulation_histogram'].items())
        print(f"Reformulations     : {crag['total_reformulations']} over "
              f"{crag['questions_reformulated']} question(s) [{histogram or 'none'}]")
        print(f"At iteration cap   : {crag['questions_at_iteration_cap']} question(s)")
        print(f"Out of scope       : {crag['out_of_scope_answers']} question(s) short-circuited")
    print(f"Quality scores     : {summary['quality']['pending_questions']} question(s) still awaiting "
          f"instructor scoring")
    print("=" * 72 + "\n")


def assert_pipeline_matches(path: str, requested_pipeline: str) -> None:
    """Refuse to mix two pipelines inside one results file.

    Appending a CRAG run onto the classic baseline (or the reverse) would produce a file
    whose latency statistics average two different pipelines, and nothing downstream
    could tell afterwards which record came from which. Easy to do by accident with
    ``--output`` or a stale ``--resume``, and unrecoverable once the run has cost two
    hours of generation, so it is checked before anything is measured.

    Args:
        path: Results file that already exists.
        requested_pipeline: Pipeline label this run would append.

    Raises:
        ValueError: If the file was recorded with a different pipeline.
    """
    existing = {record_pipeline(record) for record in load_records(path)}
    conflicting = existing - {requested_pipeline}
    if conflicting:
        raise ValueError(
            f"Results file '{path}' was recorded with pipeline {sorted(conflicting)}, but this run is "
            f"'{requested_pipeline}'. Mixing both in one file makes its statistics meaningless. Use the "
            f"default --output for this pipeline, or point --output at a fresh file."
        )


def run_evaluation(args: argparse.Namespace) -> int:
    """Execute the measurement run for the selected pipeline.

    Args:
        args: Parsed command line arguments.

    Returns:
        int: Process exit code.
    """
    pipeline = pipeline_label(args.crag)
    entries = load_dataset(args.dataset)
    logging.info(f"Loaded {len(entries)} question(s) from '{args.dataset}'")

    if args.limit is not None:
        entries = entries[:args.limit]
        logging.warning(f"--limit is set: only the first {len(entries)} question(s) will run. "
                        f"A partial run is a smoke test, not a baseline.")

    if args.resume:
        assert_pipeline_matches(args.output, pipeline)
        recorded_ids = load_recorded_ids(args.output)
        if recorded_ids:
            entries = [entry for entry in entries if str(entry['id']) not in recorded_ids]
            logging.info(f"Resuming: {len(recorded_ids)} question(s) already recorded, {len(entries)} left")
    elif os.path.exists(args.output):
        raise FileExistsError(
            f"Results file '{args.output}' already exists. Pass --resume to continue that run, "
            f"or --output to write somewhere else. Existing results are never overwritten."
        )

    if args.dry_run:
        print(f"Dry run: dataset '{args.dataset}' is valid, {len(entries)} question(s) would be measured.")
        print(f"Pipeline: {pipeline}.")
        print(f"Results would be written to '{args.output}'.")
        print(f"Configuration: model={OLLAMA_MODEL}, k={RETRIEVAL_K}, num_ctx={OLLAMA_NUM_CTX}, "
              f"crag={'enabled' if args.crag else 'disabled'}"
              f"{f', grader={PHI_MODEL}, max_iterations={MAX_CRAG_ITERATIONS}' if args.crag else ''}.")
        return 0

    if not entries:
        logging.info("Nothing left to measure; recomputing the summary from the existing results file")
        summary = build_summary(load_records(args.output), args, None, None, None)
        write_summary(args.output, summary)
        print_summary(summary)
        return 0

    check_ollama(args.ollama_url)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    context_chunked, llm, instrumentation, index_build_sec, corpus_hash = prepare_context(
        args.document_location, args.crag)
    # Resolved once, before the first question, so every record of this run carries the
    # same block even if the working tree is edited while a two-hour run is in flight.
    provenance = build_provenance(corpus_hash)

    warmup_latency_sec: Optional[float] = None
    if args.warmup:
        logging.info("Warm-up call so the first measured question does not absorb the model load time")
        _, warmup_latency_sec, _ = measure(llm, WARMUP_QUESTION, context_chunked, instrumentation)
        logging.info(f"Warm-up finished in {warmup_latency_sec:.1f} s (discarded)")

    for position, entry in enumerate(entries, start=1):
        entry_id = str(entry['id'])
        logging.info(f"[{position}/{len(entries)}] Measuring question '{entry_id}' ({pipeline})")
        answer, latency_sec, crag_stats = measure(llm, entry['question'], context_chunked, instrumentation)
        if answer.startswith(PIPELINE_ERROR_PREFIX):
            logging.error(f"Question '{entry_id}' failed inside the pipeline: {answer}")
        if crag_stats is not None:
            logging.info(f"Question '{entry_id}': {crag_stats['reformulations']} reformulation(s), "
                         f"{crag_stats['retrievals']} retrieval(s), "
                         f"out_of_scope={crag_stats['out_of_scope']}")
        logging.info(f"Question '{entry_id}' answered in {latency_sec:.1f} s")
        append_record(args.output, build_record(entry, answer, latency_sec, crag_stats, provenance))

    summary = build_summary(load_records(args.output), args, warmup_latency_sec, index_build_sec,
                            len(context_chunked), corpus_hash)
    write_summary(args.output, summary)
    print_summary(summary)
    return 0


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: Parsed arguments with defaults resolved.
    """
    parser = argparse.ArgumentParser(
        description="Measure per-question latency of the RAG pipeline against the evaluation dataset. "
                    "Measures the classic pipeline by default; pass --crag for the agentic one.",
        epilog="Quality criteria are never scored automatically; the results file leaves them null for instructors."
    )
    pipeline_group = parser.add_mutually_exclusive_group()
    pipeline_group.add_argument('--crag', dest='crag', action='store_true',
                                help="Measure the CRAG pipeline (relevance grading, bounded reformulation, "
                                     "out-of-scope short-circuit) and record per-question iteration counts.")
    pipeline_group.add_argument('--no-crag', dest='crag', action='store_false',
                                help="Measure the classic retrieve -> generate pipeline. This is the default, "
                                     "so an invocation with no flags still produces the pre-agentic baseline.")
    parser.set_defaults(crag=False)
    parser.add_argument('--dataset', default=DEFAULT_DATASET_PATH,
                        help=f"Evaluation dataset in JSON Lines format (default: {DEFAULT_DATASET_PATH})")
    parser.add_argument('--output', default=None,
                        help=f"Results file (default: {build_output_path(OLLAMA_MODEL, RETRIEVAL_K)}, or "
                             f"{build_output_path(OLLAMA_MODEL, RETRIEVAL_K, True)} with --crag)")
    parser.add_argument('--document-location', default=os.getenv('DOCUMENT_LOCATION', DEFAULT_DOCUMENT_LOCATION),
                        help="Directory with the course documents (default: $DOCUMENT_LOCATION or "
                             f"{DEFAULT_DOCUMENT_LOCATION})")
    parser.add_argument('--ollama-url', default=os.getenv('OLLAMA_SERVER_URL', DEFAULT_OLLAMA_SERVER_URL),
                        help="Ollama server URL, recorded in the summary and checked before the run "
                             "(default: $OLLAMA_SERVER_URL or the pipeline host)")
    parser.add_argument('--limit', type=int, default=None,
                        help="Only measure the first N questions. Smoke test only, not a valid baseline.")
    parser.add_argument('--resume', action='store_true',
                        help="Append to an existing results file, skipping question ids already recorded.")
    parser.add_argument('--no-warmup', dest='warmup', action='store_false',
                        help="Skip the discarded warm-up call. The first measured question then also pays "
                             "the model load time.")
    parser.add_argument('--dry-run', action='store_true',
                        help="Validate the dataset and print what would run, without loading the model.")
    parser.add_argument('--log-level', default='INFO',
                        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR'),
                        help="Logging verbosity (default: INFO)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.output is None:
        args.output = build_output_path(OLLAMA_MODEL, RETRIEVAL_K, args.crag)
    return args


def main() -> int:
    """Entry point.

    Returns:
        int: Process exit code.
    """
    args = parse_args()
    # rag/__init__.py configures the root logger at DEBUG on import, which drowns the
    # run in LangChain internals; the requested level wins here.
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    os.chdir(REPO_ROOT)
    logging.debug(f"Working directory set to '{REPO_ROOT}'")

    try:
        return run_evaluation(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as err:
        logging.error(str(err))
        return 1
    except KeyboardInterrupt:
        logging.warning(f"Interrupted. Results measured so far are in '{args.output}'; rerun with --resume.")
        return 130


if __name__ == '__main__':
    sys.exit(main())
