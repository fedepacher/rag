"""Baseline latency harness for the classic (pre-agentic) RAG pipeline.

Runs every question of the teacher-validated evaluation dataset through the
Ollama + FAISS pipeline, records the generated answer and the wall-clock
latency of each call, and leaves the four quality criteria (pertinencia,
claridad, precision, lenguaje) empty for an instructor to score by hand.

The pipeline measured here is deliberately the *classic* one: a single FAISS
retrieval followed by a single generation call. The processor is built with
``build_ollama_processor(enable_crag=False)`` so that the CRAG correction loop,
which is on by default in production since issue #15, is explicitly switched
off. Without that opt-out this harness would silently start measuring the
agentic pipeline while still labelling its output ``classic-rag`` — and since
no baseline has been recorded yet, the reference point CRAG is supposed to be
compared against would be lost before it was ever taken.

The script deliberately never scores answers automatically. Quality grading is
a human task here: a heuristic or an LLM-as-judge shortcut would invalidate the
very baseline this measurement exists to produce.

Run it from the repository root, on the target hardware, with the Ollama server
already up (see resources/eval/BASELINE.md).
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests

# The pipeline resolves resources/faiss_index and resources/files relative to the
# process working directory, so the repository root has to be both importable and
# the current directory before anything from rag/ is loaded.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rag.document_loader import LocalDocumentLoader  # noqa: E402
from rag.llm_processor import RETRIEVAL_K  # noqa: E402
from rag.main import (OLLAMA_CONTEXT_LENGTH, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_NUM_CTX,  # noqa: E402
                      OLLAMA_TEMPERATURE, OLLAMA_TOP_P, build_ollama_processor)

DEFAULT_DATASET_PATH = os.path.join("resources", "eval", "dataset.jsonl")
RESULTS_DIR = os.path.join("resources", "eval", "results")
DEFAULT_DOCUMENT_LOCATION = os.path.join("resources", "files")
DEFAULT_OLLAMA_SERVER_URL = OLLAMA_HOST

# Criteria scored by a human reviewer after the run. Written as null so an unscored
# record is impossible to mistake for a zero.
QUALITY_CRITERIA = ("pertinencia", "claridad", "precision", "lenguaje")
REQUIRED_DATASET_FIELDS = ("id", "question")

# LLMProcessorOllama swallows its own exceptions and returns a message with this prefix
# instead of raising, so a failed question has to be detected by inspecting the answer.
PIPELINE_ERROR_PREFIX = "Error:"
WARMUP_QUESTION = "Pregunta de calentamiento del modelo, la respuesta no se registra."


def build_output_path(model: str, retrieval_k: int) -> str:
    """Build the default results path for a model/retrieval-depth combination.

    Args:
        model: Ollama model tag being measured.
        retrieval_k: Number of chunks pulled from the vector store per question.

    Returns:
        str: Path such as ``resources/eval/results/baseline_<model>_k4.jsonl``.
    """
    model_slug = model.replace(":", "-").replace("/", "-")
    return os.path.join(RESULTS_DIR, f"baseline_{model_slug}_k{retrieval_k}.jsonl")


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


def load_recorded_ids(path: str) -> Set[str]:
    """Collect the question ids already present in a results file.

    Args:
        path: Path to a results JSON Lines file, which may not exist.

    Returns:
        Set[str]: Ids already measured. Empty if the file is absent.
    """
    if not os.path.exists(path):
        return set()
    recorded: Set[str] = set()
    with open(path, 'r', encoding='utf-8') as results_file:
        for raw_line in results_file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                recorded.add(str(json.loads(line)['id']))
            except (json.JSONDecodeError, KeyError, TypeError):
                logging.warning(f"Skipping unreadable record while scanning '{path}' for resumable ids")
    return recorded


def load_records(path: str) -> List[Dict[str, Any]]:
    """Read every result record from a results file.

    Args:
        path: Path to a results JSON Lines file, which may not exist.

    Returns:
        List[Dict[str, Any]]: Parsed records, skipping unreadable lines.
    """
    if not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as results_file:
        for raw_line in results_file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"Skipping unreadable record in '{path}'")
    return records


def build_record(entry: Dict[str, Any], answer: str, latency_sec: float) -> Dict[str, Any]:
    """Assemble the result record for a single measured question.

    Args:
        entry: Dataset entry the question came from.
        answer: Answer returned by the pipeline, verbatim.
        latency_sec: Wall-clock seconds spent inside the pipeline call.

    Returns:
        Dict[str, Any]: Record ready to be appended to the results file, with the
        four quality criteria left as null for a human reviewer.
    """
    return {
        "id": str(entry['id']),
        "question": entry['question'],
        "expected_answer": entry.get('expected_answer'),
        "source_doc": entry.get('source_doc'),
        "topic": entry.get('topic'),
        "difficulty": entry.get('difficulty'),
        "generated_answer": answer,
        "latency_sec": round(latency_sec, 3),
        "pipeline_error": answer.startswith(PIPELINE_ERROR_PREFIX),
        "measured_at": datetime.now().isoformat(timespec='seconds'),
        "scores": {criterion: None for criterion in QUALITY_CRITERIA},
        "scored_by": None,
        "scored_at": None,
        "reviewer_notes": ""
    }


def summarize_latencies(latencies: List[float]) -> Dict[str, Any]:
    """Compute descriptive latency statistics.

    Args:
        latencies: Per-question wall-clock seconds, successful questions only.

    Returns:
        Dict[str, Any]: Count plus min/max/mean/median/p95/stdev/total in seconds.
        Fields that need more samples than available are null.
    """
    if not latencies:
        return {"count": 0}
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "count": len(ordered),
        "min_sec": round(ordered[0], 3),
        "max_sec": round(ordered[-1], 3),
        "mean_sec": round(statistics.mean(ordered), 3),
        "median_sec": round(statistics.median(ordered), 3),
        "p95_sec": round(ordered[p95_index], 3),
        "stdev_sec": round(statistics.stdev(ordered), 3) if len(ordered) > 1 else None,
        "total_sec": round(sum(ordered), 3)
    }


def build_summary(records: List[Dict[str, Any]], args: argparse.Namespace,
                  warmup_latency_sec: Optional[float], index_build_sec: Optional[float],
                  chunk_count: Optional[int]) -> Dict[str, Any]:
    """Assemble the run summary written next to the results file.

    Args:
        records: Every result record of the run, including resumed ones.
        args: Parsed command line arguments.
        warmup_latency_sec: Latency of the discarded warm-up call, if any.
        index_build_sec: Seconds spent loading or rebuilding the FAISS index.
        chunk_count: Number of context chunks the index was built from.

    Returns:
        Dict[str, Any]: Self-describing summary of configuration and latency.
    """
    successful = [record['latency_sec'] for record in records if not record.get('pipeline_error')]
    failed = [record['id'] for record in records if record.get('pipeline_error')]
    scored = [record for record in records
              if all(record.get('scores', {}).get(criterion) is not None for criterion in QUALITY_CRITERIA)]
    return {
        "generated_at": datetime.now().isoformat(timespec='seconds'),
        "pipeline": "classic-rag",
        "pipeline_note": "Classic pipeline: FAISS retrieval + single Ollama generation call. The CRAG "
                         "correction loop is explicitly disabled (enable_crag=False), so these numbers "
                         "remain the pre-agentic reference point even though production runs CRAG.",
        "configuration": {
            "model": OLLAMA_MODEL,
            "crag_enabled": False,
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "num_ctx": OLLAMA_NUM_CTX,
            "chunk_context_length": OLLAMA_CONTEXT_LENGTH,
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
            "warmup_latency_sec": round(warmup_latency_sec, 3) if warmup_latency_sec is not None else None,
            "index_build_or_load_sec": round(index_build_sec, 3) if index_build_sec is not None else None
        },
        "latency": summarize_latencies(successful),
        "quality": {
            "criteria": list(QUALITY_CRITERIA),
            "scored_questions": len(scored),
            "pending_questions": len(records) - len(scored),
            "note": "Quality scores are filled in by hand by course instructors. This harness never auto-scores."
        }
    }


def append_record(path: str, record: Dict[str, Any]) -> None:
    """Append a single record to the results file and flush it to disk.

    Records are written one at a time so that a run interrupted after two hours
    of generation keeps everything it already measured.

    Args:
        path: Results file path.
        record: Record to append.
    """
    with open(path, 'a', encoding='utf-8') as results_file:
        results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        results_file.flush()


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


def prepare_context(document_location: str) -> Tuple[List[str], Any, float]:
    """Load the course documents and build or load the FAISS index.

    Args:
        document_location: Directory holding the course PDFs/DOCXs.

    Returns:
        Tuple[List[str], Any, float]: Context chunks, the ready processor, and the
        seconds spent building or loading the vector store.
    """
    logging.info(f"Loading course documents from '{document_location}'")
    document = LocalDocumentLoader(document_location).load_document()
    # enable_crag=False keeps this a classic-RAG measurement: linear retrieve ->
    # generate, no relevance grading, no reformulation, control model never loaded.
    llm = build_ollama_processor(enable_crag=False)
    context_chunked = document.get_chunked_text(llm.context_length)
    logging.info(f"Document split into {len(context_chunked)} chunks")

    logging.info("Building or loading the FAISS index")
    index_start = time.perf_counter()
    llm.build_or_load_vectorstore(context_chunked)
    index_build_sec = time.perf_counter() - index_start
    logging.info(f"FAISS index ready in {index_build_sec:.1f} s")
    return context_chunked, llm, index_build_sec


def measure(llm: Any, question: str, context_chunked: List[str]) -> Tuple[str, float]:
    """Run one question through the pipeline and time it.

    Args:
        llm: Processor returned by ``build_ollama_processor``.
        question: Question text to send.
        context_chunked: Context chunks backing the vector store.

    Returns:
        Tuple[str, float]: The answer and the wall-clock seconds it took.
    """
    start = time.perf_counter()
    answer = llm.process_questionnaire(question, context_chunked)
    latency_sec = time.perf_counter() - start
    return answer if isinstance(answer, str) else str(answer), latency_sec


def print_summary(summary: Dict[str, Any]) -> None:
    """Print the run summary to stdout.

    Args:
        summary: Summary produced by ``build_summary``.
    """
    latency = summary['latency']
    run = summary['run']
    print("\n" + "=" * 72)
    print(f"Baseline run - {summary['configuration']['model']} - k={summary['configuration']['retrieval_k']}")
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
    print(f"Quality scores     : {summary['quality']['pending_questions']} question(s) still awaiting "
          f"instructor scoring")
    print("=" * 72 + "\n")


def run_baseline(args: argparse.Namespace) -> int:
    """Execute the baseline run.

    Args:
        args: Parsed command line arguments.

    Returns:
        int: Process exit code.
    """
    entries = load_dataset(args.dataset)
    logging.info(f"Loaded {len(entries)} question(s) from '{args.dataset}'")

    if args.limit is not None:
        entries = entries[:args.limit]
        logging.warning(f"--limit is set: only the first {len(entries)} question(s) will run. "
                        f"A partial run is a smoke test, not a baseline.")

    if args.resume:
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
        print(f"Results would be written to '{args.output}'.")
        print(f"Configuration: model={OLLAMA_MODEL}, k={RETRIEVAL_K}, num_ctx={OLLAMA_NUM_CTX}, crag=disabled.")
        return 0

    if not entries:
        logging.info("Nothing left to measure; recomputing the summary from the existing results file")
        summary = build_summary(load_records(args.output), args, None, None, None)
        write_summary(args.output, summary)
        print_summary(summary)
        return 0

    check_ollama(args.ollama_url)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    context_chunked, llm, index_build_sec = prepare_context(args.document_location)

    warmup_latency_sec: Optional[float] = None
    if args.warmup:
        logging.info("Warm-up call so the first measured question does not absorb the model load time")
        _, warmup_latency_sec = measure(llm, WARMUP_QUESTION, context_chunked)
        logging.info(f"Warm-up finished in {warmup_latency_sec:.1f} s (discarded)")

    for position, entry in enumerate(entries, start=1):
        entry_id = str(entry['id'])
        logging.info(f"[{position}/{len(entries)}] Measuring question '{entry_id}'")
        answer, latency_sec = measure(llm, entry['question'], context_chunked)
        if answer.startswith(PIPELINE_ERROR_PREFIX):
            logging.error(f"Question '{entry_id}' failed inside the pipeline: {answer}")
        logging.info(f"Question '{entry_id}' answered in {latency_sec:.1f} s")
        append_record(args.output, build_record(entry, answer, latency_sec))

    summary = build_summary(load_records(args.output), args, warmup_latency_sec, index_build_sec,
                            len(context_chunked))
    write_summary(args.output, summary)
    print_summary(summary)
    return 0


def write_summary(output_path: str, summary: Dict[str, Any]) -> str:
    """Write the run summary next to the results file.

    Args:
        output_path: Path of the results JSON Lines file.
        summary: Summary produced by ``build_summary``.

    Returns:
        str: Path of the summary file that was written.
    """
    summary_path = f"{os.path.splitext(output_path)[0]}.summary.json"
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")
    logging.info(f"Summary written to '{summary_path}'")
    return summary_path


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: Parsed arguments with defaults resolved.
    """
    parser = argparse.ArgumentParser(
        description="Measure per-question latency of the classic RAG pipeline against the evaluation dataset.",
        epilog="Quality criteria are never scored automatically; the results file leaves them null for instructors."
    )
    parser.add_argument('--dataset', default=DEFAULT_DATASET_PATH,
                        help=f"Evaluation dataset in JSON Lines format (default: {DEFAULT_DATASET_PATH})")
    parser.add_argument('--output', default=None,
                        help=f"Results file (default: {build_output_path(OLLAMA_MODEL, RETRIEVAL_K)})")
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
        args.output = build_output_path(OLLAMA_MODEL, RETRIEVAL_K)
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
        return run_baseline(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as err:
        logging.error(str(err))
        return 1
    except KeyboardInterrupt:
        logging.warning(f"Interrupted. Results measured so far are in '{args.output}'; rerun with --resume.")
        return 130


if __name__ == '__main__':
    sys.exit(main())
