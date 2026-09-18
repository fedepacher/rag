"""Results-file format shared by the evaluation harness and the comparison tool.

This module is the single definition of what a results file looks like: how it is
named, how a record is read back, and how latency is summarised. Both
``run_baseline.py`` (which produces the files) and ``compare_runs.py`` (which reads
them) import from here so the two can never drift into disagreeing about the format.

It deliberately imports **nothing from ``rag/``** and nothing outside the standard
library. Producing a results file needs Ollama, FAISS, LangGraph and a 16 GB box;
reading two committed results files and diffing them must not. Keeping this layer
dependency-free is what lets an instructor run ``compare_runs.py`` on a laptop that
could never run the pipeline itself.
"""

import json
import logging
import os
import statistics
import subprocess
from typing import Any, Dict, List, Optional, Set

# Version of the per-record and per-summary shape written by run_baseline.py.
#
#   1 - Issue #13. Classic-pipeline records only: no `schema_version`, no `pipeline`,
#       no `out_of_scope`, no `crag` block. No file of this shape was ever committed to
#       resources/eval/results/ (no run has happened yet), so v1 exists as a label for
#       records that predate the A/B tooling rather than as a format still in use.
#   2 - Issue #16. Adds `schema_version`, `pipeline`, `out_of_scope` and `crag` so a
#       classic run and a CRAG run are directly comparable record by record.
#   3 - Renames the config key `chunk_context_length` to `chunk_size_tokens`. The old name
#       described the value as a context length when it is the chunk size the splitter
#       counts in cl100k_base tokens, and that misreading is what let chunks grow to 3.4x
#       the model's context window. No results file in this repo used the old key.
#   4 - Issue #32. Adds a `provenance` block to every record and summary: the commit, a
#       dirty-tree flag and the corpus hash. Four runs of the same six questions had been
#       compared as repeats of one system when every transition between them spanned a
#       functional commit, and one of those commits deleted a course document. Without
#       provenance a results file cannot say which system produced it, so a comparison
#       cannot know whether it is measuring a change or a different program.
#
# Readers must treat a missing `schema_version` as 1, a missing `crag` block as
# "not measured" rather than as zero iterations, and a missing `provenance` block as
# entirely unknown rather than as matching whatever it is compared against.
SCHEMA_VERSION = 4

# Provenance is read from the environment before git is consulted, because the harness
# runs as `docker compose exec rag python resources/eval/run_baseline.py` and the rag
# image ships no git binary and no .git directory -- the source is COPYed in at build
# time. Shelling out there could only ever produce nulls. The commit the image was
# built from, injected as a build arg, is both available and the honest answer for a
# run of that image.
COMMIT_ENV_VAR = "RAG_COMMIT"
DIRTY_ENV_VAR = "RAG_DIRTY"

# Fields of the provenance block, in the order a mismatch is reported.
PROVENANCE_FIELDS = ("commit", "dirty", "corpus_hash")

RESULTS_DIR = os.path.join("resources", "eval", "results")

# Pipeline labels written into every record and summary. A results file states which
# pipeline produced it instead of relying on its filename, so a renamed or relocated
# file cannot be mistaken for the other arm of the comparison.
PIPELINE_CLASSIC = "classic-rag"
PIPELINE_CRAG = "crag"

# Filename prefix per pipeline. The classic prefix is "baseline" and not "classic"
# because results/baseline_<model>_k<k>.jsonl is the name issue #13 established and
# BASELINE.md documents; renaming it would orphan the reference point.
PIPELINE_FILE_PREFIX = {PIPELINE_CLASSIC: "baseline", PIPELINE_CRAG: "crag"}

# Criteria scored by a human reviewer after the run. Written as null so an unscored
# record is impossible to mistake for a zero.
QUALITY_CRITERIA = ("pertinencia", "claridad", "precision", "lenguaje")

# LLMProcessorOllama swallows its own exceptions and returns a message with this prefix
# instead of raising, so a failed question has to be detected by inspecting the answer.
PIPELINE_ERROR_PREFIX = "Error:"


def _git_output(args: List[str], repo_root: Optional[str]) -> Optional[str]:
    """Run a read-only git command, returning None instead of raising.

    Args:
        args: Git arguments, without the leading ``git``.
        repo_root: Directory to run in, or None for the process working directory.

    Returns:
        Optional[str]: Stripped stdout, or None when git is missing, the directory is
        not a repository, or the command fails for any other reason.
    """
    try:
        completed = subprocess.run(["git", *args], cwd=repo_root, capture_output=True,
                                   text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def git_provenance(repo_root: Optional[str] = None) -> Dict[str, Optional[Any]]:
    """Identify the code a run was produced by.

    Reads ``RAG_COMMIT`` / ``RAG_DIRTY`` first and falls back to git. See
    ``COMMIT_ENV_VAR`` for why the environment has to come first.

    Never raises: a run that cannot identify itself is worth less than one that can,
    but the measurement is the expensive part and must not be lost over it. Recording
    null is also honest, and ``provenance_mismatch`` treats null as "unknown" rather
    than as "matches", so an unidentified run cannot quietly be compared to anything.

    Args:
        repo_root: Repository to inspect when falling back to git. None uses the
            process working directory.

    Returns:
        Dict[str, Optional[Any]]: ``commit`` (40-char hex or None) and ``dirty``
        (bool or None). ``dirty`` is None when it could not be established, which is
        deliberately not the same as False -- defaulting to clean would let a build
        from a modified tree claim its SHA identifies it.
    """
    commit = os.getenv(COMMIT_ENV_VAR) or None
    if commit:
        dirty_raw = os.getenv(DIRTY_ENV_VAR)
        dirty = None if dirty_raw in (None, "") else dirty_raw.strip() not in ("0", "false", "False")
        return {"commit": commit, "dirty": dirty}

    commit = _git_output(["rev-parse", "HEAD"], repo_root) or None
    if commit is None:
        return {"commit": None, "dirty": None}
    # --untracked-files=no on purpose: the course PDFs are untracked by design (#35),
    # so counting them would report every single run as dirty and the flag would carry
    # no information at all. A corpus change is caught by the corpus hash instead; what
    # makes a commit SHA a lie is a modified *tracked* file.
    status = _git_output(["status", "--porcelain", "--untracked-files=no"], repo_root)
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def build_provenance(corpus_hash: Optional[str],
                     repo_root: Optional[str] = None) -> Dict[str, Optional[Any]]:
    """Assemble the provenance block written into records and summaries.

    Args:
        corpus_hash: ``LLMProcessorOllama.get_index_hash`` over the chunks this run
            retrieved from, or None when it was not available. The corpus hash is the
            field a commit SHA cannot replace: the course documents are untracked, so
            the corpus can change with no commit to show for it -- which is exactly
            what ``dbed292`` did when it removed a PDF between two measured runs.
        repo_root: Passed through to ``git_provenance``.

    Returns:
        Dict[str, Optional[Any]]: ``commit``, ``dirty`` and ``corpus_hash``.
    """
    return {**git_provenance(repo_root), "corpus_hash": corpus_hash}


def record_provenance(record: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    """Read the provenance block back out of a record or summary.

    Args:
        record: A single result record or a run summary.

    Returns:
        Dict[str, Optional[Any]]: Every provenance field, with anything absent read as
        None. Schema 3 and earlier carry no block at all and therefore read as entirely
        unknown, which is the truth about them.
    """
    stored = record.get('provenance') or {}
    return {field: stored.get(field) for field in PROVENANCE_FIELDS}


def provenance_mismatch(left: Dict[str, Optional[Any]],
                        right: Dict[str, Optional[Any]]) -> List[str]:
    """Report which provenance fields make two runs incomparable.

    An unknown identity counts as a mismatch. Treating null as "matches" is precisely
    the mistake this guard exists to prevent: it is what allowed four runs on four
    different commits to be read as four repeats of one system.

    ``dirty`` is only reported when it is known to be True on either side. A dirty tree
    means the commit SHA is not an identity, so two such runs are not known to be the
    same system even at the same SHA. An *unknown* dirty flag is not reported, because
    a containerised run records one whenever no dirty build arg was passed, and firing
    on every such comparison would train the operator to pass the override by reflex.

    Args:
        left: Provenance of one run.
        right: Provenance of the other.

    Returns:
        List[str]: Offending field names in ``PROVENANCE_FIELDS`` order. Empty when the
        two runs are known to have come from the same system.
    """
    mismatched: List[str] = []
    for field in PROVENANCE_FIELDS:
        if field == 'dirty':
            if left.get('dirty') is True or right.get('dirty') is True:
                mismatched.append(field)
            continue
        left_value, right_value = left.get(field), right.get(field)
        if left_value is None or right_value is None or left_value != right_value:
            mismatched.append(field)
    return mismatched


def pipeline_label(crag_enabled: bool) -> str:
    """Return the pipeline label matching a CRAG switch.

    Args:
        crag_enabled: Whether the CRAG correction loop was wired in.

    Returns:
        str: ``PIPELINE_CRAG`` or ``PIPELINE_CLASSIC``.
    """
    return PIPELINE_CRAG if crag_enabled else PIPELINE_CLASSIC


def build_output_path(model: str, retrieval_k: int, crag_enabled: bool = False) -> str:
    """Build the default results path for a pipeline/model/retrieval-depth combination.

    Args:
        model: Ollama model tag being measured.
        retrieval_k: Number of chunks pulled from the vector store per question.
        crag_enabled: Whether the CRAG correction loop was active during the run.

    Returns:
        str: Path such as ``resources/eval/results/baseline_<model>_k4.jsonl`` for the
        classic pipeline, or ``resources/eval/results/crag_<model>_k4.jsonl`` for CRAG.
        The two arms therefore never write to the same file, and neither can silently
        overwrite the other.
    """
    model_slug = model.replace(":", "-").replace("/", "-")
    prefix = PIPELINE_FILE_PREFIX[pipeline_label(crag_enabled)]
    return os.path.join(RESULTS_DIR, f"{prefix}_{model_slug}_k{retrieval_k}.jsonl")


def summary_path_for(output_path: str) -> str:
    """Return the summary path that sits next to a results file.

    Args:
        output_path: Path of the results JSON Lines file.

    Returns:
        str: Same path with the extension replaced by ``.summary.json``.
    """
    return f"{os.path.splitext(output_path)[0]}.summary.json"


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
    for record in load_records(path):
        try:
            recorded.add(str(record['id']))
        except (KeyError, TypeError):
            logging.warning(f"Skipping record without an id while scanning '{path}' for resumable ids")
    return recorded


def load_summary(results_path: str) -> Optional[Dict[str, Any]]:
    """Read the summary written next to a results file.

    Args:
        results_path: Path of the results JSON Lines file.

    Returns:
        Optional[Dict[str, Any]]: Parsed summary, or None when it is absent or
        unreadable. A missing summary is not fatal: the records alone carry enough
        information to compare two runs, the summary only adds the configuration.
    """
    summary_path = summary_path_for(results_path)
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path, 'r', encoding='utf-8') as summary_file:
            return json.load(summary_file)
    except (json.JSONDecodeError, OSError) as err:
        logging.warning(f"Could not read the summary '{summary_path}': {err}")
        return None


def record_pipeline(record: Dict[str, Any]) -> str:
    """Return the pipeline a record was produced by.

    Args:
        record: A single result record.

    Returns:
        str: The recorded pipeline label. Schema v1 records carry none, and those
        could only have come from the classic harness, so they default to
        ``PIPELINE_CLASSIC``.
    """
    return str(record.get('pipeline') or PIPELINE_CLASSIC)


def is_successful(record: Dict[str, Any]) -> bool:
    """Report whether a record holds a real answer rather than a pipeline failure.

    An out-of-scope answer counts as a success: refusing to answer a question the
    bibliography does not cover is a valid outcome of the CRAG pipeline, not a
    malfunction, and dropping those records would hide exactly the behaviour the A/B
    test exists to measure.

    Args:
        record: A single result record.

    Returns:
        bool: True when the pipeline answered without erroring.
    """
    return not record.get('pipeline_error', False)


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


def write_summary(output_path: str, summary: Dict[str, Any]) -> str:
    """Write the run summary next to the results file.

    Args:
        output_path: Path of the results JSON Lines file.
        summary: Summary produced by the harness.

    Returns:
        str: Path of the summary file that was written.
    """
    summary_path = summary_path_for(output_path)
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")
    logging.info(f"Summary written to '{summary_path}'")
    return summary_path
