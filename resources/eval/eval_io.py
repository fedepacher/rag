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
from typing import Any, Dict, List, Optional, Set

# Version of the per-record and per-summary shape written by run_baseline.py.
#
#   1 - Issue #13. Classic-pipeline records only: no `schema_version`, no `pipeline`,
#       no `out_of_scope`, no `crag` block. No file of this shape was ever committed to
#       resources/eval/results/ (no run has happened yet), so v1 exists as a label for
#       records that predate the A/B tooling rather than as a format still in use.
#   2 - Issue #16. Adds `schema_version`, `pipeline`, `out_of_scope` and `crag` so a
#       classic run and a CRAG run are directly comparable record by record.
#
# Readers must treat a missing `schema_version` as 1 and a missing `crag` block as
# "not measured" rather than as zero iterations.
SCHEMA_VERSION = 2

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
