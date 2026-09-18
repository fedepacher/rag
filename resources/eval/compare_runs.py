"""A/B comparison between a classic-RAG results file and a CRAG results file.

Reads the two files ``run_baseline.py`` produces — one per pipeline, same dataset, same
model, same retrieval depth — and writes a Markdown report answering the three latency
questions issue #16 asks (typical case, worst case at the iteration cap, out-of-scope
short-circuit) plus the error rate on both sides.

Two things this tool deliberately does not do:

* **It does not score answers.** Quality and hallucination rate are human judgements
  made against ``expected_answer``. An automated proxy — string overlap, an
  LLM-as-judge, "the answer is short so it is probably a refusal" — would produce a
  number that looks like evidence and is not, and it would be the number that ends up
  in the report. The quality section prints what the instructors still have to fill in
  and computes score deltas only once they have.
* **It does not run anything.** It reads committed results files, imports only the
  standard library and ``eval_io``, and therefore runs on any machine, not just the one
  that can host Ollama.

Latency is compared over the *paired* subset: questions present and successful in both
files. Comparing the full files instead would let a question that only one arm answered
move the means, and the report would attribute to CRAG a difference that is really a
difference in which questions were measured.

Usage::

    python resources/eval/compare_runs.py                       # auto-discover in results/
    python resources/eval/compare_runs.py --baseline A --crag B
    python resources/eval/compare_runs.py --output resources/eval/results/ab_report.md
"""

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

# resources/ is not a package, so the sibling results-format module is imported by
# directory rather than by dotted path. Nothing from rag/ is imported on purpose: this
# script must stay runnable without the pipeline's dependencies.
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)
REPO_ROOT = os.path.abspath(os.path.join(EVAL_DIR, os.pardir, os.pardir))

from eval_io import (PIPELINE_CLASSIC, PIPELINE_CRAG, PIPELINE_FILE_PREFIX,  # noqa: E402
                     QUALITY_CRITERIA, RESULTS_DIR, is_successful, load_records,
                     load_summary, provenance_mismatch, record_pipeline,
                     record_provenance, summarize_latencies)

# Flag that lets an operator compare two runs this tool refuses to pair. Named in the
# error message on purpose: someone who knows the two runs are comparable needs to be
# shown the way through, or they will edit the results files by hand instead.
FORCE_PROVENANCE_FLAG = "--force-mismatched-provenance"

# Configuration keys that must match for the two runs to be a fair comparison. A
# difference in any of them means the report would be measuring that difference as much
# as it measures CRAG, so they are checked and surfaced rather than assumed.
COMPARABLE_CONFIG_KEYS = ("model", "retrieval_k", "num_ctx", "temperature", "top_p",
                          "chunk_size_tokens", "dataset", "context_chunks")

LATENCY_METRICS = (("mean_sec", "Mean"), ("median_sec", "Median"), ("p95_sec", "p95"),
                   ("min_sec", "Min"), ("max_sec", "Max"), ("stdev_sec", "Stdev"))


def discover_results(results_dir: str, pipeline: str) -> List[str]:
    """Find the results files of one pipeline inside the results directory.

    Args:
        results_dir: Directory holding the committed results files.
        pipeline: Pipeline label to look for.

    Returns:
        List[str]: Matching ``.jsonl`` paths, sorted, summary files excluded.
    """
    pattern = os.path.join(results_dir, f"{PIPELINE_FILE_PREFIX[pipeline]}_*.jsonl")
    return sorted(path for path in glob.glob(pattern) if not path.endswith('.summary.json'))


def resolve_input(explicit: Optional[str], results_dir: str, pipeline: str) -> str:
    """Resolve the results file to read for one arm of the comparison.

    Args:
        explicit: Path given on the command line, or None to auto-discover.
        results_dir: Directory to search when auto-discovering.
        pipeline: Pipeline label being resolved.

    Returns:
        str: Path of the results file.

    Raises:
        FileNotFoundError: If the explicit path is missing, or nothing was discovered.
        ValueError: If auto-discovery is ambiguous, which it is as soon as more than one
            model or retrieval depth has been measured.
    """
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"Results file '{explicit}' does not exist")
        return explicit

    candidates = discover_results(results_dir, pipeline)
    if not candidates:
        raise FileNotFoundError(
            f"No {pipeline} results file found in '{results_dir}'. Produce one with "
            f"`python resources/eval/run_baseline.py"
            f"{' --crag' if pipeline == PIPELINE_CRAG else ''}`, or pass the path explicitly."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Auto-discovery found {len(candidates)} {pipeline} results files in '{results_dir}': "
            f"{[os.path.basename(path) for path in candidates]}. Pass the one you mean explicitly, "
            f"comparing across models or retrieval depths is not a CRAG-vs-classic comparison."
        )
    return candidates[0]


def index_by_id(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index result records by question id.

    Args:
        records: Result records of one run.

    Returns:
        Dict[str, Dict[str, Any]]: Records keyed by id. A duplicated id keeps the last
        occurrence, which is the one a resumed run appended most recently.
    """
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        record_id = str(record.get('id'))
        if record_id in indexed:
            logging.warning(f"Question id '{record_id}' appears more than once; keeping the last record")
        indexed[record_id] = record
    return indexed


def assert_pipelines(records: List[Dict[str, Any]], expected: str, path: str) -> None:
    """Check that a results file holds what the caller thinks it holds.

    Guards against the pairing that silently produces a report of zeroes: passing the
    same file twice, or passing a CRAG file as the baseline.

    Args:
        records: Records read from the file.
        expected: Pipeline label the file is being used as.
        path: Path of the file, for the error message.

    Raises:
        ValueError: If any record was produced by a different pipeline.
    """
    found = {record_pipeline(record) for record in records}
    if found - {expected}:
        raise ValueError(
            f"'{path}' is being compared as the {expected} arm but contains records from {sorted(found)}. "
            f"Check the --baseline / --crag arguments."
        )


def file_provenance(records: List[Dict[str, Any]], path: str) -> Dict[str, Optional[Any]]:
    """Return the single provenance block a results file was produced under.

    Args:
        records: Records read from the file.
        path: Path of the file, for the error message.

    Returns:
        Dict[str, Optional[Any]]: The provenance every record in the file shares.

    Raises:
        ValueError: If the records disagree. Resuming a run after a code change writes
            records from two systems into one file, and nothing else in this tool would
            notice -- the file would keep its name, its pipeline label and its ids.
    """
    distinct = {json.dumps(record_provenance(record), sort_keys=True) for record in records}
    if len(distinct) > 1:
        raise ValueError(
            f"'{path}' holds records from {len(distinct)} different systems. It was most likely resumed "
            f"across a code change, which makes the file itself incomparable; re-run it from scratch."
        )
    return record_provenance(records[0])


def assert_same_provenance(baseline_records: List[Dict[str, Any]],
                           crag_records: List[Dict[str, Any]],
                           baseline_path: str, crag_path: str,
                           force: bool = False) -> None:
    """Refuse to compare two runs that did not come from the same system.

    This is the guard issue #32 exists for. Four runs of the same six questions were
    read as repeats of one pipeline; every transition between them spanned a functional
    commit, and one of those commits deleted a course PDF, which changes retrieval
    invisibly. Every per-question difference attributed to variance had a code change
    behind it. A comparison that cannot state both runs came from the same code and the
    same corpus is not measuring CRAG, and should not be allowed to look as if it were.

    Args:
        baseline_records: Records of the classic arm.
        crag_records: Records of the CRAG arm.
        baseline_path: Path of the classic results file.
        crag_path: Path of the CRAG results file.
        force: Skip the check. For an operator who knows why the two differ.

    Raises:
        ValueError: If the two runs differ, or if either cannot identify itself.
    """
    baseline_provenance = file_provenance(baseline_records, baseline_path)
    crag_provenance = file_provenance(crag_records, crag_path)
    if force:
        return
    mismatched = provenance_mismatch(baseline_provenance, crag_provenance)
    if not mismatched:
        return
    detail = ", ".join(
        f"{field}: {baseline_provenance.get(field)!r} vs {crag_provenance.get(field)!r}"
        for field in mismatched
    )
    raise ValueError(
        f"'{baseline_path}' and '{crag_path}' did not come from the same system ({detail}). "
        f"A difference here is measured as if it were CRAG. Re-run both arms at one commit over one "
        f"corpus, or pass {FORCE_PROVENANCE_FLAG} if you know why they differ. Note that an unknown "
        f"value counts as a mismatch: results files written before schema 4 recorded no provenance."
    )


def percent_delta(baseline_value: Optional[float], crag_value: Optional[float]) -> Optional[float]:
    """Express the CRAG value as a percentage change over the baseline value.

    Args:
        baseline_value: Classic-pipeline metric.
        crag_value: CRAG-pipeline metric.

    Returns:
        Optional[float]: Signed percentage, or None when it is not defined.
    """
    if baseline_value in (None, 0) or crag_value is None:
        return None
    return round((crag_value - baseline_value) / baseline_value * 100, 1)


def format_seconds(value: Optional[float]) -> str:
    """Format a latency value for a Markdown table.

    Args:
        value: Seconds, or None when the statistic needed more samples than available.

    Returns:
        str: Formatted value, or an em dash.
    """
    return "—" if value is None else f"{value:.1f} s"


def format_delta(baseline_value: Optional[float], crag_value: Optional[float]) -> str:
    """Format the absolute and relative difference between two latency values.

    Args:
        baseline_value: Classic-pipeline metric.
        crag_value: CRAG-pipeline metric.

    Returns:
        str: Something like ``+12.4 s (+38.0%)``, or an em dash.
    """
    if baseline_value is None or crag_value is None:
        return "—"
    absolute = crag_value - baseline_value
    relative = percent_delta(baseline_value, crag_value)
    relative_text = "" if relative is None else f" ({relative:+.1f}%)"
    return f"{absolute:+.1f} s{relative_text}"


def pair_records(baseline: Dict[str, Dict[str, Any]],
                 crag: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split the two runs into the ids they share and the ids they do not.

    Args:
        baseline: Classic records indexed by id.
        crag: CRAG records indexed by id.

    Returns:
        Tuple[List[str], List[str], List[str], List[str]]: Ids in both runs, ids only in
        the baseline, ids only in the CRAG run, and the subset of shared ids where both
        arms answered without erroring — the only subset latency may be compared over.
    """
    shared = sorted(set(baseline) & set(crag))
    only_baseline = sorted(set(baseline) - set(crag))
    only_crag = sorted(set(crag) - set(baseline))
    comparable = [record_id for record_id in shared
                  if is_successful(baseline[record_id]) and is_successful(crag[record_id])]
    return shared, only_baseline, only_crag, comparable


def compare_latency(baseline: Dict[str, Dict[str, Any]], crag: Dict[str, Dict[str, Any]],
                    ids: List[str]) -> Dict[str, Any]:
    """Compute the latency statistics of both arms over the same questions.

    Args:
        baseline: Classic records indexed by id.
        crag: CRAG records indexed by id.
        ids: Question ids to include, normally the comparable subset.

    Returns:
        Dict[str, Any]: Both stat blocks plus the per-metric deltas.
    """
    baseline_stats = summarize_latencies([baseline[record_id]['latency_sec'] for record_id in ids])
    crag_stats = summarize_latencies([crag[record_id]['latency_sec'] for record_id in ids])
    deltas = {metric: {"absolute_sec": (None if baseline_stats.get(metric) is None or
                                        crag_stats.get(metric) is None
                                        else round(crag_stats[metric] - baseline_stats[metric], 3)),
                       "percent": percent_delta(baseline_stats.get(metric), crag_stats.get(metric))}
              for metric, _label in LATENCY_METRICS}
    return {"question_count": len(ids), "baseline": baseline_stats, "crag": crag_stats, "deltas": deltas}


def compare_by_iterations(baseline: Dict[str, Dict[str, Any]], crag: Dict[str, Dict[str, Any]],
                          ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Break the latency comparison down by how many reformulations CRAG performed.

    This is what separates the typical case from the worst case. Bucket ``0`` is a
    question the correction loop never touched, and its delta against the baseline is
    the price of the relevance grading call alone. The highest bucket is the worst case:
    the grading calls, the rewrites and the extra retrievals all together.

    Args:
        baseline: Classic records indexed by id.
        crag: CRAG records indexed by id.
        ids: Comparable question ids.

    Returns:
        Dict[str, Dict[str, Any]]: One comparison per reformulation count, keyed by the
        count as a string. Empty when the CRAG records carry no iteration data, which is
        the case for schema v1 files.
    """
    buckets: Dict[str, List[str]] = {}
    for record_id in ids:
        crag_block = crag[record_id].get('crag')
        if not crag_block:
            continue
        bucket = str(crag_block.get('reformulations', 0))
        buckets.setdefault(bucket, []).append(record_id)
    return {bucket: compare_latency(baseline, crag, bucket_ids)
            for bucket, bucket_ids in sorted(buckets.items())}


def compare_out_of_scope(baseline: Dict[str, Dict[str, Any]], crag: Dict[str, Dict[str, Any]],
                         shared_ids: List[str]) -> Dict[str, Any]:
    """Measure how often the out-of-scope short-circuit fired and what it replaced.

    For every question CRAG refused, the classic answer to the same question is looked
    up and classified as an error or a produced answer. No judgement is made about which
    outcome was better: a refusal can be the honest answer to a question the
    bibliography does not cover, or a regression against an answer the classic pipeline
    got right. Only an instructor reading both texts can tell, so the report lists the
    ids and leaves the call to them.

    Args:
        baseline: Classic records indexed by id.
        crag: CRAG records indexed by id.
        shared_ids: Ids present in both runs.

    Returns:
        Dict[str, Any]: Counts, rate and the per-question detail to review by hand.
    """
    answered = [record_id for record_id in shared_ids if is_successful(crag[record_id])]
    short_circuited = [record_id for record_id in answered if crag[record_id].get('out_of_scope')]
    detail = [{
        "id": record_id,
        "question": crag[record_id].get('question'),
        "crag_latency_sec": crag[record_id].get('latency_sec'),
        "baseline_latency_sec": baseline[record_id].get('latency_sec'),
        "baseline_outcome": "pipeline error" if not is_successful(baseline[record_id]) else "answered",
        "baseline_answer_chars": len(str(baseline[record_id].get('generated_answer') or ""))
    } for record_id in short_circuited]
    return {
        "crag_answered": len(answered),
        "crag_out_of_scope": len(short_circuited),
        "crag_out_of_scope_rate": round(len(short_circuited) / len(answered), 4) if answered else None,
        # The classic graph has no node that can produce this answer, so anything other
        # than zero here means the files are not what they claim to be.
        "baseline_out_of_scope": sum(1 for record_id in shared_ids if baseline[record_id].get('out_of_scope')),
        "questions": detail
    }


def compare_errors(baseline: Dict[str, Dict[str, Any]], crag: Dict[str, Dict[str, Any]],
                   shared_ids: List[str]) -> Dict[str, Any]:
    """Count pipeline failures on each side of the comparison.

    Args:
        baseline: Classic records indexed by id.
        crag: CRAG records indexed by id.
        shared_ids: Ids present in both runs.

    Returns:
        Dict[str, Any]: Failure counts and the ids that failed in each arm.
    """
    baseline_failed = [record_id for record_id in shared_ids if not is_successful(baseline[record_id])]
    crag_failed = [record_id for record_id in shared_ids if not is_successful(crag[record_id])]
    return {
        "baseline_failed": len(baseline_failed),
        "baseline_failed_ids": baseline_failed,
        "crag_failed": len(crag_failed),
        "crag_failed_ids": crag_failed
    }


def score_stats(records: Dict[str, Dict[str, Any]], ids: List[str]) -> Dict[str, Any]:
    """Aggregate the human quality scores of one arm.

    Args:
        records: Records indexed by id.
        ids: Question ids to include.

    Returns:
        Dict[str, Any]: How many questions are fully scored and, per criterion, the mean
        over the questions that carry a score. Nothing is inferred for the rest.
    """
    fully_scored = [record_id for record_id in ids
                    if all(records[record_id].get('scores', {}).get(criterion) is not None
                           for criterion in QUALITY_CRITERIA)]
    means: Dict[str, Optional[float]] = {}
    for criterion in QUALITY_CRITERIA:
        values = [records[record_id]['scores'][criterion] for record_id in fully_scored]
        means[criterion] = round(sum(values) / len(values), 2) if values else None
    return {"scored_questions": len(fully_scored), "pending_questions": len(ids) - len(fully_scored),
            "means": means}


def compare_quality(baseline: Dict[str, Dict[str, Any]], crag: Dict[str, Dict[str, Any]],
                    shared_ids: List[str]) -> Dict[str, Any]:
    """Compare the human quality scores, once there are any.

    Args:
        baseline: Classic records indexed by id.
        crag: CRAG records indexed by id.
        shared_ids: Ids present in both runs.

    Returns:
        Dict[str, Any]: Per-criterion means and deltas, plus a ``blocked`` flag that is
        True while either arm is still unscored.
    """
    baseline_scores = score_stats(baseline, shared_ids)
    crag_scores = score_stats(crag, shared_ids)
    blocked = baseline_scores['scored_questions'] == 0 or crag_scores['scored_questions'] == 0
    deltas = {criterion: (None if baseline_scores['means'][criterion] is None or
                          crag_scores['means'][criterion] is None
                          else round(crag_scores['means'][criterion] - baseline_scores['means'][criterion], 2))
              for criterion in QUALITY_CRITERIA}
    return {"blocked": blocked, "baseline": baseline_scores, "crag": crag_scores, "deltas": deltas}


def compare_configuration(baseline_summary: Optional[Dict[str, Any]],
                          crag_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Diff the configuration of the two runs.

    Args:
        baseline_summary: Summary of the classic run, if it was written.
        crag_summary: Summary of the CRAG run, if it was written.

    Returns:
        Dict[str, Any]: Per-key values and the list of keys that disagree. ``crag_enabled``
        is expected to differ and is not part of the comparison.
    """
    baseline_config = (baseline_summary or {}).get('configuration', {})
    crag_config = (crag_summary or {}).get('configuration', {})
    rows = {key: {"baseline": baseline_config.get(key), "crag": crag_config.get(key)}
            for key in COMPARABLE_CONFIG_KEYS}
    mismatched = [key for key, values in rows.items()
                  if values['baseline'] != values['crag'] and not (baseline_summary is None or crag_summary is None)]
    return {"available": baseline_summary is not None and crag_summary is not None,
            "rows": rows, "mismatched_keys": mismatched,
            "grader_model": crag_config.get('grader_model'),
            "max_crag_iterations": crag_config.get('max_crag_iterations')}


def build_comparison(baseline_path: str, crag_path: str,
                     force_mismatched_provenance: bool = False) -> Dict[str, Any]:
    """Read both results files and compute every section of the comparison.

    Args:
        force_mismatched_provenance: Compare two runs from different systems anyway.
            Defaults to refusing, because the refusal is the point of issue #32.
        baseline_path: Path of the classic results file.
        crag_path: Path of the CRAG results file.

    Returns:
        Dict[str, Any]: The full comparison, ready to render or serialise.

    Raises:
        ValueError: If a file is empty, mislabelled, or the two share no question.
    """
    baseline_records = load_records(baseline_path)
    crag_records = load_records(crag_path)
    if not baseline_records:
        raise ValueError(f"'{baseline_path}' holds no readable record")
    if not crag_records:
        raise ValueError(f"'{crag_path}' holds no readable record")
    assert_pipelines(baseline_records, PIPELINE_CLASSIC, baseline_path)
    assert_pipelines(crag_records, PIPELINE_CRAG, crag_path)
    assert_same_provenance(baseline_records, crag_records, baseline_path, crag_path,
                           force=force_mismatched_provenance)

    baseline = index_by_id(baseline_records)
    crag = index_by_id(crag_records)
    shared, only_baseline, only_crag, comparable = pair_records(baseline, crag)
    if not shared:
        raise ValueError(
            f"'{baseline_path}' and '{crag_path}' share no question id. Both arms have to run the same "
            f"dataset for the comparison to mean anything."
        )

    crag_has_iterations = any(record.get('crag') for record in crag_records)
    return {
        "generated_at": datetime.now().isoformat(timespec='seconds'),
        "inputs": {"baseline": baseline_path, "crag": crag_path},
        # Recorded on every comparison, not only on a forced one: a report read six
        # months from now has to say which system produced each arm, and the whole
        # defect behind #32 was numbers circulating without that context.
        "provenance": {
            "baseline": file_provenance(baseline_records, baseline_path),
            "crag": file_provenance(crag_records, crag_path),
            "forced": force_mismatched_provenance
        },
        "configuration": compare_configuration(load_summary(baseline_path), load_summary(crag_path)),
        "coverage": {
            "baseline_questions": len(baseline),
            "crag_questions": len(crag),
            "shared_questions": len(shared),
            "comparable_questions": len(comparable),
            "only_in_baseline_ids": only_baseline,
            "only_in_crag_ids": only_crag
        },
        "latency": compare_latency(baseline, crag, comparable),
        "iterations": compare_by_iterations(baseline, crag, comparable) if crag_has_iterations else {},
        "iterations_available": crag_has_iterations,
        "out_of_scope": compare_out_of_scope(baseline, crag, shared),
        "errors": compare_errors(baseline, crag, shared),
        "quality": compare_quality(baseline, crag, shared)
    }


def render_latency_table(comparison: Dict[str, Any]) -> List[str]:
    """Render one latency comparison as a Markdown table.

    Args:
        comparison: Block produced by ``compare_latency``.

    Returns:
        List[str]: Markdown lines.
    """
    baseline_stats = comparison['baseline']
    crag_stats = comparison['crag']
    lines = ["| Metric | Classic | CRAG | Delta |", "|--------|---------|------|-------|"]
    for metric, label in LATENCY_METRICS:
        lines.append(f"| {label} | {format_seconds(baseline_stats.get(metric))} | "
                     f"{format_seconds(crag_stats.get(metric))} | "
                     f"{format_delta(baseline_stats.get(metric), crag_stats.get(metric))} |")
    return lines


def render_provenance(provenance: Dict[str, Any]) -> List[str]:
    """Render the provenance section that labels every report.

    Args:
        provenance: The ``provenance`` block of a comparison.

    Returns:
        List[str]: Markdown lines, ending in a blank line.
    """
    def describe(block: Dict[str, Optional[Any]]) -> str:
        commit = block.get('commit')
        if commit is None:
            return "unrecorded (results file predates schema 4)"
        dirty = block.get('dirty')
        suffix = {True: " **+ uncommitted changes**", False: "", None: " (dirty flag unrecorded)"}[dirty]
        return f"`{commit[:12]}`{suffix}"

    def describe_corpus(block: Dict[str, Optional[Any]]) -> str:
        corpus_hash = block.get('corpus_hash')
        return f"`{corpus_hash[:12]}`" if corpus_hash else "unrecorded"

    baseline, crag = provenance['baseline'], provenance['crag']
    lines = [
        "## Provenance", "",
        "| | Classic | CRAG |",
        "|-|---------|------|",
        f"| Commit | {describe(baseline)} | {describe(crag)} |",
        f"| Corpus | {describe_corpus(baseline)} | {describe_corpus(crag)} |",
        ""
    ]
    if provenance.get('forced'):
        lines += [
            f"> **This comparison was forced with `{FORCE_PROVENANCE_FLAG}`.** The two arms did not come",
            "> from the same system, so any difference below is the sum of CRAG and whatever else changed",
            "> between them. Do not quote a number from this report as an effect of CRAG.", ""
        ]
    return lines


def render_report(comparison: Dict[str, Any]) -> str:
    """Render the comparison as a Markdown report.

    Args:
        comparison: Comparison produced by ``build_comparison``.

    Returns:
        str: The report.
    """
    coverage = comparison['coverage']
    configuration = comparison['configuration']
    lines: List[str] = [
        "# CRAG vs classic RAG — A/B comparison",
        "",
        f"Generated at {comparison['generated_at']} from:",
        "",
        f"- Classic: `{comparison['inputs']['baseline']}`",
        f"- CRAG: `{comparison['inputs']['crag']}`",
        ""
    ]

    lines += render_provenance(comparison['provenance'])
    lines += ["## Configuration", ""]
    if not configuration['available']:
        lines += ["> One of the two `.summary.json` files is missing, so the run configurations could not be",
                  "> checked against each other. The latency comparison below assumes both runs used the same",
                  "> model, retrieval depth and hardware.", ""]
    else:
        lines += ["| Key | Classic | CRAG |", "|-----|---------|------|"]
        for key, values in configuration['rows'].items():
            lines.append(f"| `{key}` | {values['baseline']} | {values['crag']} |")
        lines.append("")
        if configuration['mismatched_keys']:
            lines += [f"> **The two runs do not share the same {', '.join(configuration['mismatched_keys'])}.**",
                      "> Any difference reported below is a difference between two configurations, not between",
                      "> two pipelines. Re-run one arm to match the other before quoting these numbers.", ""]
        if configuration['grader_model']:
            lines += [f"CRAG control model: `{configuration['grader_model']}`, "
                      f"at most {configuration['max_crag_iterations']} reformulation(s) per question.", ""]

    lines += [
        "## Coverage", "",
        f"- Questions in the classic run: **{coverage['baseline_questions']}**",
        f"- Questions in the CRAG run: **{coverage['crag_questions']}**",
        f"- Present in both: **{coverage['shared_questions']}**",
        f"- Answered without error in both (the subset latency is compared over): "
        f"**{coverage['comparable_questions']}**",
        ""
    ]
    if coverage['only_in_baseline_ids']:
        lines += [f"> Only in the classic run: `{'`, `'.join(coverage['only_in_baseline_ids'])}`", ""]
    if coverage['only_in_crag_ids']:
        lines += [f"> Only in the CRAG run: `{'`, `'.join(coverage['only_in_crag_ids'])}`", ""]

    lines += ["## Latency", "",
              f"Over the {comparison['latency']['question_count']} question(s) both pipelines answered "
              f"successfully. A positive delta means CRAG is slower.", ""]
    lines += render_latency_table(comparison['latency'])
    lines.append("")

    lines += ["## Latency by correction-loop iterations", ""]
    if not comparison['iterations_available']:
        lines += ["> The CRAG results file carries no per-question iteration data (schema v1). Re-run the CRAG",
                  "> arm with the current harness to break the latency down into typical and worst case.", ""]
    elif not comparison['iterations']:
        lines += ["> No comparable question carried iteration data.", ""]
    else:
        lines += ["Bucket `n` holds the questions where the correction loop performed `n` reformulations. "
                  "Bucket `0` is the typical case: its delta is the cost of the relevance grading call alone. "
                  "The highest bucket is the worst case, where the extra retrievals and rewrites are paid too.",
                  ""]
        for bucket, block in comparison['iterations'].items():
            lines += [f"### {bucket} reformulation(s) — {block['question_count']} question(s)", ""]
            lines += render_latency_table(block)
            lines.append("")

    out_of_scope = comparison['out_of_scope']
    rate = out_of_scope['crag_out_of_scope_rate']
    lines += [
        "## Out-of-scope short-circuit", "",
        f"- CRAG answered **{out_of_scope['crag_answered']}** shared question(s)",
        f"- Of those, **{out_of_scope['crag_out_of_scope']}** were short-circuited as out of scope"
        f"{'' if rate is None else f' (**{rate * 100:.1f}%**)'}",
        ""
    ]
    if out_of_scope['baseline_out_of_scope']:
        lines += [f"> **{out_of_scope['baseline_out_of_scope']} classic record(s) also carry the out-of-scope "
                  f"answer.** The classic graph has no node that can produce it, so the two files are not what "
                  f"they claim to be. Check how they were generated before reading anything else here.", ""]
    if out_of_scope['questions']:
        lines += ["Each of these needs an instructor to read the classic answer and decide whether the refusal "
                  "was honest or a regression. This tool does not judge that.", "",
                  "| Question | Classic outcome | Classic answer length | Classic latency | CRAG latency |",
                  "|----------|-----------------|-----------------------|-----------------|--------------|"]
        for question in out_of_scope['questions']:
            lines.append(f"| `{question['id']}` | {question['baseline_outcome']} | "
                         f"{question['baseline_answer_chars']} chars | "
                         f"{format_seconds(question['baseline_latency_sec'])} | "
                         f"{format_seconds(question['crag_latency_sec'])} |")
        lines.append("")

    errors = comparison['errors']
    lines += ["## Pipeline errors", "",
              f"- Classic: **{errors['baseline_failed']}** failed question(s)"
              + (f" (`{'`, `'.join(errors['baseline_failed_ids'])}`)" if errors['baseline_failed_ids'] else ""),
              f"- CRAG: **{errors['crag_failed']}** failed question(s)"
              + (f" (`{'`, `'.join(errors['crag_failed_ids'])}`)" if errors['crag_failed_ids'] else ""),
              "",
              "Failures are excluded from the latency statistics: a question that errored returns fast and "
              "would otherwise flatter whichever arm broke.", ""]

    lines += render_quality_section(comparison['quality'])
    return "\n".join(lines) + "\n"


def render_quality_section(quality: Dict[str, Any]) -> List[str]:
    """Render the quality and hallucination section.

    Hallucination rate is never computed, in either branch. It is not derivable from a
    results file: deciding that an answer asserts something the bibliography does not
    support means reading the answer against ``expected_answer`` and against the source
    document. Any automatic proxy would be a guess wearing a percentage sign.

    Args:
        quality: Block produced by ``compare_quality``.

    Returns:
        List[str]: Markdown lines.
    """
    lines = ["## Answer quality", ""]
    if quality['blocked']:
        lines += [
            "> **Blocked on human scoring — no quality comparison is possible yet.**",
            ">",
            f"> Classic run: {quality['baseline']['pending_questions']} question(s) unscored.  ",
            f"> CRAG run: {quality['crag']['pending_questions']} question(s) unscored.",
            ">",
            "> The four criteria (`" + "`, `".join(QUALITY_CRITERIA) + "`) are filled in by hand by the course",
            "> instructors, in the `scores` object of each record, using `expected_answer` as the reference.",
            "> Neither this tool nor the harness ever scores an answer: an automated proxy would produce a",
            "> number that reads like evidence while measuring nothing. Score both files, then re-run this",
            "> script to get the per-criterion comparison.",
            ""
        ]
    else:
        lines += [f"Over the questions scored in both runs "
                  f"(classic: {quality['baseline']['scored_questions']}, "
                  f"CRAG: {quality['crag']['scored_questions']}). A positive delta favours CRAG.", "",
                  "| Criterion | Classic | CRAG | Delta |", "|-----------|---------|------|-------|"]
        for criterion in QUALITY_CRITERIA:
            baseline_mean = quality['baseline']['means'][criterion]
            crag_mean = quality['crag']['means'][criterion]
            delta = quality['deltas'][criterion]
            lines.append(f"| {criterion} | {'—' if baseline_mean is None else baseline_mean} | "
                         f"{'—' if crag_mean is None else crag_mean} | "
                         f"{'—' if delta is None else f'{delta:+.2f}'} |")
        lines += ["",
                  "> These are means of instructor-assigned scores. If the two runs were scored by different",
                  "> people or against different scales, the delta measures the reviewers, not the pipelines.",
                  ""]

    lines += [
        "## Hallucination rate", "",
        "**Not computed, by design.** A hallucination is an assertion the course bibliography does not",
        "support, which can only be identified by reading the generated answer against `expected_answer`",
        "and the cited `source_doc`. There is no field in a results file from which it can be derived, and",
        "estimating it heuristically would put a fabricated number next to real measurements.",
        "",
        "To produce it: have an instructor mark each answer as grounded or not while scoring the four",
        "criteria, record the verdict in `reviewer_notes`, and count the marks per arm over the same",
        "question set.",
        ""
    ]
    return lines


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare a CRAG results file against the classic-RAG baseline produced by run_baseline.py.",
        epilog="Quality and hallucination rate depend on instructor scoring; this tool never scores answers."
    )
    parser.add_argument('--baseline', default=None,
                        help="Classic results file (default: the single baseline_*.jsonl in the results dir)")
    parser.add_argument('--crag', default=None,
                        help="CRAG results file (default: the single crag_*.jsonl in the results dir)")
    parser.add_argument('--results-dir', default=RESULTS_DIR,
                        help=f"Directory searched when a path is not given (default: {RESULTS_DIR})")
    parser.add_argument('--output', default=None,
                        help="Write the Markdown report to this file as well as to stdout")
    parser.add_argument('--json', dest='json_output', default=None,
                        help="Write the computed comparison as JSON to this file")
    parser.add_argument(FORCE_PROVENANCE_FLAG, dest='force_mismatched_provenance',
                        action='store_true',
                        help="Compare two runs that did not come from the same commit or corpus. "
                             "The report is labelled as forced and its numbers cannot be attributed "
                             "to CRAG alone.")
    parser.add_argument('--quiet', action='store_true',
                        help="Do not print the report to stdout (use with --output or --json)")
    parser.add_argument('--log-level', default='WARNING',
                        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR'),
                        help="Logging verbosity (default: WARNING)")
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> int:
    """Entry point.

    Returns:
        int: Process exit code.
    """
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    # Results paths are repository-relative, like everywhere else in this directory.
    os.chdir(REPO_ROOT)

    try:
        baseline_path = resolve_input(args.baseline, args.results_dir, PIPELINE_CLASSIC)
        crag_path = resolve_input(args.crag, args.results_dir, PIPELINE_CRAG)
        comparison = build_comparison(baseline_path, crag_path,
                                      force_mismatched_provenance=args.force_mismatched_provenance)
    except (FileNotFoundError, ValueError) as err:
        logging.error(str(err))
        return 1

    report = render_report(comparison)
    if not args.quiet:
        print(report)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as report_file:
            report_file.write(report)
        logging.warning(f"Report written to '{args.output}'")
    if args.json_output:
        os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
        with open(args.json_output, 'w', encoding='utf-8') as json_file:
            json.dump(comparison, json_file, ensure_ascii=False, indent=2)
            json_file.write("\n")
        logging.warning(f"Comparison JSON written to '{args.json_output}'")
    return 0


if __name__ == '__main__':
    sys.exit(main())
