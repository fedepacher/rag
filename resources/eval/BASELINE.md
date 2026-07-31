# Baseline measurement — classic RAG pipeline

Harness that measures the **classic** pipeline (FAISS retrieval at `k=4` + a
single Llama 3.1 8B generation call, no relevance grading, no reformulation) so
the CRAG/Self-RAG work has a number to be compared against.

> **The pipeline moved under this harness and the harness was pinned in place.**
> Production enabled the CRAG correction loop in the same `ask_question` this
> script calls. `prepare_context` therefore builds its processor with an explicit
> `enable_crag`, defaulting to `False`, and the summary records
> `configuration.crag_enabled`. Do not change that default: with CRAG on, an
> invocation with no flags would print `classic-rag` while measuring a slower,
> three-model-call pipeline, and the pre-agentic number — which has never been
> recorded — would become unmeasurable.

> **The same script also measures the CRAG arm.** Pass `--crag` and it runs the
> agentic pipeline, records per-question iteration counts, and writes to
> `results/crag_<model>_k<k>.jsonl` instead. One harness, one measurement
> methodology, two comparable files. See **[`AB_TESTING.md`](AB_TESTING.md)** for
> the full A/B workflow and `compare_runs.py`. This document covers the classic
> arm, which is what the rest of it assumes.

> **Status: no baseline has been measured yet.** This directory contains tooling
> only. Two things are still missing and neither can be produced by this repo:
>
> 1. `dataset.jsonl` — written and validated by the course instructors (see
>    `README.md`). The runner refuses to start without it.
> 2. The four quality scores — assigned by hand by an instructor after the run.
>    The harness never scores answers itself.
>
> No latency figure and no quality score in this repository is real until
> `results/` contains output produced by an actual run on the target hardware.

## Quick path

The run has to happen **inside the `rag` container**, on the target hardware: it
needs the Ollama server, the pulled model, and the course documents under
`resources/files/`.

```bash
# 1. Bring the stack up and let Ollama finish pulling the model
docker compose up --build rag

# 2. Validate the dataset without touching the model (fast, safe to repeat)
docker compose exec rag python resources/eval/run_baseline.py --dry-run

# 3. Run the real measurement
docker compose exec rag python resources/eval/run_baseline.py
```

Expect this to take a while: on the 16 GB target box a single 8B answer is in the
tens of seconds, so a 50-question dataset is a coffee-and-then-some run. If it is
interrupted, everything already measured is on disk — resume with:

```bash
docker compose exec rag python resources/eval/run_baseline.py --resume
```

## For the run to be a valid baseline

The point of the exercise is a number someone can trust six months from now.

- [ ] The box is otherwise **idle**. The `rag` container's own polling loop keeps
      hitting Ollama; stop it (or run on a box where it is not serving) so the
      measured latency is not competing with a real student question.
- [ ] The **whole** dataset ran. `--limit` is a smoke test, not a baseline.
- [ ] The warm-up call ran (it does by default). Without it the first question
      silently absorbs the model load time and skews the minimum.
- [ ] `questions_failed` in the summary is `0`. A failed question returns fast and
      would otherwise flatter the average — the harness excludes failures from the
      statistics and lists their ids.
- [ ] Both output files were committed, so the configuration that produced the
      numbers stays attached to them.

## Output

Two files, named after the model and retrieval depth that produced them:

| Path | Content |
|------|---------|
| `results/baseline_<model>_k<k>.jsonl` | One record per question: the generated answer, its latency, and the empty quality scores |
| `results/baseline_<model>_k<k>.summary.json` | Run configuration and the latency statistics |

With today's configuration that resolves to
`results/baseline_llama3.1-8b-instruct-q4_K_M_k4.jsonl`. The name is derived from
the live configuration rather than hardcoded, so changing the model or `k` writes
to a new file instead of overwriting an existing baseline. `--crag` swaps the
`baseline_` prefix for `crag_`, so the two arms can never collide.

### Per-question record

| Field | Written by | Description |
|-------|-----------|-------------|
| `schema_version` | runner | Version of this record shape, currently `2` |
| `id`, `question`, `expected_answer`, `source_doc`, `topic`, `difficulty` | runner | Copied from the dataset entry so the record stands alone |
| `pipeline` | runner | `classic-rag` or `crag`. The record states what produced it, so a renamed file cannot be misread |
| `generated_answer` | runner | The pipeline's answer, verbatim |
| `latency_sec` | runner | Wall-clock seconds of the pipeline call — retrieval plus generation, excluding index build and warm-up |
| `pipeline_error` | runner | `true` when the pipeline returned an error instead of an answer |
| `out_of_scope` | runner | `true` when the answer is the CRAG out-of-scope refusal. Always `false` on this arm — the classic graph has no node that can produce it |
| `crag` | runner | Iteration counters. Always `null` on this arm; see [`AB_TESTING.md`](AB_TESTING.md) |
| `measured_at` | runner | ISO 8601 timestamp of the call |
| `scores` | **instructor** | The four criteria, `null` until scored |
| `scored_by`, `scored_at` | **instructor** | Who scored it and when |
| `reviewer_notes` | **instructor** | Free-form remarks |

`scores` starts as `null` rather than `0` on purpose: an unscored answer must be
impossible to mistake for a badly scored one. `crag` is `null` on a classic record
for the same reason — "the loop was not there" is not "the loop did nothing".

Issue #16 added `schema_version`, `pipeline`, `out_of_scope` and `crag` to this
shape (v1 → v2). No v1 file was ever committed, since no run has happened, so
nothing on disk needed migrating. Readers treat a missing `schema_version` as 1.

### Summary

`configuration` records the model, temperature, `top_p`, `num_ctx`, chunk length,
retrieval `k`, chunk count and `crag_enabled` actually used — read from the pipeline
constants, not re-declared — so a results file is self-describing and cannot be
mistaken for an agentic-pipeline run. `latency` holds count, min,
max, mean, median, p95, stdev and total, over successful questions only. `crag` is
`null` on this arm and holds the reformulation histogram on the other. The same
statistics are printed to stdout when the run ends.

## Scoring the quality criteria

After the run, an instructor edits the results file and fills `scores` on every
record. The criteria come from the course's own evaluation rubric:

| Criterion | Question it answers |
|-----------|---------------------|
| `pertinencia` | Does the answer address what was actually asked? |
| `claridad` | Is it understandable to the student who asked it? |
| `precision` | Is it factually correct and grounded in the course bibliography? |
| `lenguaje` | Is the Spanish correct and appropriate for the course register? |

Suggested scale: integer **1–5** per criterion, with `expected_answer` in the same
record as the reference. If the instructors adopt a different scale, record it here
so later comparisons are read against the same ruler.

Re-running the runner after scoring is safe: it never overwrites an existing
results file (pass `--output` or `--resume` deliberately), and the summary reports
how many questions are still awaiting scores.

## Options

| Flag | Purpose |
|------|---------|
| `--no-crag` | Measure the classic pipeline. **This is the default**, so a bare invocation still produces the baseline |
| `--crag` | Measure the CRAG pipeline instead, writing to `results/crag_*` — see [`AB_TESTING.md`](AB_TESTING.md) |
| `--dry-run` | Validate the dataset and print what would run; does not load the model |
| `--resume` | Append to an existing results file, skipping ids already recorded. Refuses if that file was recorded with the other pipeline |
| `--limit N` | Measure only the first N questions — smoke test only |
| `--no-warmup` | Skip the discarded warm-up call |
| `--dataset`, `--output` | Override the input and output paths |
| `--document-location`, `--ollama-url` | Override the defaults from `$DOCUMENT_LOCATION` / `$OLLAMA_SERVER_URL` |
| `--log-level` | Quieten the DEBUG logging that `rag/__init__.py` turns on at import |

## Next step

Once `dataset.jsonl` exists, run this harness twice — once bare for the baseline,
once with `--crag` — and diff the two files with `compare_runs.py`. The workflow,
the CRAG-only record fields and how to read the report are documented in
**[`AB_TESTING.md`](AB_TESTING.md)**.

A scored baseline is still the gate on the quality half of that comparison: the
report covers latency and short-circuit behaviour from the runs alone, but leaves
quality and hallucination rate blocked until instructors fill `scores` in both
files.
