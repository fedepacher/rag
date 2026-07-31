# A/B testing — CRAG vs classic RAG

Tooling to measure both pipelines against the same dataset and diff the results.
`run_baseline.py` produces one results file per pipeline; `compare_runs.py` reads
the two and writes a Markdown report.

> **Status: no A/B test has been run.** This directory contains tooling only.
> Nothing under `results/` exists yet, and no latency, quality or hallucination
> figure in this repository is real. Two blockers, neither of which this repo can
> unblock:
>
> 1. `dataset.jsonl` — written and validated by the course instructors (see
>    `README.md`). Both runners refuse to start without it.
> 2. The four quality scores — assigned by hand after the runs. The comparison
>    reports latency and short-circuit behaviour today; the quality section stays
>    blocked until instructors fill `scores` in both files.

## Quick path

Everything except the comparison has to happen **inside the `rag` container**, on
the target hardware: it needs Ollama, both pulled models and the course documents.

```bash
# 1. Classic arm — the pre-agentic reference point (this is the default, no flag)
docker compose exec rag python resources/eval/run_baseline.py

# 2. CRAG arm — same dataset, same harness, correction loop on
docker compose exec rag python resources/eval/run_baseline.py --crag

# 3. Compare. Reads committed files only, so it runs anywhere
python resources/eval/compare_runs.py --output resources/eval/results/ab_report.md
```

Steps 1 and 2 write to different files and cannot collide. Step 3 auto-discovers
both as long as there is exactly one of each in `results/`.

Validate before committing to hours of generation:

```bash
docker compose exec rag python resources/eval/run_baseline.py --crag --dry-run
```

## What each arm measures

| Arm | Flag | Pipeline | Output |
|-----|------|----------|--------|
| Classic | *(default)* or `--no-crag` | `retrieve → generate`. Control model never loaded. | `results/baseline_<model>_k<k>.jsonl` |
| CRAG | `--crag` | `retrieve → grade → (reformulate → retrieve)* → generate \| out_of_scope` | `results/crag_<model>_k<k>.jsonl` |

Both arms run through the same `measure()` on the same `process_questionnaire`
entry point, which is why this is one script and not two. A comparison between two
harnesses would be partly a comparison between two measurement methodologies.

The classic arm stays the default so that an invocation with no flags keeps
producing the baseline, exactly as `BASELINE.md` documented before the A/B tooling
landed.

## For the comparison to mean anything

- [ ] **Same dataset, both arms.** `--limit` is a smoke test. The report lists ids
      that appear in only one run and excludes them from the latency statistics.
- [ ] **Same box, same conditions, nothing else running.** The `rag` container's
      polling loop competes with the measurement for Ollama.
- [ ] **Same model and `k`.** The report diffs the two `configuration` blocks and
      prints a warning banner when they disagree — at that point the numbers
      measure the configuration difference as much as the pipeline difference.
- [ ] **Both arms warmed up** (the default). CRAG loads a second model; without a
      warm-up its first question absorbs the Phi load time.
- [ ] **`questions_failed` is 0 in both summaries.** A failed question returns fast
      and would flatter whichever arm broke.
- [ ] **All four files committed** — both `.jsonl` and both `.summary.json` — so the
      configuration stays attached to the numbers.

## Ordering constraint: run classic first

The CRAG arm keeps two models resident (~8.5–9 GiB on the 16 GB box). Running it
first leaves Phi in the Ollama cache while the classic arm runs, which changes the
memory pressure the classic arm is measured under. Run classic first, on a fresh
container, and the baseline is measured in the conditions it describes.

## Per-question CRAG record

The CRAG arm records how the correction loop behaved on every question. Without
this the report could not answer two of the three questions issue #16 asks —
worst-case latency and short-circuit rate are properties of the *walk through the
graph*, not of the answer text.

| Field | Description |
|-------|-------------|
| `crag.reformulations` | Query rewrites performed, 0 to `MAX_CRAG_ITERATIONS` |
| `crag.retrievals` | FAISS calls, always `reformulations + 1` |
| `crag.relevance_grades` | Phi grading calls |
| `crag.relevance_verdicts` | Every verdict in order, e.g. `["irrelevante", "relevante"]` |
| `crag.hit_iteration_cap` | The loop exhausted its budget on this question |
| `crag.out_of_scope` | The short-circuit fired |
| `out_of_scope` | Same check, at the top level, recorded for **both** arms |

`crag` is `null` on a classic record, not a block of zeroes. "The loop was not
there" and "the loop ran and did nothing" are different facts, and a comparison
that confused them would report a 0% reformulation rate for a pipeline that has no
reformulation node.

### How the counters are captured

`ask_question` returns a plain `str`; the final graph state, where
`iteration_count` lives, never leaves the processor. Rather than widen that return
type — a production contract the API and email paths depend on — for a
measurement-only concern, `CragInstrumentation` wraps the bound node methods on the
processor instance and counts calls from outside. The wrappers delegate and return
the state update untouched, so the measured pipeline is the production pipeline.

They are installed before the first question, because `build_graph` captures the
node callables when `ask_question` lazily compiles the graph.

## Reading the report

| Section | Answers |
|---------|---------|
| Configuration | Are the two runs comparable at all? |
| Coverage | Which questions are in both, and how many are usable |
| Latency | Overall cost of CRAG, over the paired successful subset |
| Latency by iterations | Typical case (bucket `0`) vs worst case (bucket `2`) |
| Out-of-scope short-circuit | How often CRAG refused, and what classic answered instead |
| Pipeline errors | Failures per arm |
| Answer quality | Per-criterion means — blocked until instructors score |
| Hallucination rate | Never computed; instructions for producing it by hand |

Bucket `0` is the interesting one: its delta is the price of the relevance grading
call on a question the loop never corrects, i.e. what CRAG costs the *typical*
student. The overall mean blends that with the worst case and understates how often
the pipeline is cheap.

Latency is compared over questions present **and successful in both** files. Using
the full files would let a question only one arm answered move the means, and the
report would attribute to CRAG a difference that is really a difference in which
questions were measured.

## What the tooling refuses to do

**It never scores answers, and it never estimates a hallucination rate.** Both are
human judgements made against `expected_answer` and the cited `source_doc`. A
proxy — string overlap, an LLM-as-judge, "short answers are probably refusals" —
would produce a number that reads like evidence while measuring nothing, and that
is the number that would end up in the report.

The quality section prints what is still missing until both files are scored, then
computes per-criterion means and deltas from the instructor-assigned scores.
Hallucination rate is never computed in either branch; the report explains how to
produce it by hand instead.

## Safety rails

| Situation | What happens |
|-----------|--------------|
| `--crag` and `--no-crag` both passed | argparse rejects it, they are mutually exclusive |
| Results file already exists | Refuses to start; pass `--resume` or `--output` |
| `--resume` onto a file from the other pipeline | Refuses before measuring anything |
| Classic and CRAG files swapped in `compare_runs.py` | Refuses; records carry their own `pipeline` label |
| Same file passed as both arms | Same refusal, for the same reason |
| Two `crag_*.jsonl` in `results/` | Auto-discovery refuses; pass the path explicitly |
| Out-of-scope answer in a classic file | Report prints a banner — the classic graph cannot produce one |

## Results schema versioning

Records and summaries carry `schema_version`, defined in `eval_io.py`:

| Version | Shape |
|---------|-------|
| 1 | Issue #13. Classic records only: no `schema_version`, `pipeline`, `out_of_scope` or `crag`. |
| 2 | Issue #16. Adds all four, so the two arms are comparable record by record. |

**No v1 file was ever committed** — no run has happened — so v2 broke nothing in
practice. Readers treat a missing `schema_version` as 1 and a missing `crag` block
as "not measured" rather than as zero iterations; `compare_runs.py` degrades to
skipping the per-iteration breakdown rather than reporting it as all-zeroes.

## Files

| Path | Role | Needs the pipeline's dependencies |
|------|------|-----------------------------------|
| `run_baseline.py` | Measures one arm | Yes — Ollama, FAISS, LangGraph |
| `compare_runs.py` | Diffs two results files | No — standard library only |
| `eval_io.py` | Results-file format shared by both | No |

`eval_io.py` imports nothing from `rag/` on purpose. Producing a results file needs
a 16 GB box; reading two committed ones and diffing them must not, so an instructor
can run the comparison on a laptop that could never host the pipeline.

## Next step

Issue #17 adds the Self-RAG grounding-verification node and #18 extends this
tooling to a three-way comparison with a confidence-level field. The seams are
already in place: `pipeline` is a label rather than a boolean, and
`build_output_path` takes the pipeline as an argument.
