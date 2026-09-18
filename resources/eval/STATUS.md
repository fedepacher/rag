# Status — what is implemented vs. what is measured

Written for the URUCON 2026 write-up. Its whole job is to make one distinction
impossible to blur: **what has been observed, and what has only been written.**

> **Updated 2026-09-18.** This document previously said the architecture had never
> been measured and that no node of the graph had ever run against a live Ollama
> server. **That is no longer true**, and the paragraphs below have been corrected.
> Five runs of 6 course questions have been recorded, every node has executed, and
> [`EVOLUTION.md`](EVOLUTION.md) traces what each of them showed.

What exists now, and did not before:

- **Five recorded runs** of the same 6 course questions (2026-09-09 → 2026-09-18),
  every answer read by a person.
- **The pipeline is reproducible**: the same question at the same commit produces a
  byte-identical answer (4 of 4 samples, full-body SHA-256).
- **Measured latencies** for every node, and measured retrieval quality (chunks from
  the topically correct document: 21% → 92%).
- **Four root causes found and fixed** with before/after evidence: a prompt 3.4x over
  the context window, retrieval bias toward one document, unbounded generation, and
  PDF extraction debris reaching students.

What still does not exist, and is what the rest of this document is about:

- **`dataset.jsonl`** — not as a file, and **not as 40–60 questions either.** The
  instructor cannot produce that set, so the evaluation is frozen at the 6 questions
  already in use (§3). The questions exist verbatim; each still needs an approved
  `expected_answer` before the harness will run.
- **`resources/eval/results/`** — no run has gone through `run_baseline.py`; all five
  went through the production email/Mongo path.
- **Any quality score.** Twelve answers have been read; none has been scored on the
  four criteria. One of them is wrong and labelled `alta`.
- **A comparison against v1.0** (Mistral 7B, `k=2`, `RetrievalQA`).

So the distinction has moved rather than disappeared. It is no longer
"architecture vs. nothing"; it is **"exact on six questions vs. unknown on the
course."** The write-up may describe the mechanisms in the past tense now, and may
state per-question results as exact, because the pipeline is reproducible at a fixed
commit. It may not present a confidence-level count as a quality result, and it may
**never** generalise from 6 questions to the course — that limit is now permanent
rather than pending.

The remaining gates: hardware, and one approved reference answer per question. The
second is much smaller than the dataset it replaces.

## Legend

| Marker | Meaning |
|--------|---------|
| **Observed** | Has executed against the live models and the real course documents, on the target box. Observed on 6 ad hoc questions — not measured against a validated dataset |
| **Code** | Implemented and reviewed. Never executed against a live model or the real course documents |
| **Box** | Blocked on a run: CPU-only 16 GB machine with Ollama, all three models pulled and the course PDFs under `resources/files/` |
| **Instructors** | Blocked on the course teaching staff. No amount of code closes it |
| **Decision** | Blocked on a choice nobody has made yet |

## 1. Implemented — what has run, and what has not

| Component | Where | Status |
|-----------|-------|--------|
| Llama 3.1 8B generation model | `rag/main.py` (`OLLAMA_MODEL`) | **Observed** |
| Phi-3.5-mini control model, one shared instance | `rag/main.py` (`build_phi_control_llm`) | **Observed** — 32 grading calls |
| bge-m3 embeddings over Ollama | `rag/main.py` (`EMBEDDING_MODEL`) | **Observed** |
| FAISS retrieval at `k=4` | `rag/llm_processor.py` (`RETRIEVAL_K`) | **Observed** |
| LangGraph state graph, CRAG shape | `rag/llm_processor.py` (`build_graph`) | **Observed** |
| LangGraph state graph, classic shape (`enable_crag=False`) | `rag/llm_processor.py` (`build_graph`) | **Code** — the baseline arm has never been run |
| CRAG relevance grading | `rag/llm_processor.py` | **Observed** — and it has returned `irrelevante` **0 times in 30 question-runs** |
| Bounded reformulation | `rag/llm_processor.py` | **Observed once**, in Run E. One occurrence is not a rate |
| Out-of-scope short-circuit | `rag/llm_processor.py` | **Observed** — including one refusal later proven false |
| Self-RAG grounding verification | `rag/llm_processor.py` | **Observed** |
| Bounded regeneration with the strict prompt | `rag/llm_processor.py` | **Observed** in runs A–B; not triggered since |
| Ungrounded fallback | `rag/llm_processor.py` | **Observed** |
| Confidence-level note on the answer | `rag/llm_processor.py` (`derive_confidence`) | **Observed** — and shown to overstate quality; see [#31](https://github.com/fedepacher/rag/issues/31) |
| Extraction-noise filter | `rag/document_loader.py` | **Observed** — 163 → 148 chunks live |
| Results-file provenance and the comparison guard | `eval_io.py`, `compare_runs.py` | **Code** |
| Baseline / CRAG measurement harness | `run_baseline.py` | **Code** — all five runs went through the production email/Mongo path instead |
| A/B comparison report generator | `compare_runs.py`, `eval_io.py` | **Code** — it has never had two results files to read |
| Dataset format spec | `README.md` | **Code** |

The graph's decision logic is deliberately made of pure functions —
`parse_relevance`, `parse_grounding`, `sanitize_reformulated_query`,
`derive_confidence`, `route_after_grading`, `route_after_grounding` — which is what
lets 85 unit tests exercise the routing, the loop bounds and the confidence
derivation in under two seconds without the model stack. `compare_runs.py` and
`eval_io.py` import nothing from `rag/` on purpose and need only the standard
library.

**What "Observed" does not mean.** It means the node ran and its behaviour was read
off the logs of 6 questions on one machine. It does not mean the behaviour is
characterised: the reformulation path has fired exactly once, the classic arm has
never run at all, and the harness these files were written for has still never
produced a results file. Several failure modes the defensive parsing exists to
absorb remain anticipated rather than observed — no grader exception, no unparseable
verdict and no empty grader response has actually occurred.

## 2. Blocked on a live box

None of this can happen on a laptop. It needs the `rag` container on the target
machine, otherwise idle, with both models resident.

| Item | Status | Notes |
|------|--------|-------|
| Classic-arm latency baseline | **Box** + **Instructors** | Needs `dataset.jsonl` first; the runner refuses to start without it |
| CRAG-arm latency | **Box** + **Instructors** | Same gate |
| Reformulation-rate and out-of-scope-rate figures | **Box** + **Instructors** | Recorded automatically once a run happens |
| Worst-case latency (the 2-reformulation bucket) | **Box** + **Instructors** | Reported per bucket by `compare_runs.py` |
| Whether the multi-model footprint actually fits under load | **Observed** | It fits. Five full runs plus a three-repeat test completed on the 16 GB box with no swap death. Note it is now *three* models, not two — bge-m3 added ~1.2 GB of weights; Ollama unloads idle ones |
| Whether Phi-3.5-mini's verdicts are usable at all | **Observed — and the answer is no, for one of its two jobs** | This row read: *"the single largest unknown in the design. Every degradation path assumes a confused grader; none of them help against a consistently wrong one."* That is exactly what happened. The relevance grader has returned `irrelevante` **0 times in 30 question-runs**, including twice on chunks the generator had just declared insufficient, so `grade_relevance` has never once opened the correction loop. Grounding verification, its other job, behaves sensibly. The design survives only because the generator's escape hatch was given an edge to the loop ([#29](https://github.com/fedepacher/rag/issues/29)) |
| Cold-start cost of pulling weights | **Box** | Still estimated. Now ~8.5 GB across three models |
| Index rebuild cost | **Observed** | ~14.5 min for 148 chunks with bge-m3 on CPU, paid on every corpus change — and on every image rebuild, because the index has no volume ([#33](https://github.com/fedepacher/rag/issues/33)) |

## 3. The instructor review of 2026-09-18

Five of the six questions that had been waiting on the course staff were answered. What
they settled, and what it cost:

| Item | Outcome |
|------|---------|
| Is the common-mode rejection answer wrong? | **Yes.** The rejection factor is differential gain over common-mode gain. The pipeline stated it inverted |
| Is question 6's sign convention wrong? | **No** — v(−) − v(+) is this course's convention. An earlier draft of `EVOLUTION.md` over-flagged it; corrected |
| Are the refusals pedagogically acceptable? | **Yes.** *"Está bien: si no sabe, que no delibere, porque puede crear confusión."* Withholding is the intended trade |
| Does the confidence note help a student? | **Yes** — it tells them whether to take an answer at face value or with caution, and the system is understood to be under test and improving |
| `dataset.jsonl` — 40–60 validated questions | **Will not happen.** See below |
| Quality scores on the four criteria | **Still open**, and the review is why it matters — see below |

### The dataset is frozen at 6 questions, by decision rather than by blockage

The instructor cannot produce 40–60 validated question/answer pairs, so the evaluation
will use the 6 questions already in use. That is a **decision**, and it changes what this
repository can claim permanently, not temporarily:

- No result may be generalised to the course. Six questions across 2 topics out of 15
  documents is the sample, for good.
- Per-question results remain exact — the pipeline is reproducible at a fixed commit — so
  "this system answers these six questions this way" is a sound statement. "This system
  answers *N*% of course questions correctly" never will be.
- `run_baseline.py` refuses to start without `dataset.jsonl`, so the 6 questions have to
  be written into that file for any of the harness to run at all. The questions exist
  verbatim; `expected_answer` for each still needs instructor approval, and that approval
  is now the single remaining gate on the whole harness.

### Why the quality scores still matter, demonstrated by this very review

Asked whether the recorded answers were correct, the instructor's reading was *"lo que
leí de las respuestas parecían estar bien"* — and in the same review confirmed that the
common-mode rejection answer is wrong.

**Both statements are honest and they are about the same set of answers.** An inverted
definition reads as natural prose; it does not look like an error until someone checks
the relationship. That is not a lapse by the reviewer — it is the reason a confidence
level, a grounding verdict and a fluent reading all fail to substitute for scoring
against a reference answer. It is the strongest argument in this repository for keeping
the four criteria, and it was produced by accident.

## 4. Unmeasured, and blocked on a decision

| Item | Status | Notes |
|------|--------|-------|
| The reference point for "before" | **Decision** | `--no-crag` is the Llama 3.1 8B / `k=4` baseline, **not** v1.0. See `AB_TESTING.md` for the two ways out |
| Any comparison against the original Mistral 7B / `k=2` system | **Decision** + **Box** | v1.0 was replaced in place; the last commit where it is intact is `87ca5f3`. Measuring it means a worktree, a backported harness, and a third label in the tooling |
| Self-RAG counters in the results record | **Code, not written** | `CragInstrumentation` wraps the retrieval and CRAG nodes only. Grounding verdicts, regeneration count, ungrounded-fallback rate and the confidence level have no fields yet |

## 5. Claims the write-up cannot support today

Listed explicitly, because these are the sentences that write themselves.

- ❌ "The agentic pipeline improved answer precision by *N*%." No quality score exists.
- ❌ "CRAG reduced hallucinations." No hallucination rate has been computed, by anyone, ever.
  One specific fabrication was eliminated and the cause identified, which is an
  anecdote with a mechanism — not a rate.
- ❌ "Latency increased by *N* seconds **compared to the classic pipeline**." The CRAG
  arm's node latencies are now measured; the classic arm has still never run, so
  there is nothing to subtract from.
- ❌ "Compared to the original v1.0 system…" There is no runnable v1.0 on this branch.
- ❌ "The system detects out-of-scope questions correctly *N*% of the time." Worse than
  unknown: one recorded out-of-scope refusal was later **proven false** — the material
  was in the corpus and a later question quoted it.
- ❌ "Evaluated on a teacher-validated dataset." The dataset does not exist yet.
- ❌ Any figure attributed to Phi-3.5-mini's grading **accuracy**. What can be stated is
  the bare observation that `irrelevante` was returned 0 times in 30 question-runs.
- ❌ **Any confidence-level count presented as a quality result.** `alta` measures the
  pipeline's agreement with itself. Two runs show that agreement at its maximum over a
  definition stated backwards ([#31](https://github.com/fedepacher/rag/issues/31)). This
  prohibition is new and it is the easiest one to violate by accident, because the
  distribution looks like a result and is trivially countable.
- ❌ Anything generalised from 6 questions to the course. They were submitted ad hoc by
  email during debugging and cover 2 topics across 15 documents.

What the write-up **can** say today, accurately:

- ✅ The architecture is implemented: LangGraph state graph, CRAG relevance grading
  with a bounded 2-reformulation correction loop and an out-of-scope
  short-circuit, Self-RAG grounding verification with one bounded regeneration and
  an ungrounded fallback, and a confidence level derived from those signals.
- ✅ **Every node has executed on the target hardware**, and the per-node latencies of
  the CRAG arm are measured. The control plane is roughly half of total runtime: each
  Phi call costs about what the Llama generation costs.
- ✅ **The pipeline is reproducible.** The same question at the same commit yields a
  byte-identical answer (4 of 4 samples, full-body SHA-256, one from a separate
  session). Wall-clock varies by up to 49% on identical work.
- ✅ **Retrieval quality is measured**, on the 6 questions: chunks drawn from the
  topically correct document went from 21% to 92% after replacing the embedding model,
  with the embedding margin measured directly (+0.0346 → +0.1249 on a correct/distractor
  pair).
- ✅ **Four root causes, each with before/after evidence**: a prompt 3.4x over the
  context window that silently evicted the instruction header; retrieval bias toward one
  document; unbounded generation that ran 18673 tokens in 2 h 39 min; and PDF extraction
  debris reaching a student's email. These are engineering findings with mechanisms, and
  they are the most defensible material this project has.
- ✅ **A negative result worth publishing**: the CRAG relevance grader, the component the
  correction loop is nominally built on, never fired. The loop only ever ran because the
  generator's own "I don't know" was wired to it. A 3.8B control model reading a
  3000-character preview was outperformed by the 8B generator reading the chunks in full.
- ✅ **A structural limitation of Self-RAG, demonstrated twice**: grounding verification
  checks provenance, not truth. Extraction debris and an inverted definition both traced
  back to the retrieved chunks perfectly, so both were correctly labelled `fundamentada`
  and shipped with high confidence.
- ✅ The design decisions and their trade-offs, which are documented in the code and
  in `CLAUDE.md` — why the loops are bounded where they are, why both parsers
  degrade towards *keep answering* rather than towards *refuse*, why the control
  model is a separate 3.8B model rather than the generator.
- ✅ The evaluation methodology, as a methodology: what would be measured, on what
  dataset, with which criteria, and why no proxy metric is computed in place of
  instructor scoring.
- ✅ The engineering constraint that shaped all of it: CPU-only inference on a 16 GB
  box, which is why the control model is small, why the loops are bounded at 2, and
  why the retry uses a different prompt rather than resampling.

Framing that stays honest: **an implemented architecture, a working system observed on
6 ad hoc questions, four root causes found and fixed with evidence, and a measurement
campaign still pending.** That is a legitimate contribution, and a stronger one than it
was — the engineering findings and the negative result about the control model stand on
their own. Presenting it as a completed experiment is not.

[`EVOLUTION.md`](EVOLUTION.md) is the narrative version of that material, with every
figure traced to where it came from.

## 6. Order of operations

Each step is blocked by the one above it.

1. **Instructors deliver `dataset.jsonl`.** Everything else waits on this.
2. **Decide the reference point** (`AB_TESTING.md`, options A and B). Cheap now,
   expensive after the runs — option A adds a third run on the same box under the
   same conditions.
   *Cheaper than it was:* the pipeline is now known to be reproducible at a fixed
   commit, and results files record their commit and corpus hash, so a run measured
   today remains comparable to one measured later provided both are pinned. Runs
   recorded before that — every run so far — are not comparable to each other and the
   tooling now refuses to pretend otherwise.
3. *(Optional, before the runs)* **Add the Self-RAG counters.** Same wrapping
   technique as `CragInstrumentation`, two more nodes, a `self_rag` block and a
   `SCHEMA_VERSION` bump. Doing it afterwards means re-running.
4. **Run the classic arm first**, on an otherwise idle box, then the CRAG arm.
   Ordering matters: the CRAG arm leaves Phi in the Ollama cache.
5. **Instructors score both files.**
6. **Run `compare_runs.py`** and commit all four result files plus the report.
7. **Only then** write the results section.
