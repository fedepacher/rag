# Status — what is implemented vs. what is measured

Written for the URUCON 2026 write-up. Its whole job is to make one distinction
impossible to blur: **the CRAG + Self-RAG architecture exists as code, and none of
it has ever been measured.**

Every latency, quality and hallucination figure the paper needs is still missing,
and no file in this repository contains one. `resources/eval/results/` does not
exist. `dataset.jsonl` does not exist. No node of the agentic graph has ever run
against a live Ollama server.

That is not a defect — the evaluation was always gated on hardware and on the
course instructors, neither of which this repository can provide. It only becomes
a problem if the write-up describes the architecture in the past tense of a
completed experiment.

## Legend

| Marker | Meaning |
|--------|---------|
| **Code** | Implemented and reviewed. Never executed against a live model or the real course documents |
| **Box** | Blocked on a run: CPU-only 16 GB machine with Ollama, both models pulled and the course PDFs under `resources/files/` |
| **Instructors** | Blocked on the course teaching staff. No amount of code closes it |
| **Decision** | Blocked on a choice nobody has made yet |

## 1. Implemented — believed correct, never run live

| Component | Where | Status |
|-----------|-------|--------|
| Llama 3.1 8B generation model | `rag/main.py` (`OLLAMA_MODEL`) | **Code** |
| Phi-3.5-mini control model, one shared instance | `rag/main.py` (`build_phi_control_llm`) | **Code** |
| FAISS retrieval at `k=4` | `rag/llm_processor.py` (`RETRIEVAL_K`) | **Code** |
| LangGraph state graph, both shapes | `rag/llm_processor.py` (`build_graph`) | **Code** |
| CRAG relevance grading, bounded reformulation, out-of-scope short-circuit | `rag/llm_processor.py` | **Code** |
| Self-RAG grounding verification, bounded retry, ungrounded fallback | `rag/llm_processor.py` | **Code** |
| Confidence-level note on the answer | `rag/llm_processor.py` (`derive_confidence`) | **Code** |
| Baseline / CRAG measurement harness | `run_baseline.py` | **Code** |
| A/B comparison report generator | `compare_runs.py`, `eval_io.py` | **Code** |
| Dataset format spec | `README.md` | **Code** |

What "believed correct" rests on: code review, and the fact that the decision
logic is deliberately made of pure functions — `parse_relevance`,
`parse_grounding`, `sanitize_reformulated_query`, `derive_confidence`,
`route_after_grading`, `route_after_grounding` — that can be reasoned about and
exercised without the model stack. `compare_runs.py` and `eval_io.py` import
nothing from `rag/` on purpose and need only the standard library.

What it does **not** rest on: any observation of the real system. Nobody has seen
Phi-3.5-mini answer a relevance prompt on this hardware. The failure modes the
defensive parsing exists to absorb are anticipated, not observed.

## 2. Blocked on a live box

None of this can happen on a laptop. It needs the `rag` container on the target
machine, otherwise idle, with both models resident.

| Item | Status | Notes |
|------|--------|-------|
| Classic-arm latency baseline | **Box** + **Instructors** | Needs `dataset.jsonl` first; the runner refuses to start without it |
| CRAG-arm latency | **Box** + **Instructors** | Same gate |
| Reformulation-rate and out-of-scope-rate figures | **Box** + **Instructors** | Recorded automatically once a run happens |
| Worst-case latency (the 2-reformulation bucket) | **Box** + **Instructors** | Reported per bucket by `compare_runs.py` |
| Whether the ~8.5–9 GiB two-model footprint actually fits under load | **Box** | Estimated from `num_ctx=6000`, never observed |
| Whether Phi-3.5-mini's verdicts are usable at all | **Box** | The single largest unknown in the design. Every degradation path assumes a *confused* grader; none of them help against a *consistently wrong* one |
| Cold-start cost of pulling ~7.3 GB of weights | **Box** | Estimated from model sizes |

## 3. Blocked on the course instructors

| Item | Status | Notes |
|------|--------|-------|
| `dataset.jsonl` — 40–60 validated course questions | **Instructors** | Format frozen in `README.md`. Generating these synthetically would make the whole evaluation meaningless, so the repo refuses to fake them |
| Quality scores: `pertinencia`, `claridad`, `precision`, `lenguaje` | **Instructors** | Assigned by hand per record after a run. The harness never auto-scores |
| Hallucination rate | **Instructors** | Never computed by any tool here. It is a human judgement against `expected_answer` and `source_doc` |
| Whether the confidence note reads usefully to a student | **Instructors** | Never shown to a student |
| Whether the out-of-scope and ungrounded refusals are acceptable pedagogically | **Instructors** | The system now sometimes declines to answer where v1.0 would have replied. Nobody has agreed that this is the right trade |

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
- ❌ "Latency increased by *N* seconds." No latency has been measured on either arm.
- ❌ "Compared to the original v1.0 system…" There is no runnable v1.0 on this branch.
- ❌ "The system detects out-of-scope questions correctly *N*% of the time." The short-circuit is implemented; its accuracy is unknown.
- ❌ "Evaluated on a teacher-validated dataset." The dataset does not exist yet.
- ❌ Any figure attributed to Phi-3.5-mini's grading accuracy.

What the write-up **can** say today, accurately:

- ✅ The architecture is implemented: LangGraph state graph, CRAG relevance grading
  with a bounded 2-reformulation correction loop and an out-of-scope
  short-circuit, Self-RAG grounding verification with one bounded regeneration and
  an ungrounded fallback, and a confidence level derived from those signals.
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

Framing that stays honest: **an implemented architecture and a designed evaluation,
with the measurement campaign still pending.** That is a legitimate contribution.
Presenting it as a completed experiment is not.

## 6. Order of operations

Each step is blocked by the one above it.

1. **Instructors deliver `dataset.jsonl`.** Everything else waits on this.
2. **Decide the reference point** (`AB_TESTING.md`, options A and B). Cheap now,
   expensive after the runs — option A adds a third run on the same box under the
   same conditions.
3. *(Optional, before the runs)* **Add the Self-RAG counters.** Same wrapping
   technique as `CragInstrumentation`, two more nodes, a `self_rag` block and a
   `SCHEMA_VERSION` bump. Doing it afterwards means re-running.
4. **Run the classic arm first**, on an otherwise idle box, then the CRAG arm.
   Ordering matters: the CRAG arm leaves Phi in the Ollama cache.
5. **Instructors score both files.**
6. **Run `compare_runs.py`** and commit all four result files plus the report.
7. **Only then** write the results section.
