# How this system changed, and what we actually know about it

Between 2026-09-09 and 2026-09-18 this pipeline went from answering course questions
out of the model's parametric memory — with a `alta` confidence stamp on the
result — to refusing what it cannot ground and correcting its own retrieval when the
first attempt fails. Every claim below is traced to a measurement, and the measurements
that were never taken are named as such.

**Read the two sections at the end before quoting any number from this document.**
The honest limits are not a disclaimer appended out of modesty; they decide which of
these numbers can carry an argument.

---

## The bottom line

| | Then (2026-09-09) | Now (2026-09-18) |
|---|---|---|
| Answers the 6 recorded course questions | 5 of 6, with 3 self-flagged as needing correction | **6 of 6** |
| Fabricated content | Yes — invented a FET variety that does not exist, labelled `alta` | None observed in the last 12 answers |
| Raw PDF extraction debris in a student's email | Yes | None observed |
| Prompt fits the model's context window | No — 20137 tokens against a 6000-token window | Yes, 4939 worst case |
| Retrieval reaches the topically correct document | 21% of chunks | **92%** |
| Reproducible | Unknown, assumed not | **Byte-identical** at a fixed commit |
| Wrong answers delivered with high confidence | Yes | **Yes — still** (see [Confidence is not quality](#confidence-is-not-quality)) |

The last row is the one that matters most and it is the one that has not moved.

---

## Three things this document is not

1. **It is not a benchmark.** There are 6 questions, chosen ad hoc, covering 2 topics
   out of a 15-document corpus. `resources/eval/dataset.jsonl` — the instructor-written
   question set the harness is built around — does not exist yet.
2. **It is not a comparison against the original system.** "v1.0" (Mistral 7B, `k=2`,
   `RetrievalQA`) was replaced in place rather than kept configurable. The last commit
   where it is intact is `87ca5f3`, and it has never been run. See
   [`AB_TESTING.md`](AB_TESTING.md) for the two ways out.
3. **It contains no quality score.** Scoring an answer against the course material is a
   human judgement. Every answer discussed below was read by a person; none was scored
   on the four criteria the harness reserves for instructors.

---

## The timeline is a sequence of systems, not a sample of one

Five runs of the same 6 questions were recorded. It was tempting to read the flips
between them as noise. They are not noise: **every transition spans at least one
functional commit**, and one of them changed the corpus itself.

| Run | Date | What was different |
|-----|------|--------------------|
| A | 09-09 17:26 | Derived chunk size live; old chunking (whole corpus concatenated, blind token cuts) |
| B | 09-10 01:43 | New loader: per document, structural boundaries, source tags (`8e70c18`) |
| C | 09-10 19:19 | `num_predict` enforced (`568a0e9`); **a course PDF deleted** (`dbed292`); diagnostics added |
| D | 09-10 23:28 | Generator's "No sé" no longer verified (`cc8b1c2`); **bge-m3 embeddings** (`6729bec`) |
| E | 09-17 21:34 | Generator's "No sé" opens the CRAG loop (`4eac958`); extraction noise filtered (`4204313`) |

Confidence level per question, or the refusal that replaced it:

| # | Question | A | B | C | D | **E** |
|---|----------|---|---|---|---|---|
| 1 | Two fundamental FET varieties | alta | alta | media | alta | **alta** |
| 2 | Why FETs have high input impedance | baja | refused | refused | refused | **media** |
| 3 | The three FET terminals | baja | refused | alta | alta | **alta** |
| 4 | Common-mode rejection ratio | refused | alta | alta | alta | **alta** |
| 5 | An ideal differential amplifier | media | refused | refused | alta | **alta** |
| 6 | What a differential amplifier is | media | alta | alta | alta | **alta** |

Run E is the first with no refusals. It is also the first whose reproducibility was
tested, which is what makes the column meaningful rather than a lucky draw.

---

## What was fixed, and what the fix was worth

### The prompt did not fit the context window

The root cause of the fabrications, and the one that made everything downstream
misleading.

`OLLAMA_CONTEXT_LENGTH = 5000` read like a model context window. It was a chunk size,
and `TokenTextSplitter` counts **tokens**, not characters. One chunk nearly filled
`num_ctx` by itself, so a `k=4` prompt measured **20137 tokens against a 6000-token
window**. Ollama truncates rather than failing, and llama.cpp keeps the *tail*, so the
first thing discarded was the instruction header — the sentence ordering the model to
answer only from the provided document, and the `"No sé la respuesta..."` escape hatch.

**The generator was answering from parametric memory with no way to know it, and
shipping the result with a confidence note.**

The fix derives the value so the three constants cannot drift apart again:

```python
CHUNK_SIZE_TOKENS = (OLLAMA_NUM_CTX - OLLAMA_ANSWER_TOKEN_BUDGET - PROMPT_OVERHEAD_TOKENS) // RETRIEVAL_K
```

| | Before | After |
|---|--------|-------|
| Chunks in the corpus | 32 | 150 |
| Worst-case prompt | 20137 tokens, force-truncated | 4939 tokens, `truncated = 0` |
| Answer to "the two fundamental FET varieties" | *"canal de nitrurio (NFET)"* — a fabrication, labelled `alta` | JFET and IGFET/MOSFET, N- and P-channel — correct, labelled `alta` |

The `alta` on that answer was unearned before and is earned now. That is the single
clearest improvement in this document.

### Retrieval was biased toward one document

Measured, not assumed. The query *"¿Por qué los FET presentan una elevada impedancia de
entrada?"* against the correct gate-insulation passage and a power-amplifier distractor:

| Embedding model | dim | correct | distractor | margin |
|---|---|---|---|---|
| GPT4AllEmbeddings | 384 | 0.4030 | 0.3684 | +0.0346 |
| bge-m3 | 1024 | 0.5669 | 0.4421 | **+0.1249** |

Both rank the right passage first in isolation. The **margin** is what decides survival
into the top 4 of 163 real chunks, and 3.6x is the whole difference.

| | Before | After |
|---|--------|-------|
| Chunks from the topically correct document | 5/24 (21%) | **22/24 (92%)** |
| Slots taken by `AMPLIFICADORES DE POTENCIA - GALIANO` (11% of the corpus) | 9/24 (37.5%) | **0/24** |

Two side effects worth recording. A documented gotcha disappeared with it:
`GPT4AllEmbeddings` fetched a model index over the public internet at startup and an
intermittent CDN 503 had taken the service down three times, so a bounded-retry wrapper
existed solely to survive that. And a bug we would have introduced ourselves was caught
in the process — the stored index hash covered only chunk text, so a 384-dimension index
would have been reloaded against a 1024-dimension model.

**Caveat stated at the time and still true:** this measures that retrieval reaches the
right *document*, not the right *page*. Question 2 in Run D proved the distinction —
right document, wrong pages, and a refusal.

### Generation had no upper bound

`OLLAMA_ANSWER_TOKEN_BUDGET = 900` existed only as a divisor in the chunk-size
arithmetic. It was never sent to the model, so the completion half of `num_ctx` was
unbounded while the prompt half was carefully derived.

On 2026-09-10 one regeneration entered a repetition loop and ran to **18673 tokens over
2 h 39 min at 2.05 tok/s**. Past 6000 tokens llama.cpp does not stop — it context-shifts
(`n_keep = 5`, `n_discard = 3069` per shift), so the chunks and the instruction header
were evicted and the model generated from an empty context with no way to reach EOS. The
queue is serial, so one runaway stalls every pending question.

`num_predict` is now enforced. `repeat_penalty = 1.1` was added as secondary
prophylaxis, but it is worth being precise: the runaway ran with `repeat_penalty = 1.000`
and at temperature 0 nothing else breaks a loop. The hard stop is what fixed it.

This is the same unit-and-enforcement confusion as the context-window bug, one layer up:
**a budget that appears only inside an arithmetic expression constrains nothing.**

### PDF extraction debris reached students

Run D's answer to question 4 contained this, pasted into an email:

```
Fm / FΠcmcm / iC oC iD oDS / rβrRgRg / vvvv / AA≈ / +≈ == / 22FRs
```

The filter that removed it drops runs of 6 or more consecutive non-prose lines. The
threshold is measured, and the measurement is what made the design non-obvious:

| Run length | 1 | 2 | 3-4 | 5-6 | 7-9 | 10-19 | 20+ |
|---|---|---|---|---|---|---|---|
| Mean chars/line | **20.9** | 12.1 | 8.5 | 8.1 | 6.9 | 6.6 | 5.6 |

**A per-line filter would have deleted 39% of the corpus**, including
`Tn = 290 K ( F − l). (233)` — surviving formulas that *are* the course content. Those
live in runs of one, which is why runs of one average 20.9 characters and runs of twenty
average 5.6. The run is the signal, not the line. Six specifically because a readable
run of five exists in `disipa.pdf`, and the reported debris block is a run of nine, so
any threshold above nine would not have fixed the bug that prompted the work.

| | Before | After |
|---|--------|-------|
| Corpus characters | 464286 | 445880 (−4.0%) |
| Chunks | 163 | **148** |
| Worst-affected file | — | `Amplificador diferencial.pdf`, 8.9% removed |
| The 7 `Amplificación - *.pdf` files | — | byte-for-byte unchanged (0% noise) |

**This does not recover a formula.** Extraction already destroyed them; the filter only
stops the wreckage being quoted to a student as if it were an answer.

And Self-RAG could never have caught it: the debris *is* traceable to the retrieved
chunks, so `fundamentada` was the correct verdict and `alta` the correct label.

### The correction loop had never run

This is the mechanism the whole agentic design rests on, and for 24 recorded
question-runs it was unreachable code.

`grade_relevance` returned `irrelevante` **zero times**. It still has, across what is now
30 question-runs and 32 grading calls. The loop was opened for the first time in Run E,
and not by the grader — by the generator. When the generator uses the `INITIAL_PROMPT`
escape hatch it is reporting that retrieval failed, and it is reporting it having read
the chunks *in full* where the grader sees only a 3000-character preview of each.

Question 2, Run E:

```
21:54:55 retrieve  → Amplificador diferencial, FET, FET, Otras etapas   (right doc, wrong pages)
22:01:00 grade     → relevante (iteration 0)
22:11:34 generate  → "No sé" → REFORMULATION 1/2
22:12:49 retrieve  → FET, FET, Amplificador diferencial, FET
22:22:55 grade     → relevante (iteration 1)
22:31:30 generate  → "No sé" → REFORMULATION 2/2
22:32:01 retrieve  → FET, FET, FET, Amplificador diferencial
22:37:48 grade     → relevante (iteration 2)
22:52:31 grounding → fundamentada (attempt 1/2)   → ANSWERED
```

> Los FET presentan una elevada impedancia de entrada debido a que el terminal de puerta
> **no maneja virtualmente corriente, salvo alguna corriente de fuga**. Esto resulta
> esencial en variadas aplicaciones como ser: llaves analógicas, amplificadores de muy
> alta impedancia de entrada, etc.

That is verbatim the passage Run D's question 3 had quoted twenty minutes after
question 2 refused to answer. **Run D's refusal is proven false, not suspected false** —
the material was in the corpus and the system had already shown it could find it.

The cost is real: **57m39s against Run D's 13m39s refusal**, a 4.2x on that question.
Note also that the second rewrite is worse Spanish than the first
(*"transistores de campo-transistor"*) and still retrieved 3 of 4 on-topic chunks.
Reformulation quality is not what made this work.

The grader saying `relevante` at iterations 0, 1 **and** 2 — on chunks the generator had
just declared insufficient — is a defect in its own right, and it is open.

---

## The pipeline is reproducible

Believed non-deterministic for a week, on the strength of the flips in the table above.
Tested on 2026-09-18, and the belief was wrong.

Every recorded answer to question 4, full-body SHA-256:

| Sample | Run | Code | sha256 | chars |
|---|---|---|---|---|
| S1 | A | pre-`2e1c68b` | `55b5370b99d5e8ea` | 324 |
| S2 | B | `8e70c18` | `7c10bd0e0c3826f6` | 459 |
| S3 | C | `9eb0a9c` | `b195b073896c91ba` | 528 |
| S4 | D | `6729bec` | `8fb61a07e8baab5d` | 500 |
| **S5** | E | **`4204313`** | **`06029dfb5073b433`** | 673 |
| **S6-S8** | repeat test | **`4204313`** | **`06029dfb5073b433`** | 673 |

**4 of 4 differ across code changes. 4 of 4 are byte-identical within the same code.**
S5 carries the most weight: a separate session 4.5 hours before the others, at a
different queue position, preceded by different questions. The graph path was identical
too. A bare Ollama call at temperature 0 is likewise identical 3 of 3.

Two of the three suspected causes were unsound on inspection rather than merely
unconfirmed. **A fixed seed could not have helped** — at temperature 0 llama.cpp decodes
greedily, so there is no RNG for a seed to control. And **`repeat_penalty` does not cause
run-to-run variance**; it is deterministic given the same context, so changing it from
1.0 to 1.1 is a code-change effect, not randomness.

### Why this changes how the table above should be read

The `n = 1` caveat does not disappear, it **changes shape**. It was:
*"this number might have come out differently, so it proves nothing."* It is now:
*"this number is exactly what this system produces for this question, and says nothing
about the other questions in the course."*

Repeating a question adds no information. **Adding questions does.** The sampling
limitation was never repeats — it was, and remains, the absent dataset.

Results files now record the commit, a dirty-tree flag and a corpus hash, and
`compare_runs.py` refuses to diff two runs that did not come from the same system. The
corpus hash is the field a commit SHA cannot replace: the course documents are
untracked, so the corpus can change with nothing in the diff to show for it — which is
exactly what `dbed292` did between runs B and C.

---

## What this cost

Answered question, current pipeline, measured in Run E:

| Node | Measured |
|------|----------|
| FAISS retrieval (k=4) | 1–13 s |
| Phi relevance grade | 342–606 s (mean ~403 s over 8 calls) |
| Query reformulation | 28 s, 67 s |
| Generation + grounding verification | 817–1091 s (one opaque block) |
| **Whole run, 6 questions** | **168 min** |

- **The control plane is not the cheap half.** Each Phi call costs roughly what the Llama
  generation costs, so the CRAG/Self-RAG arm roughly doubles the classic pipeline rather
  than adding a margin.
- **Retrieval stopped being free.** It was 0.019 s with in-process embeddings; bge-m3 over
  Ollama makes it an HTTP call with model inference. Still negligible against a 400 s
  grade, but the old "retrieval is free" line is no longer literally true.
- **Generation and grounding cannot be separated** — no log line marks the generator
  returning. Those ~950 s are one block, which is why the question "does the control
  model really cost more than the generator" cannot currently be answered.

### Latency varies by 49%, and that is deliberately not a problem

The four byte-identical samples of question 4 took **21m46s, 32m26s, 29m19s and
24m36s**. Identical input, identical output, a 49% spread in wall-clock.

Response time is not a goal for this system. Answers are delivered asynchronously by
email precisely because it runs on a CPU-only 16 GB box, and the async design is what
makes those resources sufficient. With fewer than 15 students asking occasionally,
throughput is not a constraint either: even if all 15 wrote on the same evening and
every one of them triggered the correction loop, the queue clears overnight.

So the variance is recorded as a property of the system and not treated as a defect.

---

## Confidence is not quality

The confidence note is a faithful reading of the pipeline's internal state. It is not a
statement about whether the answer is right, and **two of Run E's six answers carry a
note that actively misleads.**

Question 4, labelled `alta`, verified `fundamentada` on the first attempt, from the
correct document:

> El factor de rechazo de modo común (FRc) se define como la relación entre **la ganancia
> a modo común y la ganancia a modo diferencial** compuesto. […] Además, el factor de
> rechazo depende del transistor (g m) y de la resistencia de la fuente de corriente
> (r F), **a mayor resistencia menor ganancia a modo común y, en consecuencia, mayor
> factor de rechazo.**

CMRR is Ad/Ac. The opening sentence states the ratio **inverted**. And the answer refutes
itself two paragraphs later: a lower common-mode gain can only *raise* the rejection
factor if the ratio is Ad/Ac. No course material is needed to establish that one of the
two sentences is wrong — the answer contains its own counterexample.

Question 6, also `alta`, defines the differential input as *"la diferencia entre la señal
aplicada a la entrada inversora y la señal aplicada a la entrada no inversora"* — v(−) −
v(+), the inverse of the usual convention. That one needs the course's own convention
confirmed before it can be called an error.

Every mechanism behaved **correctly** on question 4:

| Signal | Value | Correct? |
|---|---|---|
| Relevance | `relevante`, iteration 0 | Yes — the right document, 4 of 4 chunks |
| Grounding | `fundamentada`, attempt 1/2 | Yes — the inverted statement **is** traceable to the chunks |
| Confidence | `alta` | Yes, by its definition — nothing had to be corrected |

**Grounding verification checks provenance, and provenance is not truth.** An inverted
ratio traces back perfectly: the chunks discuss both gains and their relationship, and
the generator assembled them in the wrong order. This is the same structural limit the
extraction debris hit from the other direction.

The confidence note presented a provenance check to a student as a reliability signal.
[Issue #31](https://github.com/fedepacher/rag/issues/31) addressed the part that can be
addressed without a person reading the answer:

- **Every level now advises checking against the course material**, `alta` included. It
  had been the only level that advised nothing — and it is the label both wrong answers
  carried. The checks report on retrieval and on provenance, so reserving the advice for
  levels where a check already complained withheld it from exactly the failure no check
  can see.
- **`alta` now states the distinction to the student**: *"Eso verifica su procedencia, no
  su exactitud: una afirmación puede provenir del material y aun así estar mal expresada
  o invertida."*
- **No coherence-checking node was added.** The judge would be the same 3.8B control
  model that has returned `irrelevante` 0 times in 30 question-runs; it would not have
  caught Run E's answer anyway, since the conflict there surfaces only by inference
  rather than as an explicit contradiction; and it would have to degrade toward letting
  answers through, like every other parser here, so an unsure judge changes nothing.

**None of that makes the answer right.** The wrong answer still ships, now with a note
that no longer implies it was checked for correctness. Whether the answer is correct
remains a human judgement, and that is what the instructor criteria exist for.

**Consequence for this document and any successor: no confidence-level distribution may
be presented as a quality metric.** Counting `alta` answers measures the pipeline's
agreement with itself. Two runs now show that agreement at its maximum over a definition
stated backwards.

---

## What is not measured

Read this as the specification for the next phase of work, not as hedging.

| Gap | Why it is not closed | Who unblocks it |
|-----|---------------------|-----------------|
| **No question dataset** | `dataset.jsonl` is intentionally absent. Synthetic course questions would produce a number that reads like evidence while measuring nothing | Course instructors |
| **No quality scores** | Scoring against the course material is a human judgement; the harness never auto-scores, by design | Course instructors |
| **No v1.0 comparison** | Mistral 7B / `k=2` / `RetrievalQA` was replaced in place, not kept configurable. Intact at `87ca5f3`, never run | A decision, then a 16 GB box |
| **6 questions, 2 topics** | Ad hoc set, submitted by email during debugging. Says nothing about the other 13 documents | Follows from the dataset |
| **Correction-loop frequency** | Fired once, ever. One occurrence is not a rate | More questions |
| **Generation vs verification cost** | No log line marks the generator returning ([#38](https://github.com/fedepacher/rag/issues/38)) | One `logging.info` |
| **Grader preview size** | Raised 1500 → 3000 chars (9% → 80% of a mean chunk) for a quality reason, never tested for its effect on grading quality ([#34](https://github.com/fedepacher/rag/issues/34)) | A measurement |
| **Why the grader never says `irrelevante`** | 0 in 30 question-runs. Unexplained | Investigation |
| **Answer correctness beyond 12 read answers** | Two people-hours of reading, not a method | Follows from the dataset |

The corpus is also not reproducible from this repository: 14 of the 15 course documents
are untracked in a public repo, by decision ([#35](https://github.com/fedepacher/rag/issues/35)).
The corpus hash in every results file is what makes a run identifiable despite that.

---

## Next step

The dataset is the bottleneck for everything in the table above, and it is the one item
this project cannot unblock by itself. The work that does not wait on it is
[#31](https://github.com/fedepacher/rag/issues/31): the confidence note currently tells a
student that an inverted definition is reliable, and that is a defect the system can be
measured against with the 6 questions already in hand.

---

*Evidence for every figure in this document lives in the referenced commits, the
`test_nosql.prompts` collection, and the issue threads linked above. Where a number came
from a single run on one machine, the document says so.*
