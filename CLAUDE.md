# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Async Q&A system for a university electronics course (FACeIA / UNR). Students email questions; the system retrieves relevant context from uploaded course documents (PDF/DOCX) using RAG, generates an answer via LLM, checks that answer against the retrieved material before sending it, and delivers it back by email. Designed to run on low-spec hardware (CPU-only, 16 GB) — processing is fully asynchronous.

The retrieval pipeline is agentic: a CRAG correction loop grades whether the retrieved chunks can answer the question and rewrites the query when they cannot, and a Self-RAG verification step checks that the generated answer is traceable to those chunks. Both refuse rather than speculate when they run out of budget. See **Architecture** below.

## Services

Four Docker containers, all on a private `rag_network` bridge:

| Service | Path | Entry point | Port |
|---------|------|-------------|------|
| `api` | `api/` | `api/main.py` | 8000 |
| `rag` | `rag/` | `rag/main.py` | — |
| `email` | `email/app/` | `email/app/email_main.py` | — |
| `database` / `mongodb` | — | stock images | — |

## Commands

```bash
# Start everything
docker compose up --build

# Start a single service (e.g. after code change)
docker compose up --build api

# Swagger UI (API)
# http://127.0.0.1:8000/docs
```

There is no test suite. `pytest` used to be listed in `requirements_api.txt`; it was removed when that file was pinned, since it shipped a test dependency into the production image for tests that do not exist. Add it back in the same commit that adds the first test.

## Pre-flight: passwords.json

Before `docker compose up`, create `passwords.json` in the repo root (never committed):

```json
{
  "forgot_password": "...",
  "smtp_password": "...",
  "imap_password": "..."
}
```

These are the only credentials NOT in env vars — by design, so they don't appear in `docker-compose.yml`.

## Architecture

### Data flow

```
Inbound email → IMAP (email svc) → MongoDB {output: null}
                                        ↓
                              API GET /receive-prompt  ← RAG polls
                                        ↓
    RAG: LangGraph CRAG + Self-RAG graph
         FAISS retrieve (k=4) → Phi-3.5-mini grade relevance ⟲ reformulate query (≤2)
         → Llama 3.1 8B generate ⟲ regenerate, strict prompt (≤2) → Phi-3.5-mini ground check
         → answer + confidence note | out-of-scope refusal | ungrounded fallback
                                        ↓
                              MongoDB update_one {output: <answer>}
                                        ↓
                              email svc finds output≠null, status=null → SMTP → {status: "sent"}

REST client → POST / (JWT) → MongoDB (same queue, same flow)
```

### MongoDB as FIFO queue

No Redis or RabbitMQ. The `prompts` collection acts as a job queue: `GET /receive-prompt` returns `find_one({'output': None}, sort=[("date_in", ASCENDING)])`. Only one job is handed out at a time; RAG processes serially.

**An empty queue answers `204 No Content`, not an error.** `prompt_service.get_prompts` returns `None` and the router turns that into a 204, which is the branch `rag/message_clients.py` already implemented (an INFO log and `EMPTY_QUEUE_INTERVAL_SEC`). That branch used to be dead code: the service raised a 404 from *inside* its `try`, the bare `except Exception` caught its own 404 and re-raised it as a 500, and the idle RAG service logged `Received response code 500` every 15 seconds while nothing was wrong. Only the Mongo read belongs inside that `try`.

### LLM backend selection (RAG service)

At startup, the first matching env var wins:

1. `OPENAI_API_KEY` → `LLMProcessorOpenAI` (ChromaDB + GPT-4-Turbo)
2. `OPENLLM_SERVER` → `LLMProcessorOpenLLM`
3. `HUGGINGFACE_SERVER` → `LLMProcessorHuggingFace`
4. _(default)_ → `LLMProcessorOllama` (LangGraph + FAISS + GPT4AllEmbeddings + Llama 3.1 8B)

Two models, both in `rag/main.py`: `llama3.1:8b-instruct-q4_K_M` (~4.9 GB weights, `OLLAMA_TEMPERATURE=0.0`)
generates the answers, `phi3.5:3.8b-mini-instruct-q4_K_M` (~2.4 GB weights, `PHI_MODEL`, `temperature=0.0`)
is the CRAG/Self-RAG control model. One Phi client is built by `build_phi_control_llm()` and shared by every
control node (relevance grading, reformulation, grounding verification) — do not instantiate one per node.

### LangGraph state graph (Ollama path only)

`LLMProcessorOllama` runs its pipeline as a compiled LangGraph `StateGraph` instead of the
`RetrievalQA` chain it used before. The other three processors still use plain LangChain chains —
the graph is scoped to the Ollama/CRAG/Self-RAG path. The shape depends on whether a `grader_llm`
was passed to the constructor:

```
# build_ollama_processor(enable_crag=False) — classic, what the baseline measures
START → retrieve → generate → END

# build_ollama_processor() — production, CRAG + Self-RAG active
START → retrieve → grade_relevance
grade_relevance ─[relevante | parcialmente_relevante]──────────────→ generate
grade_relevance ─[irrelevante, iteration_count <  2]─→ reformulate_query → retrieve
grade_relevance ─[irrelevante, iteration_count >= 2]──────────────→ out_of_scope → END
generate ─────────────────────────────────────────────────────────→ verify_grounding
verify_grounding ─[fundamentada]───────────────────────────────────────────────→ END
verify_grounding ─[no_fundamentada, generation_attempts <  2]─────→ generate (strict prompt)
verify_grounding ─[no_fundamentada, generation_attempts >= 2]─────→ ungrounded → END
```

- State is `RAGGraphState` (a `TypedDict` in `rag/llm_processor.py`): `question`, `search_query`,
  `retrieved_docs`, `relevance`, `iteration_count`, `answer`, `grounding`, `generation_attempts`.
  `question` is the student's verbatim wording and never changes; `search_query` is what the
  retriever actually sees and is what reformulation replaces, so a rewrite improves retrieval
  without changing what gets answered.
- `retrieve_node` runs `vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K}).invoke(search_query)`.
- `generate_node` joins the chunks with `DOCUMENT_SEPARATOR` (`"\n\n"`, the old
  `StuffDocumentsChain` default) into `INITIAL_PROMPT` and calls the Ollama LLM. It swaps in
  `GROUNDED_RETRY_PROMPT` when a previous answer failed grounding verification.
- The graph is compiled lazily on the first question and cached on `self.graph`.
- `ask_question` still returns a plain `str` and still swallows exceptions into an `"Error: ..."`
  string. Graph node exceptions propagate out of `graph.invoke()` and land in the same handler, so
  the error-prefix contract `resources/eval/run_baseline.py` depends on is unchanged.

`langgraph` is held at `>=0.3,<0.4` because this repo still uses LangChain 0.1-era import paths
(`langchain.chains`, `langchain.llms`, `langchain.prompts`) that LangChain 1.0 deleted, and 0.3 is
the newest line that still caps `langchain-core < 0.4`. See **Dependency pinning** below for the
rest of the lock.


### CRAG mechanism

- **Iteration cap.** `MAX_CRAG_ITERATIONS = 2` counts *reformulations*, not retrievals: the first
  retrieval is not an iteration of the correction loop, it is what the loop corrects. Worst case per
  question is 3 retrievals, 3 relevance grades and 2 reformulations. `reformulate_query_node` is the
  only node that increments `iteration_count`, and it increments even when the rewrite fails, so the
  loop is bounded by construction.
- **`parcialmente_relevante` goes straight to generation.** Only `irrelevante` opens the loop. Partial
  relevance means the bibliography does cover the topic, and `INITIAL_PROMPT` already forces
  *"No sé la respuesta basada en la información proporcionada."* for whatever is missing — reformulating
  from a partial hit would risk trading a usable answer for an out-of-scope refusal the classic
  pipeline would never have produced.
- **Defensive degradation.** `parse_relevance` keyword-matches the grader's free text (order matters:
  `irrelevante` contains `relevante`) and falls back to `parcialmente_relevante`; a grader exception is
  caught and does the same. A confused or unreachable control model must never be able to turn an
  answerable question into a refusal — worst case the pipeline degrades to classic RAG.
- **Out-of-scope answer** (`OUT_OF_SCOPE_ANSWER`) deliberately does *not* start with `"Error:"`: being
  out of scope is a valid outcome, not a pipeline failure, and `run_baseline.py` flags records by that
  prefix.
- Grader prompts see only the first `GRADER_CHUNK_PREVIEW_CHARS` (3000) of each chunk, ~80% of a
  chunk at the current size. It was 1500, justified by a comment claiming chunks were "5000
  characters"; they were 5000 *tokens* (15104 characters on average), so the graders were judging
  9% of each chunk — which is how a fabricated claim once passed grounding verification. It is
  still a bound rather than the whole chunk, because three worst-case grading calls have to stay
  affordable on a CPU-only box.

### Self-RAG grounding verification

Every answer produced by the CRAG-enabled graph is checked by `verify_grounding_node` before it
leaves the pipeline: Phi-3.5-mini is asked whether each claim in the answer can be traced back to
the retrieved chunks (`fundamentada`) or whether it adds figures, formulas, names or conclusions
that are not there (`no_fundamentada`).

- **Retry, then fall back — not one or the other.** `MAX_GENERATION_ATTEMPTS = 2` allows exactly one
  regeneration. Going straight to the fallback on the first failed verdict would let a 3.8B grader
  veto an answer the classic pipeline would have sent; retrying blindly would be near-useless
  because `OLLAMA_TEMPERATURE = 0.0` makes the same prompt over the same chunks reproduce the same
  text. So the retry uses `GROUNDED_RETRY_PROMPT`, a stricter variant of `INITIAL_PROMPT` — a
  *different* prompt is what makes a different answer possible at temperature 0. A third pass would
  only multiply the slowest call in the pipeline.
- **Separate counter.** `generation_attempts` is not folded into `iteration_count`. The two bound
  different loops (re-query with a rewritten query vs. regenerate from the same chunks), both can
  run for the same question, and sharing one counter would let CRAG starve Self-RAG. `generate_node`
  increments its counter unconditionally, the same structural bound `reformulate_query_node` uses.
- **"No sé" counts as grounded.** `GROUNDING_PROMPT` says so explicitly: `INITIAL_PROMPT` forces
  *"No sé la respuesta basada en la información proporcionada."* when the context falls short, and
  flagging that would spend two Llama calls replacing one honest refusal with another.
- **Defensive degradation, biased the other way.** `parse_grounding` mirrors `parse_relevance`
  (`no_fundamentada` contains `fundamentada`, so the negative form is matched first) but falls back
  to `fundamentada`. A verifier exception does the same. Self-RAG is a safety net; a torn net must
  let the pipeline through, never block it.
- **`UNGROUNDED_FALLBACK_ANSWER`** is worded differently from `OUT_OF_SCOPE_ANSWER` because the
  failure differs — out of scope means the bibliography does not cover the question, ungrounded
  means it may cover it but what was written could not be traced back. Like the out-of-scope answer
  it does not start with `"Error:"`: withholding an unverifiable answer is the mechanism working.
- **Cost.** Answered question, happy path: +1 Phi call. Worst case with a failed verification:
  +2 Phi calls and +1 Llama call. No extra RAM: the verifier reuses the same Phi client. Longest
  possible walk through the full graph is 13 nodes (3 retrievals, 3 grades, 2 reformulations, 2
  generations, 2 verifications, 1 fallback), inside LangGraph's default recursion limit of 25.

  **A Phi call is not the cheap one.** This paragraph used to say "Llama on CPU is the expensive
  one". Measured on 2026-09-10, on the 16 GB target box, that is backwards:

  | Node | Question 1 | Question 2 |
  |------|-----------|-----------|
  | FAISS retrieval (k=4) | 0.019 s | — |
  | Phi relevance grade | 324 s | 264 s |
  | Llama generation | 482 s | 453 s |
  | Phi grounding verify | 322 s | 287 s |
  | **Total** | **1128 s** (18.8 min) | did not finish |

  Each Phi call costs roughly what the Llama generation costs, so the two control calls on the
  happy path make the control plane ~57% of total runtime — the CRAG/Self-RAG arm roughly doubles
  the classic pipeline rather than adding a margin. Retrieval is free by comparison and is not
  worth optimising. A plausible driver is `GRADER_CHUNK_PREVIEW_CHARS` (1500 → 3000, so ~12000
  characters of preview for a 3.8B model on CPU), but **that has not been measured** — confirming
  it means re-running the same question at 1500, and the raise had a quality reason behind it.
  n = 2 questions, one run: an order-of-magnitude signal, not a benchmark.

- **Generation is bounded, and both halves of `num_ctx` have to be.** The generator is built with
  `num_predict=OLLAMA_ANSWER_TOKEN_BUDGET` (900) and the control model with `PHI_NUM_PREDICT`
  (200). Before that, the answer budget existed only as a divisor inside `CHUNK_SIZE_TOKENS` and
  was never sent to the model, so the completion side of `num_ctx` was unbounded while the prompt
  side was carefully derived. On 2026-09-10 a `GROUNDED_RETRY_PROMPT` regeneration entered a
  repetition loop and ran to 18673 tokens over 2 h 39 min at 2.05 tok/s. Past 6000 tokens
  llama.cpp does not stop, it context-shifts (`n_keep = 5`, `n_discard = 3069` per shift), so the
  chunks and the instruction header were evicted and the model generated from an empty context
  with no way to reach EOS. The queue is serial, so one runaway stalls every pending question.
  `OLLAMA_REPEAT_PENALTY = 1.1` is secondary prophylaxis — the runaway ran with
  `repeat_penalty = 1.000`, and at temperature 0 nothing else breaks a loop — but `num_predict` is
  the hard stop. This is the same unit/enforcement confusion as `OLLAMA_CONTEXT_LENGTH`, one layer
  up: a budget that only appears in an arithmetic expression constrains nothing.

### Confidence level in the answer

Answers produced by the CRAG-enabled path carry a student-facing confidence note,
appended to the answer text after a `"\n\n---\n"` separator. It is a pure reading of the
final graph state — no extra model call, nothing new measured:

| Level | Condition on the final state | Meaning |
|-------|------------------------------|---------|
| `alta` | `relevance == relevante`, `iteration_count == 0`, one generation pass | Nothing had to be corrected |
| `media` | Grounded, but `parcialmente_relevante` or `iteration_count > 0` | Retrieval needed help; the answer may be incomplete (the "partial bibliography coverage" case) |
| `baja` | Grounded only on the second generation pass | The generator drifted from the sources once on this question |
| `no_aplica` | `relevance == irrelevante` or `grounding == no_fundamentada` at the end | Out-of-scope / ungrounded fallback. **No note is appended** |

- **Precedence is by severity, and `baja` outranks the retrieval signals.** A failed
  grounding check is the only signal about the *answer text*; the other two are about
  the *retrieval*.
- **Terminal state is read from verdicts, not from string comparison.** Both loops
  overwrite their verdict on every pass, so a verdict that survives to the end can only
  have come from the node that ended the run: terminal `irrelevante` ⟺ `out_of_scope`
  fired, terminal `no_fundamentada` ⟺ `ungrounded` fired. Rewording either fallback
  constant therefore cannot start mislabelling refusals as answers.
- **Appended to the string rather than widening the return type.** `ask_question` still
  returns `str`. The confidence level exists for the student, who only ever sees the
  email body, so it has to end up in the text regardless; a richer return type would have
  forced changes in `MessageProcessor`, `BaseLLMProcessor.process_questionnaire` (shared
  with the three processors that have no graph) and `run_baseline.py` to deliver the same
  bytes. Contrast with `CragInstrumentation`, which faced the same contract and went the
  other way *because* its concern is measurement-only and never reaches the student.
- **Classic pipeline untouched.** Gated on `self.grader_llm is not None`. Without the
  control model there is no relevance or grounding verdict to derive a level from, and
  the classic arm is what the baseline harness measures.
- Because `no_aplica` appends nothing, both fallbacks stay byte-for-byte equal to their
  constants, so `run_baseline.is_out_of_scope`'s verbatim comparison still works. The note
  is a suffix, so the `"Error:"` prefix contract is unaffected too.

### FAISS index invalidation

`LLMProcessorOllama` persists the vector index to `resources/faiss_index/`. On startup it computes a SHA-256 of the chunked document content and compares it to a stored hash. If documents changed, the index is rebuilt; otherwise it loads from disk.

`load_local` is called with `allow_dangerous_deserialization=True`. A FAISS index is a pickle, and `langchain-community >= 0.0.27` refuses to unpickle one unless the caller vouches for its origin. Vouching is correct here: the else-branch of the same method writes the file with `save_local` from the course documents, and `document_has_changed` gates the load on a SHA-256 of that same content. Nothing is ever downloaded. Note there is **no volume for the index** — `docker-compose.yml` mounts only `ollama:/root/.ollama` — so it lives in the container's writable layer. That is why the missing flag looked intermittent: a first boot takes the build branch and succeeds, a restart takes the load branch and crashed.

Retrieval depth is `k=4` (`RETRIEVAL_K` in `rag/llm_processor.py`), raised from the original `k=2` to give the CRAG/Self-RAG grading nodes more candidate chunks to filter. This applies to `LLMProcessorOllama` only — the OpenAI processor still uses the LangChain default retriever.

### Chunk size is derived, not chosen

`CHUNK_SIZE_TOKENS` in `rag/main.py` is computed, never hand-picked:

```python
CHUNK_SIZE_TOKENS = (OLLAMA_NUM_CTX - OLLAMA_ANSWER_TOKEN_BUDGET - PROMPT_OVERHEAD_TOKENS) // RETRIEVAL_K
```

= `(6000 - 900 - 300) // 4` = **1200 tokens**. `num_ctx` bounds prompt *and* completion together, so
the answer budget comes off the top; `PROMPT_OVERHEAD_TOKENS` covers the longest template
(`GROUNDED_RETRY_PROMPT`) plus a student question. Measured worst case: 4939 tokens for
`INITIAL_PROMPT` and 5029 for `GROUNDED_RETRY_PROMPT`, leaving ~1000 for the answer.

It replaces a hand-picked `OLLAMA_CONTEXT_LENGTH = 5000`, and that constant is the cautionary tale
of this repo. Its name read like a model context window; its value was a chunk size, and
`TokenTextSplitter` counts **tokens**, not characters. One chunk therefore nearly filled `num_ctx`
on its own, and a `k=4` prompt measured **20137 tokens against a 6000-token window**. Ollama
truncates instead of failing, and llama.cpp keeps the tail (`n_keep=4`), so the first thing dropped
was the instruction header — *"responde utilizando únicamente la información contenida en el
documento"* and the `"No sé la respuesta..."` escape hatch. The generator was answering from
parametric memory with no way to know it, and shipping the result with a confidence note. Deriving
the value is what stops `num_ctx`, `RETRIEVAL_K` and the chunk size from drifting apart again.

### Dependency pinning

All three requirements files are full locks (`pip freeze`), each with a header naming the direct
dependencies, what was removed and why, and how to regenerate. This is not housekeeping — the
unpinned state cost real failures:

- **`bcrypt`.** Unpinned, `requirements_api.txt` resolved to `bcrypt 5.0.0`, which `passlib 1.7.4`
  (2020, final release) cannot detect as a backend. `CryptContext(schemes=["bcrypt"]).hash()` then
  raises `password cannot be longer than 72 bytes` for *any* input, a single character included.
  The build stays green and every login in the built image fails at runtime, so nothing catches it
  until a user tries to authenticate. Hence `bcrypt<4.1`.
- **5.5 GB of CUDA.** `requirements_rag.txt` declared `sentence-transformers` and `transformers`,
  which appear in no import anywhere in the repo and pulled `torch`, 20 `nvidia-*` wheels and
  `triton` onto a CPU-only box. `site-packages`: 6.4 GB → 901 MB. Embeddings come from
  `GPT4AllEmbeddings`, generation from Ollama over HTTP, and `LLMProcessorHuggingFace` posts to a
  remote endpoint and runs nothing locally.
- Also removed: `langchain-ollama` / `langchain-openai` (the code imports `langchain.llms.Ollama`
  and `langchain.chat_models.ChatOpenAI`), a duplicate `pypdf2`, and in the API `pycparser`,
  `pydantic_core`, `starlette`, `websockets`, `wsproto`, `requests`, `mysql-connector-python`
  (peewee's `MySQLDatabase` uses `pymysql`) and `pytest`.
- Added: `requests` and `typing_extensions`, imported directly by `rag/` but previously arriving
  only by accident as transitives.

`email/requirements.txt` is dead — no Dockerfile references it. The three files in use are
`requirements_{api,email,rag}.txt`.

To regenerate one: edit its direct list, resolve it in a `python:3.10-slim` container, re-run the
import check against the real modules (including runtime-only deps no code imports — `gunicorn`,
`uvicorn.workers`, `pymysql`, `multipart`, `email_validator` — and an actual `passlib` hash), then
replace the lock with `pip freeze` minus `pip`/`setuptools`/`wheel`. An audit that only follows
imports will miss exactly the deps that break in production.

### Evaluation harness (`resources/eval/`)

Everything needed to measure the two pipelines against the course bibliography, and nothing that has actually been measured. **No baseline, no A/B run and no quality score in this repository is real** — see `resources/eval/STATUS.md` for the line-by-line breakdown of what exists, what needs a live 16 GB box and what needs the course instructors.

| File | Role | Needs the pipeline's deps |
|------|------|---------------------------|
| `README.md` | `dataset.jsonl` schema. The dataset itself is intentionally absent until instructors write and validate the questions — do not generate synthetic course questions to fill it | — |
| `run_baseline.py` | Measures one arm (documented in `BASELINE.md`) | Yes |
| `compare_runs.py` | Diffs a classic and a CRAG results file (documented in `AB_TESTING.md`) | No — stdlib only |
| `eval_io.py` | Results-file format shared by both. Imports nothing from `rag/` on purpose, so the comparison runs on a laptop | No |
| `STATUS.md` | What is implemented vs. what is still unmeasured, and who unblocks each item | — |

Run both arms inside the `rag` container, on the target hardware, classic first:

```bash
docker compose exec rag python resources/eval/run_baseline.py          # classic arm
docker compose exec rag python resources/eval/run_baseline.py --crag   # agentic arm
python resources/eval/compare_runs.py --output resources/eval/results/ab_report.md
```

The harness defaults to `build_ollama_processor(enable_crag=False)` on purpose. Production turned CRAG on by default; without that explicit opt-out a bare invocation would silently measure the agentic pipeline while still labelling its output `classic-rag`, and since no baseline has been recorded yet there would be no pre-CRAG reference point left to record. `--crag` flips the same switch and writes to `results/crag_*` instead of `results/baseline_*`.

The CRAG arm additionally records per-question `crag.reformulations` / `out_of_scope`, so worst-case latency and short-circuit rate are measurable. Those counters come from `CragInstrumentation`, which wraps the processor's bound graph-node methods from the harness rather than widening `ask_question`'s `str` return type — the API and email paths depend on that contract, and it should not change for a measurement-only concern. The wrappers must be installed before the first question, since `build_graph` binds the node callables when `ask_question` lazily compiles the graph.

Two gaps in the CRAG arm's record, both known:

- **Self-RAG is timed but not counted.** Grounding verification runs and is included in the measured latency, but `CragInstrumentation` does not wrap `verify_grounding_node` or `generate_node`, so grounding verdicts, regeneration count, ungrounded-fallback rate and the confidence level have no fields of their own.
- **The confidence note is inside `generated_answer`.** Split on `"\n\n---\n"` to recover the bare answer, and score the answer rather than the note.

Records carry `schema_version` (currently `3`; `1` was the #13 shape with no `pipeline`/`crag`/`out_of_scope` fields, `2` added them, `3` renames the config key `chunk_context_length` to `chunk_size_tokens`). No results file was ever committed at any version, so nothing needed migrating.

**There is no runnable "v1.0" to compare against.** The comparison originally asked for was against the Mistral 7B / `k=2` / `RetrievalQA` system, but the model swap, the `k` bump and the LangGraph migration each replaced it *in place* rather than keeping it configurable, so `--no-crag` is the Llama 3.1 8B / `k=4` baseline, not v1.0. The last commit where v1.0 is intact is `87ca5f3`. `AB_TESTING.md` documents the two ways out — measure `87ca5f3` in a worktree with a backported harness, or redeclare the current baseline as the reference point — and the decision has not been made.

Neither `run_baseline.py` nor `compare_runs.py` ever scores an answer or estimates a hallucination rate. Those are human judgements against `expected_answer`; an automated proxy would produce a number that reads like evidence while measuring nothing.

### Peewee + FastAPI async

The MySQL connection uses a custom `ContextVar`-based `PeeweeConnectionState` (`api/model/database.py`) so connection state doesn't bleed between async requests. Every MySQL-touching endpoint depends on `get_db`.

## Key env vars

Defined in `docker-compose.yml`. The API reads them via a pydantic-settings `Settings` class; the email service via plain `os.getenv()`.

| Group | Vars |
|-------|------|
| MySQL | `DB_NAME`, `DB_USER`, `DB_PASS`, `DB_HOST`, `DB_PORT` |
| MongoDB | `MONGO_HOST`, `MONGO_PORT`, `MONGO_USER`, `MONGO_PASS`, `MONGO_DB_NAME` |
| JWT | `SECRET_KEY`, `ACCESS_TOKEN_EXPIRE_MINUTES` |
| RAG | `API_URL`, `DOCUMENT_LOCATION`, `OLLAMA_SERVER_URL` |
| Alt LLMs | `OPENAI_API_KEY`, `OPENLLM_SERVER`, `HUGGINGFACE_SERVER` |
| Email | `SMTP_SERVER/PORT/USERNAME`, `IMAP_SERVER/PORT/USERNAME`, `EMAIL_SBJT_CODE`, `EMAIL_REST_SEC` |

## Gotchas

- `Dockerfile_rag` (Python 3.11, no Ollama) is **not** used by `docker-compose.yml`. The active one is `Dockerfile-combined` (Python 3.10-slim + Ollama bundled via curl).
- **`Dockerfile-combined` installs `zstd` and pins `ARG OLLAMA_VERSION`.** Ollama's install script switched its release artifact from a gzip `.tgz` to `ollama-linux-amd64.tar.zst`; it now shells out to `unzstd` and aborts with `ERROR: This version requires zstd for extraction`, which `python:3.10-slim` does not ship. Pinning the version is the actual fix — `curl | sh` of an unversioned upstream script is a build that any vendor release can break. The script honours `OLLAMA_VERSION` (`VER_PARAM="${OLLAMA_VERSION:+?version=$OLLAMA_VERSION}"`). `ARG`, not `ENV`, so a build-time input does not persist into the runtime container.
- **`GPT4AllEmbeddings()` reaches the public internet at startup.** `gpt4all`'s `list_models()` does a bare `requests.get` on `https://gpt4all.io/models/models3.json` (301 → `raw.githubusercontent.com`) and raises on any non-200. GitHub's CDN returns an intermittent `503 Backend.max_conn reached`, which with no retry took the whole RAG service down three times in a row. `build_gpt4all_embeddings()` in `rag/main.py` now wraps it in a bounded retry (`EMBEDDING_INIT_ATTEMPTS = 6`, linear backoff, ~105s total) and re-raises after the last attempt — without embeddings there is no retrieval, so there is nothing to degrade to.
- **Two models, and both stay resident.** `entrypoint.sh` starts Ollama, waits for readiness, downloads `llama3.1:8b-instruct-q4_K_M` and `phi3.5:3.8b-mini-instruct-q4_K_M` if missing, then runs `rag/main.py` — a cold start with an empty `ollama` volume pulls ~7.3 GB of weights and is slow. At runtime the loaded footprint is larger than the weights: ~5.6 GiB for Llama plus ~3.2 GiB for Phi at `num_ctx=6000`, so ~8.5–9 GiB combined on the 16 GB target box. `enable_crag=False` never builds the Phi client, which is why a classic/baseline run stays at the pre-CRAG footprint.
- The API container blocks on `wait-for-it.sh` until MySQL port 3306 is ready before gunicorn starts.
- **Schema creation is a one-shot step in `Dockerfile_api`'s CMD, before gunicorn forks.** It used to run from `api/main.py` at import time, which meant all four `gunicorn -w 4` workers executed DDL concurrently; against an empty database they raced and the boot died with `pymysql.err.OperationalError: (1061, "Duplicate key name 'users_email'")` → `Worker failed to boot`. `create_tables` does pass `safe=True`, but that cannot help: MySQL has no `CREATE INDEX IF NOT EXISTS`, so `MySQLDatabase.safe_create_index` is `False` and peewee's early return only covers a run where the table already exists. The bug self-healed on the next boot, which is what made it look intermittent.
- Chroma telemetry is explicitly disabled in `llm_processor.py` (`ANONYMIZED_TELEMETRY=False`).
- The `forgot_password` endpoint is language-aware: pass `language=es` or `language=us` to select the email template suffix.
- `agentic-rag-informe.html` in the repo root is **untracked and not a repo doc.** It is the technical proposal the CRAG/Self-RAG work was derived from, structured as before/after: its `k=2` + Mistral 7B flow diagram is the deliberate "v1.0 actual" panel, not a stale copy of the current architecture. Do not "fix" it to match the code — it has never been committed, so it is the author's own working document.
