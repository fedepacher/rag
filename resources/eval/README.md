# Evaluation dataset

Teacher-validated question/answer pairs used to measure retrieval and answer
quality of the RAG pipeline against the course bibliography.

**Status: empty on purpose.** `dataset.jsonl` does not exist yet. The questions
and their expected answers must be written and validated by the course
instructors — they cannot be generated automatically without making the
evaluation meaningless. This directory only fixes the storage format so the
real dataset slots in without any code change.

## File

| Path | Format | Content |
|------|--------|---------|
| `resources/eval/dataset.jsonl` | JSON Lines (UTF-8, one object per line) | One validated Q&A pair per line |

JSON Lines is used instead of a single JSON array so that entries can be
appended one at a time and diffs stay line-scoped in git as instructors submit
questions in batches.

## Schema

Every line is a standalone JSON object with these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Stable unique identifier, never reused. Convention: `<unit>-<nnn>`, e.g. `amplif-003`. |
| `question` | string | yes | The question exactly as a student would send it by email (Spanish, as the course is taught). |
| `expected_answer` | string | yes | Reference answer approved by an instructor. Used as the ground truth for answer grading. |
| `source_doc` | string | yes | Filename of the course document under `resources/files/` that supports the answer. Used to check whether retrieval pulled the right document. |
| `source_locator` | string | no | Where inside `source_doc` the answer lives (page, section or heading). Helps diagnose chunk-level retrieval misses. |
| `validated_by` | string | yes | Name or initials of the instructor who approved `expected_answer`. |
| `validated_at` | string | yes | ISO 8601 date of validation (`YYYY-MM-DD`). |
| `topic` | string | no | Course unit or subject area, for per-topic score breakdowns. |
| `difficulty` | string | no | One of `basic`, `intermediate`, `advanced`. |
| `notes` | string | no | Free-form remarks from the instructor (ambiguities, accepted alternative answers). |

Unknown extra fields are tolerated by convention, but prefer extending this
table over inventing ad-hoc keys.

## Example entry

The following line is a **NON-REAL PLACEHOLDER** written only to illustrate the
schema. It is not part of the dataset, it was not validated by any instructor,
and it must never be copied into `dataset.jsonl`.

```json
{"id": "example-000", "question": "PLACEHOLDER - not a real course question", "expected_answer": "PLACEHOLDER - not a real validated answer", "source_doc": "placeholder.pdf", "source_locator": "p. 0", "validated_by": "PLACEHOLDER", "validated_at": "1970-01-01", "topic": "placeholder", "difficulty": "basic", "notes": "Schema illustration only."}
```

## Adding entries

1. Instructors draft questions covering the course bibliography.
2. An instructor reviews and approves each `expected_answer`, then fills
   `validated_by` and `validated_at`.
3. Append one JSON object per line to `dataset.jsonl` and commit it. The file is
   versioned in git — treat entries as append-only and bump `validated_at` when
   an answer is revised, rather than silently editing it in place.

No evaluation runner reads this file yet.
