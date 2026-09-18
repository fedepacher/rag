"""Provenance tests for the evaluation results format (issue #32).

Four runs of the same six questions were compared as if they were repeats of one
system. They were four different systems: every transition in the recorded A->D table
spans at least one functional commit, and one of them (``dbed292``) deleted a course
PDF, which changes retrieval without changing any code the reader can see. The fix is
that a results file states what produced it, and the comparison refuses to draw a
conclusion across two different systems.

Everything exercised here is pure or shells out to ``git``, so the module imports
nothing from ``rag/`` and needs no model, no corpus and no FAISS index -- the same
constraint ``eval_io`` itself is held to.
"""
import json
import os
import subprocess
import sys

import pytest

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "resources", "eval")
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import compare_runs  # noqa: E402
from eval_io import (  # noqa: E402
    SCHEMA_VERSION,
    build_provenance,
    git_provenance,
    provenance_mismatch,
    record_provenance,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_HASH = "a" * 64
OTHER_CORPUS_HASH = "b" * 64
COMMIT = "0" * 40
OTHER_COMMIT = "1" * 40


class TestSchemaVersion:
    """The provenance block is a format change and has to be announced as one."""

    def test_schema_version_is_at_least_4(self):
        assert SCHEMA_VERSION >= 4


def _git_available():
    """Report whether a usable ``git`` exists, because the rag image ships none."""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


needs_git = pytest.mark.skipif(not _git_available(),
                               reason="no git binary; the rag image does not ship one")


class TestGitProvenanceFromEnvironment:
    """The environment variable is the path that works where the harness actually runs.

    ``run_baseline.py`` is invoked as ``docker compose exec rag python ...``. The rag
    image has no ``git`` binary and no ``.git`` directory -- the source is COPYed in at
    build time -- so shelling out could only ever return nulls there. The commit the
    image was built from, injected at build time, is both available and the honest
    answer for a run of that image.
    """

    def test_the_environment_variable_supplies_the_commit(self, monkeypatch):
        monkeypatch.setenv("RAG_COMMIT", COMMIT)
        assert git_provenance(REPO_ROOT)['commit'] == COMMIT

    def test_the_environment_variable_wins_over_the_git_binary(self, monkeypatch):
        """Inside the image the env var is the truth; on a bind mount both exist and
        the env var still describes what was built."""
        monkeypatch.setenv("RAG_COMMIT", COMMIT)
        monkeypatch.setenv("RAG_DIRTY", "0")
        assert git_provenance(REPO_ROOT) == {"commit": COMMIT, "dirty": False}

    def test_the_dirty_flag_is_read_from_its_own_variable(self, monkeypatch):
        monkeypatch.setenv("RAG_COMMIT", COMMIT)
        monkeypatch.setenv("RAG_DIRTY", "1")
        assert git_provenance(REPO_ROOT)['dirty'] is True

    def test_a_commit_without_a_dirty_variable_reads_as_unknown_not_clean(self, monkeypatch):
        """Absent information is not good news. Defaulting to clean would let a build
        from a modified tree claim its SHA identifies it."""
        monkeypatch.delenv("RAG_DIRTY", raising=False)
        monkeypatch.setenv("RAG_COMMIT", COMMIT)
        assert git_provenance(REPO_ROOT)['dirty'] is None

    def test_a_blank_variable_is_ignored(self, monkeypatch):
        """An unset build arg expands to the empty string, which must not be recorded
        as a commit called "";  it has to fall through to the git fallback."""
        monkeypatch.setenv("RAG_COMMIT", "")
        assert git_provenance(REPO_ROOT)['commit'] != ""


class TestGitProvenanceFromGit:
    """The fallback, for a developer running the harness or the tests on the host."""

    @needs_git
    def test_reports_the_current_commit_of_this_repository(self, monkeypatch):
        monkeypatch.delenv("RAG_COMMIT", raising=False)
        expected = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                  capture_output=True, text=True, check=True).stdout.strip()
        assert git_provenance(REPO_ROOT)['commit'] == expected

    @needs_git
    def test_commit_is_a_full_length_hex_sha(self, monkeypatch):
        monkeypatch.delenv("RAG_COMMIT", raising=False)
        commit = git_provenance(REPO_ROOT)['commit']
        assert len(commit) == 40
        assert all(character in "0123456789abcdef" for character in commit)

    @needs_git
    def test_untracked_files_do_not_make_the_tree_dirty(self, tmp_path, monkeypatch):
        """The course PDFs are untracked by design (#35), so counting untracked files
        would report every run as dirty and the flag would carry no information. What
        makes a commit SHA a lie is a modified *tracked* file."""
        monkeypatch.delenv("RAG_COMMIT", raising=False)
        monkeypatch.delenv("RAG_DIRTY", raising=False)
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
        (tmp_path / "tracked.txt").write_text("original\n")
        subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)

        (tmp_path / "untracked.pdf").write_text("a course document nobody commits\n")
        assert git_provenance(str(tmp_path))['dirty'] is False

        (tmp_path / "tracked.txt").write_text("modified\n")
        assert git_provenance(str(tmp_path))['dirty'] is True

    def test_degrades_to_nulls_without_git_or_a_repository(self, tmp_path, monkeypatch):
        """A results file produced without provenance is worth less, but the run must
        not die for it -- the measurement is the expensive part. This is also exactly
        what happens inside the rag image when no build arg was passed."""
        monkeypatch.delenv("RAG_COMMIT", raising=False)
        monkeypatch.delenv("RAG_DIRTY", raising=False)
        assert git_provenance(str(tmp_path)) == {"commit": None, "dirty": None}


class TestBuildProvenance:
    """``build_provenance`` assembles the block written into records and summaries."""

    def test_carries_commit_dirty_and_corpus_hash(self):
        provenance = build_provenance(CORPUS_HASH, repo_root=REPO_ROOT)
        assert set(provenance) == {"commit", "dirty", "corpus_hash"}
        assert provenance['corpus_hash'] == CORPUS_HASH

    def test_corpus_hash_may_be_unknown(self):
        """A resumed run recomputes its summary without rebuilding the index, so the
        corpus hash is not always available. Null says "not recorded", which is not the
        same fact as a hash that happens to differ."""
        assert build_provenance(None, repo_root=REPO_ROOT)['corpus_hash'] is None


class TestRecordProvenance:
    """Reading provenance back has to be safe on the schemas that predate it."""

    def test_reads_the_block_from_a_record(self):
        record = {"provenance": {"commit": COMMIT, "dirty": False, "corpus_hash": CORPUS_HASH}}
        assert record_provenance(record) == {"commit": COMMIT, "dirty": False,
                                            "corpus_hash": CORPUS_HASH}

    def test_a_record_without_provenance_reads_as_all_unknown(self):
        assert record_provenance({"id": "q1"}) == {"commit": None, "dirty": None,
                                                  "corpus_hash": None}

    def test_a_partial_block_fills_the_missing_fields_with_none(self):
        assert record_provenance({"provenance": {"commit": COMMIT}}) == {
            "commit": COMMIT, "dirty": None, "corpus_hash": None}


class TestProvenanceMismatch:
    """The pure comparison the guard is built on."""

    def test_identical_provenance_has_no_mismatch(self):
        block = {"commit": COMMIT, "dirty": False, "corpus_hash": CORPUS_HASH}
        assert provenance_mismatch(block, dict(block)) == []

    def test_a_different_commit_is_a_mismatch(self):
        left = {"commit": COMMIT, "dirty": False, "corpus_hash": CORPUS_HASH}
        right = {"commit": OTHER_COMMIT, "dirty": False, "corpus_hash": CORPUS_HASH}
        assert provenance_mismatch(left, right) == ["commit"]

    def test_a_different_corpus_hash_is_a_mismatch(self):
        """This is the one a commit SHA cannot catch: dbed292 deleted a course PDF, and
        with the documents untracked the corpus can change with no commit at all."""
        left = {"commit": COMMIT, "dirty": False, "corpus_hash": CORPUS_HASH}
        right = {"commit": COMMIT, "dirty": False, "corpus_hash": OTHER_CORPUS_HASH}
        assert provenance_mismatch(left, right) == ["corpus_hash"]

    def test_both_fields_can_mismatch_at_once(self):
        left = {"commit": COMMIT, "dirty": False, "corpus_hash": CORPUS_HASH}
        right = {"commit": OTHER_COMMIT, "dirty": False, "corpus_hash": OTHER_CORPUS_HASH}
        assert provenance_mismatch(left, right) == ["commit", "corpus_hash"]

    def test_a_dirty_tree_on_either_side_is_a_mismatch_even_at_the_same_commit(self):
        """Two dirty runs at the same SHA are not known to be the same system: the
        working tree is not captured by the hash, so the SHA stops being an identity."""
        left = {"commit": COMMIT, "dirty": True, "corpus_hash": CORPUS_HASH}
        right = {"commit": COMMIT, "dirty": True, "corpus_hash": CORPUS_HASH}
        assert provenance_mismatch(left, right) == ["dirty"]

    def test_unknown_provenance_on_either_side_is_a_mismatch(self):
        """Schema v3 and earlier recorded nothing. Treating "unknown" as "matches" is
        exactly the mistake this issue exists to stop -- it is what let four runs be
        compared as one."""
        known = {"commit": COMMIT, "dirty": False, "corpus_hash": CORPUS_HASH}
        unknown = {"commit": None, "dirty": None, "corpus_hash": None}
        assert provenance_mismatch(known, unknown) == ["commit", "corpus_hash"]


class TestAssertSameProvenance:
    """The guard ``compare_runs`` applies before it diffs anything."""

    @staticmethod
    def _records(commit, corpus_hash, dirty=False):
        return [{"id": "q1", "provenance": {"commit": commit, "dirty": dirty,
                                            "corpus_hash": corpus_hash}}]

    def test_matching_runs_pass(self):
        records = self._records(COMMIT, CORPUS_HASH)
        compare_runs.assert_same_provenance(records, list(records), "a.jsonl", "b.jsonl")

    def test_a_different_commit_is_refused(self):
        with pytest.raises(ValueError) as err:
            compare_runs.assert_same_provenance(
                self._records(COMMIT, CORPUS_HASH),
                self._records(OTHER_COMMIT, CORPUS_HASH), "a.jsonl", "b.jsonl")
        assert "commit" in str(err.value)

    def test_a_different_corpus_hash_is_refused(self):
        with pytest.raises(ValueError) as err:
            compare_runs.assert_same_provenance(
                self._records(COMMIT, CORPUS_HASH),
                self._records(COMMIT, OTHER_CORPUS_HASH), "a.jsonl", "b.jsonl")
        assert "corpus_hash" in str(err.value)

    def test_records_predating_the_provenance_block_are_refused(self):
        with pytest.raises(ValueError):
            compare_runs.assert_same_provenance([{"id": "q1"}], [{"id": "q1"}],
                                                "a.jsonl", "b.jsonl")

    def test_the_error_names_both_files_so_the_operator_knows_which_to_rerun(self):
        with pytest.raises(ValueError) as err:
            compare_runs.assert_same_provenance(
                self._records(COMMIT, CORPUS_HASH),
                self._records(OTHER_COMMIT, CORPUS_HASH), "left.jsonl", "right.jsonl")
        message = str(err.value)
        assert "left.jsonl" in message and "right.jsonl" in message

    def test_the_error_explains_how_to_override(self):
        """An operator who knows the two runs are comparable has to be told the way
        through, or they will edit the results files by hand."""
        with pytest.raises(ValueError) as err:
            compare_runs.assert_same_provenance(
                self._records(COMMIT, CORPUS_HASH),
                self._records(OTHER_COMMIT, CORPUS_HASH), "a.jsonl", "b.jsonl")
        assert "--force-mismatched-provenance" in str(err.value)

    def test_force_lets_a_deliberate_cross_commit_comparison_through(self):
        compare_runs.assert_same_provenance(
            self._records(COMMIT, CORPUS_HASH),
            self._records(OTHER_COMMIT, OTHER_CORPUS_HASH), "a.jsonl", "b.jsonl",
            force=True)

    def test_a_run_with_inconsistent_provenance_within_one_file_is_refused(self):
        """Resuming a run after a code change writes records from two systems into one
        file, and nothing else would notice."""
        mixed = (self._records(COMMIT, CORPUS_HASH) +
                 self._records(OTHER_COMMIT, CORPUS_HASH))
        with pytest.raises(ValueError) as err:
            compare_runs.assert_same_provenance(mixed, self._records(COMMIT, CORPUS_HASH),
                                                "mixed.jsonl", "b.jsonl")
        assert "mixed.jsonl" in str(err.value)


class TestProvenanceReachesTheReport:
    """A comparison states what it compared, so a report cannot be read out of context."""

    def test_the_comparison_carries_both_provenance_blocks(self, tmp_path):
        baseline_path = tmp_path / "baseline_m_k4.jsonl"
        crag_path = tmp_path / "crag_m_k4.jsonl"
        provenance = {"commit": COMMIT, "dirty": False, "corpus_hash": CORPUS_HASH}
        for path, pipeline in ((baseline_path, "classic-rag"), (crag_path, "crag")):
            record = {"schema_version": SCHEMA_VERSION, "id": "q1", "pipeline": pipeline,
                      "question": "¿Qué es un FET?", "generated_answer": "Un transistor.",
                      "latency_sec": 1.0, "pipeline_error": False, "out_of_scope": False,
                      "crag": {"reformulations": 0} if pipeline == "crag" else None,
                      "provenance": provenance,
                      "scores": {}, "scored_by": None, "scored_at": None}
            path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding='utf-8')

        comparison = compare_runs.build_comparison(str(baseline_path), str(crag_path))
        assert comparison['provenance']['baseline'] == provenance
        assert comparison['provenance']['crag'] == provenance
