"""Index invalidation tests — the precondition for persisting the FAISS index (#33).

`resources/faiss_index/` had no volume, so every container recreation wiped it and it
was rebuilt from scratch: ~14.5 min for 148 chunks with bge-m3 on CPU. Mounting a volume
removes that cost and, in the same move, removes the accident that was protecting the
pipeline from a worse bug — a persisted index reloaded against a *different embedding
model* would serve 384-dimensional vectors to a 1024-dimensional model, and retrieval
would degrade while looking healthy.

`get_index_hash` is what makes the volume safe, and it is a pure function of the
embedding fingerprint and the chunk text. So the issue's acceptance criterion — "change
the embedding model, restart, confirm the index rebuilds" — is checked here in
milliseconds instead of by a 14.5-minute restart, and stays checked.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rag.llm_processor import LLMProcessorOllama  # noqa: E402

CHUNKS = ["Un FET es un transistor de efecto de campo.", "La puerta no maneja corriente."]
OTHER_CHUNKS = ["Un FET es un transistor de efecto de campo.", "La puerta casi no conduce."]


class StubEmbedding:
    """Embedding stub exposing the attribute LangChain models expose."""

    def __init__(self, model=None, model_name=None):
        if model is not None:
            self.model = model
        if model_name is not None:
            self.model_name = model_name


def processor_for(embedding):
    return LLMProcessorOllama(llm=None, embedding=embedding, context_length=1200,
                              grader_llm=None)


class TestEmbeddingFingerprint:
    """The fingerprint is what pulls the model into the hash."""

    def test_reads_the_model_attribute(self):
        assert "bge-m3" in processor_for(StubEmbedding(model="bge-m3")).embedding_fingerprint()

    def test_falls_back_to_model_name(self):
        """Different LangChain embedding classes expose one or the other."""
        fingerprint = processor_for(StubEmbedding(model_name="bge-m3")).embedding_fingerprint()
        assert "bge-m3" in fingerprint

    def test_an_unidentifiable_embedding_hashes_as_a_question_mark(self):
        """Degrading to "?" stops distinguishing that one case, which is no worse than
        not hashing the model at all — and strictly better than raising during startup."""
        assert processor_for(StubEmbedding()).embedding_fingerprint().endswith(":?")

    def test_the_class_name_is_part_of_the_fingerprint(self):
        """Two providers can serve the same model tag at different dimensions."""
        assert "StubEmbedding" in processor_for(StubEmbedding(model="bge-m3")).embedding_fingerprint()


class TestIndexHashInvalidation:
    """What a persisted index is allowed to survive, and what must invalidate it."""

    def test_the_same_corpus_and_model_reuse_the_index(self):
        """Without this the volume buys nothing: every boot would rebuild anyway."""
        first = processor_for(StubEmbedding(model="bge-m3")).get_index_hash(CHUNKS)
        second = processor_for(StubEmbedding(model="bge-m3")).get_index_hash(CHUNKS)
        assert first == second

    def test_changing_the_embedding_model_invalidates_the_index(self):
        """**This is the acceptance criterion of #33.** Before `6729bec` the hash covered
        chunk text alone, so a persisted index built with GPT4All's 384 dimensions would
        have been reloaded against bge-m3's 1024. The missing volume was hiding that by
        wiping the index on every rebuild — luck, not design, and the volume removes it."""
        gpt4all = processor_for(StubEmbedding(model_name="all-MiniLM-L6-v2")).get_index_hash(CHUNKS)
        bge = processor_for(StubEmbedding(model="bge-m3")).get_index_hash(CHUNKS)
        assert gpt4all != bge

    def test_changing_the_embedding_class_invalidates_the_index(self):
        class OtherEmbedding(StubEmbedding):
            pass

        same_tag_other_class = processor_for(OtherEmbedding(model="bge-m3")).get_index_hash(CHUNKS)
        assert same_tag_other_class != processor_for(StubEmbedding(model="bge-m3")).get_index_hash(CHUNKS)

    def test_changing_the_corpus_invalidates_the_index(self):
        """The extraction-noise filter (#30) changed chunk text and count, so this is the
        path that must fire whenever the corpus or the loader changes."""
        embedding = StubEmbedding(model="bge-m3")
        assert (processor_for(embedding).get_index_hash(CHUNKS)
                != processor_for(embedding).get_index_hash(OTHER_CHUNKS))

    def test_chunk_order_is_part_of_the_hash(self):
        """Retrieval results are order-independent, but the stored index is not rebuilt
        from a set — a reordered corpus is a different index."""
        embedding = StubEmbedding(model="bge-m3")
        assert (processor_for(embedding).get_index_hash(CHUNKS)
                != processor_for(embedding).get_index_hash(list(reversed(CHUNKS))))

    def test_the_hash_is_a_sha256_hex_digest(self):
        digest = processor_for(StubEmbedding(model="bge-m3")).get_index_hash(CHUNKS)
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)

    @pytest.mark.parametrize("chunks", [[], [""]])
    def test_a_degenerate_corpus_still_hashes(self, chunks):
        """An empty corpus must not raise during startup; it should simply not match the
        hash of a real one."""
        digest = processor_for(StubEmbedding(model="bge-m3")).get_index_hash(chunks)
        assert len(digest) == 64
        assert digest != processor_for(StubEmbedding(model="bge-m3")).get_index_hash(CHUNKS)
