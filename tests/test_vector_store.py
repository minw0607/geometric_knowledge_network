"""Regression tests for EmbeddingVectorStore numerical hygiene.

Non-finite embeddings (NaN/inf) from a provider must not propagate into the
similarity matmul — otherwise search() raises divide-by-zero / overflow /
invalid-value RuntimeWarnings and returns NaN scores that corrupt ranking.
"""
from __future__ import annotations

import warnings

import numpy as np

from geometric_knowledge_network.ingest import Chunk
from geometric_knowledge_network.vector_store import EmbeddingVectorStore


def _bare_store() -> EmbeddingVectorStore:
    # Bypass __init__ (which would try to reach a cloud/local model); we only
    # exercise the pure-numpy normalization + search-fallback paths.
    return EmbeddingVectorStore.__new__(EmbeddingVectorStore)


def test_normalize_zeros_out_non_finite_rows():
    store = _bare_store()
    emb = np.array(
        [[1.0, 2.0, 3.0], [np.inf, 0.0, 0.0], [np.nan, 1.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning fails the test
        out = store._normalize_embeddings(emb)
    assert np.isfinite(out).all()
    # the clean row is unit-normalized
    np.testing.assert_allclose(np.linalg.norm(out[0]), 1.0, rtol=1e-5)


def test_search_fallback_no_warnings_and_finite_scores():
    store = _bare_store()
    store.index = None  # force the numpy matmul fallback (faiss path skipped)
    store.chunks = [
        Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", text=f"text {i}", start_idx=0, end_idx=6)
        for i in range(3)
    ]
    raw = np.array([[1.0, 0.0, 0.0], [np.inf, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    store.embeddings = store._normalize_embeddings(raw)
    # stub the query embedding so no network/model is needed
    store._embed_texts = lambda texts: np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = store.search("anything", top_k=3)

    assert len(results) == 3
    assert all(np.isfinite(r.score) for r in results)
    # the clean row aligned with the query ranks first; the inf row didn't win
    assert results[0].doc_id == "d0"
