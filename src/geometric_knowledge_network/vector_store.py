from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingest import Chunk


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float


class SimpleVectorStore:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_matrix = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self.chunk_matrix = self.vectorizer.fit_transform([chunk.text for chunk in chunks])

    def search(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        if self.chunk_matrix is None:
            raise ValueError("Vector store has not been built.")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.chunk_matrix)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[RetrievalResult] = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=float(scores[idx]),
                )
            )
        return results
