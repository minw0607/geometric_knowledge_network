from __future__ import annotations

import hashlib
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import GKNConfig
from .ingest import Chunk

# Heavy / optional dependencies are imported lazily so that the TF-IDF baseline
# (SimpleVectorStore) and the graph/retrieval modules can be imported and used
# without faiss, openai, or sentence-transformers installed.
try:
    import faiss
except ImportError:
    faiss = None

try:
    from openai import AzureOpenAI, OpenAI
except ImportError:
    AzureOpenAI = None
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


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


class EmbeddingVectorStore:
    def __init__(self, config: GKNConfig) -> None:
        self.config = config
        self.client = self._build_client() if config.embedding_choice != "local" else None
        self.local_model = self._build_local_model() if config.embedding_choice == "local" else None
        # Exact inner-product search on L2-normalized vectors == cosine similarity.
        # faiss is used when available; otherwise we fall back to a NumPy matmul,
        # which is exact and fast enough for the corpus sizes in this MVP.
        self.index = None
        self.embeddings = None
        self.chunks: List[Chunk] = []
        self.embedding_dim = None
        if faiss is None:
            print("[INFO] faiss not installed; using NumPy exact-search fallback for the embedding index.")

    def build(self, chunks: List[Chunk]) -> None:
        cache_path = self._cache_path(chunks)
        cached = self._load_cache(cache_path)
        if cached is not None and not self.config.force_rebuild_vector_store:
            print(f"[INFO] Loading cached vector store from {cache_path}")
            self.chunks = chunks
            self._build_index(cached)
            return

        print(f"[INFO] Building new vector store and saving to {cache_path}")
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = self._embed_texts(texts)
        normalized = self._normalize_embeddings(embeddings)
        self._build_index(normalized)
        self._save_cache(cache_path, normalized)

    def _build_index(self, normalized: np.ndarray) -> None:
        normalized = np.ascontiguousarray(normalized, dtype=np.float32)
        self.embeddings = normalized
        self.embedding_dim = normalized.shape[1]
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(normalized)
        else:
            self.index = None

    def search(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        if self.embeddings is None:
            raise ValueError("Embedding vector store has not been built.")

        query_embedding = self._embed_texts([query])
        query_embedding = self._normalize_embeddings(query_embedding)

        if self.index is not None:
            scores, indices = self.index.search(query_embedding, top_k)
            ranked = list(zip(scores[0], indices[0]))
        else:
            similarities = self.embeddings @ query_embedding[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            ranked = [(float(similarities[idx]), int(idx)) for idx in top_indices]

        results: List[RetrievalResult] = []
        for score, idx in ranked:
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=float(score),
                )
            )
        return results

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.config.embedding_choice == "local":
            if self.local_model is None:
                raise ValueError("Local embedding model is not available.")
            vectors = self.local_model.encode(texts, show_progress_bar=False)
            return np.array(vectors, dtype=np.float32)

        batch_size = 16 if self.config.embedding_choice == "small" else 8
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for attempt in range(5):
                try:
                    response = self.client.embeddings.create(
                        model=self.config.openai_embedding_model,
                        input=batch,
                    )
                    vectors.extend(item.embedding for item in response.data)
                    break
                except Exception as exc:
                    error_text = str(exc)
                    if "429" in error_text or "RateLimit" in error_text or "RateLimitReached" in error_text:
                        wait_seconds = 60 * (attempt + 1)
                        print(f"[WARN] Embedding rate limit hit. Waiting {wait_seconds} seconds before retrying...")
                        time.sleep(wait_seconds)
                        continue
                    raise
        return np.array(vectors, dtype=np.float32)

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def _build_client(self):
        if OpenAI is None or AzureOpenAI is None:
            raise ImportError(
                "openai is required for non-local embeddings. Install 'openai', "
                "or set EMBEDDING_CHOICE=local to use a local sentence-transformer."
            )
        extra_headers = {}
        if self.config.openai_apim_header_name and self.config.openai_apim_subscription_key:
            extra_headers[self.config.openai_apim_header_name] = self.config.openai_apim_subscription_key

        if self.config.openai_api_version:
            return AzureOpenAI(
                api_key=self.config.openai_api_key,
                azure_endpoint=self.config.openai_base_url,
                api_version=self.config.openai_api_version,
                default_headers=extra_headers or None,
            )

        return OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url or None,
            default_headers=extra_headers or None,
        )

    def _build_local_model(self):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for local embeddings.")
        return SentenceTransformer(self.config.local_embedding_model)

    def _cache_path(self, chunks: List[Chunk]) -> Path:
        self.config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        model_name = self.config.openai_embedding_model if self.config.embedding_choice != "local" else self.config.local_embedding_model
        signature_source = "|".join(
            [
                self.config.embedding_choice,
                model_name,
                str(self.config.hotpotqa_sample_size),
                str(self.config.hotpotqa_random_seed),
                str(len(chunks)),
                self._chunk_signature(chunks),
            ]
        )
        signature = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()[:12]
        return self.config.checkpoints_dir / f"hotpotqa_vs_{self.config.embedding_choice}_{self.config.hotpotqa_sample_size}_{self.config.hotpotqa_random_seed}_{signature}.pkl"

    def _save_cache(self, path: Path, embeddings: np.ndarray) -> None:
        payload = {
            "embeddings": embeddings,
            "embedding_choice": self.config.embedding_choice,
            "embedding_model": self.config.openai_embedding_model if self.config.embedding_choice != "local" else self.config.local_embedding_model,
            "shape": embeddings.shape,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def _load_cache(self, path: Path):
        if not path.exists():
            return None
        with open(path, "rb") as f:
            payload = pickle.load(f)
        model_name = self.config.openai_embedding_model if self.config.embedding_choice != "local" else self.config.local_embedding_model
        if payload.get("embedding_choice") != self.config.embedding_choice:
            return None
        if payload.get("embedding_model") != model_name:
            return None
        return payload.get("embeddings")

    def _chunk_signature(self, chunks: List[Chunk]) -> str:
        joined = "|".join(chunk.chunk_id + ":" + chunk.doc_id + ":" + chunk.text[:100] for chunk in chunks)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
