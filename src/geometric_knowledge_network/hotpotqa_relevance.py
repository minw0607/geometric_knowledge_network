from __future__ import annotations

from typing import List, Set

from .hotpotqa_loader import HotpotQASample
from .ingest import Chunk


class HotpotRelevanceMapper:
    def relevant_chunk_ids(self, sample: HotpotQASample, chunks: List[Chunk]) -> Set[str]:
        relevant_docs = self.relevant_doc_ids(sample)
        return {chunk.chunk_id for chunk in chunks if chunk.doc_id in relevant_docs}

    def relevant_doc_ids(self, sample: HotpotQASample) -> Set[str]:
        """Normalized supporting-document ids (the multi-hop targets)."""
        return {self._normalize(title) for title in sample.supporting_titles}

    def _normalize(self, title: str) -> str:
        return title.lower().replace(" ", "_").replace("/", "_")
