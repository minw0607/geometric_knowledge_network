from __future__ import annotations

from typing import List, Set

from .hotpotqa_loader import HotpotQASample
from .ingest import Chunk


class HotpotRelevanceMapper:
    def relevant_chunk_ids(self, sample: HotpotQASample, chunks: List[Chunk]) -> Set[str]:
        relevant_doc_ids = {self._normalize(title) for title in sample.supporting_titles}
        return {chunk.chunk_id for chunk in chunks if chunk.doc_id in relevant_doc_ids}

    def _normalize(self, title: str) -> str:
        return title.lower().replace(" ", "_").replace("/", "_")
