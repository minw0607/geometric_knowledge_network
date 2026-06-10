from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Set

from .ingest import Chunk


class ConceptExtractor:
    def __init__(self, keywords: Iterable[str]):
        self.keywords = {kw.lower() for kw in keywords}

    def extract_concepts(self, chunk: Chunk) -> List[str]:
        text = chunk.text.lower()
        found: Set[str] = set()

        for kw in self.keywords:
            if kw in text:
                found.add(kw)

        token_counts = Counter(re.findall(r"\b[a-z]{5,}\b", text))
        for token, count in token_counts.items():
            if count >= 2:
                found.add(token)

        return sorted(found)
