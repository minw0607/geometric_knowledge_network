from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .ingest import Chunk, Document, DocumentIngestor


@dataclass
class HotpotQASample:
    question_id: str
    question: str
    answer: str
    supporting_titles: list[str]
    supporting_sent_ids: list[tuple[str, int]]


class HotpotQALoader:
    def load_samples(
        self,
        filepath: Path,
        sample_size: int = 100,
        random_seed: int = 42,
    ) -> tuple[List[Document], List[Chunk], List[HotpotQASample]]:
        payload = json.loads(filepath.read_text(encoding="utf-8"))
        rng = random.Random(random_seed)
        if sample_size < len(payload):
            selected = rng.sample(payload, sample_size)
        else:
            selected = payload

        title_to_sentences: dict[str, list[str]] = {}
        samples: List[HotpotQASample] = []

        for item in selected:
            context = item.get("context", [])
            for title, sentences in context:
                title_to_sentences.setdefault(title, [])
                for sentence in sentences:
                    if sentence not in title_to_sentences[title]:
                        title_to_sentences[title].append(sentence)

            supporting_facts = item.get("supporting_facts", [])
            samples.append(
                HotpotQASample(
                    question_id=item.get("_id", ""),
                    question=item.get("question", ""),
                    answer=item.get("answer", ""),
                    supporting_titles=sorted({fact[0] for fact in supporting_facts}),
                    supporting_sent_ids=[(fact[0], int(fact[1])) for fact in supporting_facts],
                )
            )

        documents: List[Document] = []
        for title, sentences in sorted(title_to_sentences.items()):
            text = title + "\n\n" + " ".join(sentences)
            doc_id = self._normalize_title(title)
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            documents.append(
                Document(
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    source_path=str(filepath),
                    text_hash=text_hash,
                )
            )

        ingestor = DocumentIngestor()
        chunks = ingestor.chunk_documents(documents, chunk_size=500, chunk_overlap=80)
        return documents, chunks, samples

    def _normalize_title(self, title: str) -> str:
        return title.lower().replace(" ", "_").replace("/", "_")
