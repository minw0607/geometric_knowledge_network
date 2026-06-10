from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import List


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source_path: str
    text_hash: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_idx: int
    end_idx: int


class DocumentIngestor:
    def load_text_documents(self, folder: Path) -> List[Document]:
        documents: List[Document] = []
        for path in sorted(folder.glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            text_hash = sha256(text.encode("utf-8")).hexdigest()
            documents.append(
                Document(
                    doc_id=path.stem,
                    title=path.stem.replace("_", " ").title(),
                    text=text,
                    source_path=str(path),
                    text_hash=text_hash,
                )
            )
        return documents

    def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        step = max(1, chunk_size - chunk_overlap)

        for doc in documents:
            text = doc.text.strip()
            for i, start in enumerate(range(0, len(text), step)):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end].strip()
                if not chunk_text:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}_chunk_{i}",
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        start_idx=start,
                        end_idx=end,
                    )
                )
                if end >= len(text):
                    break
        return chunks
