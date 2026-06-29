from __future__ import annotations

import re
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

        for doc in documents:
            segments = self._segment_document(doc.text)
            merged_segments = self._merge_segments(segments, chunk_size=chunk_size)
            cursor = 0

            for i, segment in enumerate(merged_segments):
                normalized_segment = segment.strip()
                if not normalized_segment:
                    continue

                start_idx = doc.text.find(normalized_segment, cursor)
                if start_idx == -1:
                    start_idx = cursor
                end_idx = start_idx + len(normalized_segment)
                cursor = end_idx

                if len(normalized_segment) <= chunk_size:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.doc_id}_chunk_{i}",
                            doc_id=doc.doc_id,
                            text=normalized_segment,
                            start_idx=start_idx,
                            end_idx=end_idx,
                        )
                    )
                else:
                    chunks.extend(
                        self._fallback_split(
                            doc_id=doc.doc_id,
                            chunk_index=i,
                            text=normalized_segment,
                            start_idx=start_idx,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                    )
        return chunks

    def _segment_document(self, text: str) -> List[str]:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        segments: List[str] = []

        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) <= 1:
                segments.append(block)
                continue

            current: List[str] = []
            for line in lines:
                if self._is_header_like(line) and current:
                    segments.append(" ".join(current).strip())
                    current = [line]
                else:
                    current.append(line)
            if current:
                segments.append(" ".join(current).strip())

        return segments

    def _merge_segments(self, segments: List[str], chunk_size: int) -> List[str]:
        merged: List[str] = []
        buffer = ""

        for segment in segments:
            if not buffer:
                buffer = segment
                continue

            candidate = f"{buffer}\n\n{segment}"
            if len(buffer) < max(180, chunk_size // 2) and len(candidate) <= chunk_size:
                buffer = candidate
            else:
                merged.append(buffer)
                buffer = segment

        if buffer:
            merged.append(buffer)
        return merged

    def _fallback_split(
        self,
        doc_id: str,
        chunk_index: int,
        text: str,
        start_idx: int,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[Chunk]:
        fallback_chunks: List[Chunk] = []
        step = max(1, chunk_size - chunk_overlap)

        for j, start in enumerate(range(0, len(text), step)):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue
            fallback_chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_index}_{j}",
                    doc_id=doc_id,
                    text=chunk_text,
                    start_idx=start_idx + start,
                    end_idx=start_idx + end,
                )
            )
            if end >= len(text):
                break

        return fallback_chunks

    def _is_header_like(self, line: str) -> bool:
        return bool(re.match(r"^[A-Z][A-Za-z0-9\s\-/&]{0,80}$", line)) and not line.endswith(".")
