from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .ingest import Chunk, Document, DocumentIngestor

# Official HotpotQA distribution (https://hotpotqa.github.io/).
# The loader downloads the requested split once and caches it locally; on later
# runs it loads directly from the local copy.
HOTPOTQA_DOWNLOAD_URLS = {
    "hotpot_train_v1.1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    "hotpot_dev_distractor_v1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    "hotpot_dev_fullwiki_v1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
}

# Fallback mirror: the Hugging Face dataset hub hosts the same data as parquet.
# Used automatically when the official CMU host is unreachable. The parquet rows
# are converted back to the canonical HotpotQA JSON schema before caching.
HOTPOTQA_HF_SPLITS = {
    "hotpot_train_v1.1.json": ("distractor", "train"),
    "hotpot_dev_distractor_v1.json": ("distractor", "validation"),
    "hotpot_dev_fullwiki_v1.json": ("fullwiki", "validation"),
}
HOTPOTQA_HF_PARQUET_API = "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/{config}/{split}"


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
        filepath = Path(filepath)
        self.ensure_dataset(filepath)
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

    def ensure_dataset(self, filepath: Path) -> Path:
        """Ensure the HotpotQA file exists locally, downloading it once if missing.

        If the file is already present, it is used directly (no network call).
        Otherwise the loader tries the official CMU mirror first and, if that is
        unreachable, falls back to the Hugging Face mirror. The result is saved
        to ``filepath`` in the canonical HotpotQA JSON schema so that subsequent
        runs load directly from the local copy.
        """
        filepath = Path(filepath)
        if filepath.exists() and filepath.stat().st_size > 0:
            return filepath

        if filepath.name not in HOTPOTQA_DOWNLOAD_URLS and filepath.name not in HOTPOTQA_HF_SPLITS:
            raise FileNotFoundError(
                f"HotpotQA file not found at {filepath} and '{filepath.name}' is not a "
                f"known split. Known splits: {sorted(set(HOTPOTQA_DOWNLOAD_URLS) | set(HOTPOTQA_HF_SPLITS))}."
            )

        filepath.parent.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []

        url = HOTPOTQA_DOWNLOAD_URLS.get(filepath.name)
        if url:
            print(f"[INFO] HotpotQA file not found locally. Downloading {filepath.name} from official mirror.")
            try:
                self._download_to(url, filepath)
                print(f"[INFO] Saved HotpotQA dataset to {filepath} ({filepath.stat().st_size / 1e6:.1f} MB)")
                return filepath
            except Exception as exc:  # noqa: BLE001 (fall through to mirror)
                errors.append(f"official mirror ({url}): {exc}")
                print(f"[WARN] Official mirror failed ({exc}). Trying Hugging Face fallback...")

        hf_split = HOTPOTQA_HF_SPLITS.get(filepath.name)
        if hf_split:
            try:
                self._download_from_huggingface(hf_split[0], hf_split[1], filepath)
                print(f"[INFO] Saved HotpotQA dataset to {filepath} ({filepath.stat().st_size / 1e6:.1f} MB)")
                return filepath
            except Exception as exc:  # noqa: BLE001
                errors.append(f"huggingface ({hf_split[0]}/{hf_split[1]}): {exc}")

        raise RuntimeError(
            "Could not obtain HotpotQA dataset from any source. Tried:\n  - "
            + "\n  - ".join(errors)
            + f"\nDownload it manually to {filepath} from https://hotpotqa.github.io/."
        )

    @staticmethod
    def _download_to(url: str, destination: Path) -> None:
        """Stream a (large) file to ``destination`` via a temporary .part file."""
        tmp_path = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (trusted dataset host)
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1 << 20  # 1 MB
                next_report = 0
                with open(tmp_path, "wb") as handle:
                    while True:
                        block = response.read(chunk_size)
                        if not block:
                            break
                        handle.write(block)
                        downloaded += len(block)
                        if total and downloaded >= next_report:
                            pct = 100 * downloaded / total
                            print(f"[INFO]   downloaded {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB ({pct:.0f}%)")
                            next_report += total // 10
            tmp_path.replace(destination)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _download_from_huggingface(self, config: str, split: str, destination: Path) -> None:
        """Download the split from the Hugging Face parquet mirror and write it
        to ``destination`` in canonical HotpotQA JSON schema."""
        import pandas as pd

        api_url = HOTPOTQA_HF_PARQUET_API.format(config=config, split=split)
        print(f"[INFO] Fetching parquet file list from Hugging Face ({config}/{split}).")
        with urllib.request.urlopen(api_url, timeout=30) as response:  # noqa: S310
            parquet_urls = json.loads(response.read().decode("utf-8"))
        if not parquet_urls:
            raise RuntimeError(f"No parquet files listed for {config}/{split}.")

        tmp_json = destination.with_suffix(destination.suffix + ".part")
        record_count = 0
        try:
            with open(tmp_json, "w", encoding="utf-8") as handle:
                handle.write("[")
                first = True
                for shard_idx, purl in enumerate(parquet_urls):
                    print(f"[INFO]   downloading parquet shard {shard_idx + 1}/{len(parquet_urls)}")
                    tmp_parquet = destination.with_suffix(f".shard{shard_idx}.parquet")
                    try:
                        self._download_to(purl, tmp_parquet)
                        frame = pd.read_parquet(tmp_parquet)
                        for record in frame.to_dict(orient="records"):
                            canonical = self._hf_record_to_canonical(record)
                            handle.write(("" if first else ",") + json.dumps(canonical))
                            first = False
                            record_count += 1
                    finally:
                        if tmp_parquet.exists():
                            tmp_parquet.unlink()
                handle.write("]")
            tmp_json.replace(destination)
            print(f"[INFO]   converted {record_count} HotpotQA records from Hugging Face mirror.")
        finally:
            if tmp_json.exists():
                tmp_json.unlink()

    @staticmethod
    def _hf_record_to_canonical(record: dict) -> dict:
        """Convert a Hugging Face hotpot_qa parquet row to the canonical JSON schema."""
        context = record.get("context", {}) or {}
        titles = list(context.get("title", []))
        sentence_groups = context.get("sentences", [])
        canonical_context = [[title, list(sentences)] for title, sentences in zip(titles, sentence_groups)]

        facts = record.get("supporting_facts", {}) or {}
        fact_titles = list(facts.get("title", []))
        fact_sent_ids = list(facts.get("sent_id", []))
        canonical_facts = [[title, int(sent_id)] for title, sent_id in zip(fact_titles, fact_sent_ids)]

        return {
            "_id": record.get("id", ""),
            "question": record.get("question", ""),
            "answer": record.get("answer", ""),
            "type": record.get("type", ""),
            "level": record.get("level", ""),
            "context": canonical_context,
            "supporting_facts": canonical_facts,
        }

    def _normalize_title(self, title: str) -> str:
        return title.lower().replace(" ", "_").replace("/", "_")
