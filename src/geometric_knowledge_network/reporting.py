from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


class ArtifactManager:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.results_dir = base_dir / "results"
        self.reports_dir = base_dir / "reports"
        self.figures_dir = base_dir / "figures"
        self.graph_dir = base_dir / "graph"

        for folder in [self.base_dir, self.results_dir, self.reports_dir, self.figures_dir, self.graph_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    def timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_json(self, payload: Any, filename: str, subdir: str = "results") -> Path:
        target_dir = self._target_dir(subdir)
        filepath = target_dir / filename
        normalized = self._normalize(payload)
        filepath.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
        return filepath

    def save_dataframe(self, dataframe: pd.DataFrame, filename: str, subdir: str = "reports") -> Path:
        target_dir = self._target_dir(subdir)
        filepath = target_dir / filename
        dataframe.to_csv(filepath, index=False)
        return filepath

    def save_text(self, text: str, filename: str, subdir: str = "reports") -> Path:
        target_dir = self._target_dir(subdir)
        filepath = target_dir / filename
        filepath.write_text(text, encoding="utf-8")
        return filepath

    def save_graph_summary(self, graph: nx.Graph, filename: str = "graph_summary.json") -> Path:
        summary = {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "node_types": self._count_values([graph.nodes[node].get("node_type", "Unknown") for node in graph.nodes]),
            "edge_types": self._count_values([graph.edges[edge].get("edge_type", "Unknown") for edge in graph.edges]),
        }
        return self.save_json(summary, filename, subdir="graph")

    def _target_dir(self, subdir: str) -> Path:
        mapping = {
            "results": self.results_dir,
            "reports": self.reports_dir,
            "figures": self.figures_dir,
            "graph": self.graph_dir,
        }
        return mapping[subdir]

    def _normalize(self, payload: Any) -> Any:
        if is_dataclass(payload):
            return asdict(payload)
        if isinstance(payload, dict):
            return {key: self._normalize(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [self._normalize(item) for item in payload]
        if isinstance(payload, tuple):
            return [self._normalize(item) for item in payload]
        if isinstance(payload, set):
            return sorted(self._normalize(item) for item in payload)
        if hasattr(payload, "__dict__"):
            return {key: self._normalize(value) for key, value in payload.__dict__.items()}
        return payload

    def _count_values(self, values: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return counts
