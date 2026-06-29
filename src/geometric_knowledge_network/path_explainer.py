from __future__ import annotations

from typing import List

import networkx as nx


class GraphPathExplainer:
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph

    def explain(self, source_chunk_id: str, target_chunk_id: str, max_hops: int = 4) -> dict:
        try:
            path = nx.shortest_path(self.graph, source=source_chunk_id, target=target_chunk_id)
        except nx.NetworkXNoPath:
            return {
                "source_chunk_id": source_chunk_id,
                "target_chunk_id": target_chunk_id,
                "path_exists": False,
                "path": [],
            }

        if len(path) - 1 > max_hops:
            return {
                "source_chunk_id": source_chunk_id,
                "target_chunk_id": target_chunk_id,
                "path_exists": False,
                "path": [],
                "reason": "Path exceeds hop threshold",
            }

        explained_path: List[dict] = []
        for node_id in path:
            node_data = self.graph.nodes[node_id]
            explained_path.append(
                {
                    "node_id": node_id,
                    "node_type": node_data.get("node_type", "Unknown"),
                    "label": node_data.get("label") or node_data.get("title") or node_id,
                }
            )

        edge_types = []
        for left, right in zip(path[:-1], path[1:]):
            edge_types.append(self.graph.edges[left, right].get("edge_type", "Unknown"))

        return {
            "source_chunk_id": source_chunk_id,
            "target_chunk_id": target_chunk_id,
            "path_exists": True,
            "path": explained_path,
            "edge_types": edge_types,
            "hop_count": len(path) - 1,
        }
