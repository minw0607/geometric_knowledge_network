from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx


def draw_subgraph(graph: nx.Graph, center_node: str, radius: int = 1) -> None:
    nodes = nx.single_source_shortest_path_length(graph, center_node, cutoff=radius).keys()
    subgraph = graph.subgraph(nodes)
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(subgraph, seed=42)
    labels = {
        node: graph.nodes[node].get("label")
        or graph.nodes[node].get("title")
        or node
        for node in subgraph.nodes
    }
    nx.draw(subgraph, pos, with_labels=False, node_size=700, font_size=8)
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8)
    plt.title(f"Subgraph around: {center_node}")
    plt.axis("off")
    plt.show()
