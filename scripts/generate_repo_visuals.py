from __future__ import annotations

from pathlib import Path
import subprocess
import sys

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle
except ImportError:
    plt = None
    FancyArrowPatch = None
    Rectangle = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from geometric_knowledge_network.config import GKNConfig
from geometric_knowledge_network.extraction import ConceptExtractor
from geometric_knowledge_network.graph_builder import KnowledgeNetworkBuilder
from geometric_knowledge_network.hybrid_retriever import HybridRetriever
from geometric_knowledge_network.ingest import DocumentIngestor
from geometric_knowledge_network.vector_store import SimpleVectorStore
from geometric_knowledge_network.visualization import draw_subgraph


DOCS_IMAGE_DIR = REPO_ROOT / "docs" / "images"
DOCS_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_matplotlib() -> None:
    if plt is not None and FancyArrowPatch is not None and Rectangle is not None:
        return

    python_exe = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if python_exe.exists():
        subprocess.run([str(python_exe), "-m", "pip", "show", "matplotlib"], check=False)

    raise ImportError("matplotlib is required to generate visuals.")


def _box(ax, xy, width, height, text, fc="#EAF2FF", ec="#3A6EA5", fontsize=11):
    rect = Rectangle(xy, width, height, facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, start, end, color="#555555"):
    arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=15, linewidth=1.5, color=color)
    ax.add_patch(arrow)


def generate_architecture_overview(save_path: Path) -> None:
    _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    _box(ax, (0.5, 2.2), 1.8, 1.0, "Documents")
    _box(ax, (2.8, 2.2), 1.8, 1.0, "Chunking")
    _box(ax, (5.1, 3.4), 2.2, 1.0, "Vector Index")
    _box(ax, (5.1, 1.0), 2.2, 1.0, "Typed Extraction")
    _box(ax, (8.0, 2.2), 2.6, 1.0, "GKN Graph")
    _box(ax, (11.2, 2.2), 2.2, 1.0, "Hybrid Retrieval")
    _box(ax, (11.2, 0.6), 2.2, 1.0, "Artifacts & Reports", fc="#EEF9EE", ec="#3F8F3F")

    _arrow(ax, (2.3, 2.7), (2.8, 2.7))
    _arrow(ax, (4.6, 2.7), (5.1, 3.9))
    _arrow(ax, (4.6, 2.7), (5.1, 1.5))
    _arrow(ax, (7.3, 3.9), (8.0, 2.7))
    _arrow(ax, (7.3, 1.5), (8.0, 2.7))
    _arrow(ax, (10.6, 2.7), (11.2, 2.7))
    _arrow(ax, (12.3, 2.2), (12.3, 1.6))

    ax.text(7, 5.3, "Geometric Knowledge Network Architecture", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(7, 4.8, "Local-first hybrid retrieval pipeline", ha="center", va="center", fontsize=11, color="#444444")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_baseline_vs_hybrid(save_path: Path) -> None:
    _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(3.5, 5.3, "Baseline Vector Retrieval", ha="center", fontsize=15, fontweight="bold")
    ax.text(10.5, 5.3, "GKN-Enhanced Hybrid Retrieval", ha="center", fontsize=15, fontweight="bold")

    _box(ax, (1.0, 3.5), 1.6, 0.9, "Query")
    _box(ax, (3.2, 3.5), 2.2, 0.9, "Vector Similarity")
    _box(ax, (6.0, 3.5), 1.8, 0.9, "Top-k Chunks")
    _arrow(ax, (2.6, 3.95), (3.2, 3.95))
    _arrow(ax, (5.4, 3.95), (6.0, 3.95))

    _box(ax, (8.1, 4.0), 1.6, 0.9, "Query")
    _box(ax, (10.1, 4.0), 2.0, 0.9, "Vector Hits")
    _box(ax, (10.1, 2.5), 2.0, 0.9, "Graph Expansion")
    _box(ax, (12.6, 3.2), 1.8, 0.9, "Re-ranked Chunks")
    _arrow(ax, (9.7, 4.45), (10.1, 4.45))
    _arrow(ax, (11.1, 4.0), (11.1, 3.4))
    _arrow(ax, (12.1, 4.45), (12.6, 3.65))
    _arrow(ax, (12.1, 2.95), (12.6, 3.45))

    ax.text(3.5, 1.5, "Finds semantically similar chunks", ha="center", fontsize=11, color="#444444")
    ax.text(10.5, 1.5, "Adds structurally connected chunks\nthrough typed graph relations", ha="center", fontsize=11, color="#444444")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_math_intuition(save_path: Path) -> None:
    _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(3.5, 5.3, "Semantic Closeness", ha="center", fontsize=15, fontweight="bold")
    ax.text(10.5, 5.3, "Structural Closeness", ha="center", fontsize=15, fontweight="bold")

    _box(ax, (1.0, 3.6), 1.6, 0.9, "Query")
    _box(ax, (3.1, 3.6), 2.2, 0.9, "Chunk A")
    _box(ax, (5.8, 3.6), 2.2, 0.9, "Chunk B")
    _arrow(ax, (2.6, 4.05), (3.1, 4.05))
    _arrow(ax, (2.6, 3.75), (5.8, 3.75), color="#999999")
    ax.text(4.5, 2.8, "Similarity in text/vector space", ha="center", fontsize=11, color="#444444")

    _box(ax, (8.2, 4.0), 1.8, 0.9, "Chunk A")
    _box(ax, (10.6, 4.0), 1.8, 0.9, "Requirement")
    _box(ax, (10.6, 2.6), 1.8, 0.9, "Control")
    _box(ax, (13.0, 3.3), 1.8, 0.9, "Chunk B")
    _arrow(ax, (10.0, 4.45), (10.6, 4.45))
    _arrow(ax, (11.5, 4.0), (11.5, 3.5))
    _arrow(ax, (12.4, 4.45), (13.0, 3.95))
    _arrow(ax, (12.4, 3.05), (13.0, 3.55))
    ax.text(11.5, 1.6, "Closeness through graph paths\nand typed relationships", ha="center", fontsize=11, color="#444444")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_example_subgraph(save_path: Path) -> None:
    _ensure_matplotlib()
    config = GKNConfig()
    ingestor = DocumentIngestor()
    documents = ingestor.load_text_documents(config.sample_docs_dir)
    chunks = ingestor.chunk_documents(documents, config.chunk_size, config.chunk_overlap)
    vector_store = SimpleVectorStore()
    vector_store.build(chunks)
    extractor = ConceptExtractor(config.concept_keywords)
    graph = KnowledgeNetworkBuilder(extractor).build(documents, chunks)
    hybrid = HybridRetriever(vector_store, graph)
    results = hybrid.search("What evidence is required for validation approval?", top_k=config.top_k, graph_hops=config.graph_hops)
    if results:
        draw_subgraph(graph, results[0].chunk_id, radius=2, save_path=save_path)


if __name__ == "__main__":
    generate_architecture_overview(DOCS_IMAGE_DIR / "gkn_architecture_overview.png")
    generate_baseline_vs_hybrid(DOCS_IMAGE_DIR / "baseline_vs_hybrid.png")
    generate_math_intuition(DOCS_IMAGE_DIR / "math_semantic_vs_structural.png")
    generate_example_subgraph(DOCS_IMAGE_DIR / "example_subgraph.png")
    print(f"Saved visuals to: {DOCS_IMAGE_DIR}")


