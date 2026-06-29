# Geometric Knowledge Network (GKN)

> A pragmatic, local-first framework for **document-grounded reasoning, evidence traceability, and AI evaluation** — built by augmenting vector retrieval with a lightweight, inspectable knowledge graph.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Core idea:** a vector store tells us what is *semantically close*; a geometric knowledge network adds what is *structurally connected* — supporting, dependent, mapped, or implicated — and makes the evidence path visible.

---

## 🌐 Working thesis

A full enterprise-scale knowledge graph is expensive to build and hard to govern. This project takes the pragmatic middle path:

> **raw documents + vector embeddings + a lightweight knowledge graph + traces + LLM orchestration.**

This gives much of the benefit of graph-geometric reasoning — multi-hop retrieval, explicit evidence paths, structural relevance — **without** requiring a universal world model, a graph database, or graph neural networks up front. The graph is treated as an *indexed, reviewable layer over the source documents*, not as the ultimate source of truth.

The framework is being built **general-purpose first, then specialized per use case** (see [Use cases](#-use-cases)). The two reference adapters today are **AI governance / model-risk documents** and the **HotpotQA** multi-hop QA benchmark.

> ⚠️ **Status & scope:** this is an early-stage, evolving research MVP. The architecture and schema below are an *initial plan* and are expected to change as experiments accumulate. Treat numbers and design choices as provisional.

---

## 📚 Table of contents

- [Why this work matters](#why-this-work-matters)
- [What is a Geometric Knowledge Network here?](#what-is-a-geometric-knowledge-network-here)
- [GKN vs. a traditional vector store](#gkn-vs-a-traditional-vector-store)
- [Architecture](#-architecture)
- [Graph schema (initial)](#-graph-schema-initial)
- [Theoretical formulation](#-theoretical-formulation)
- [Use cases](#-use-cases)
- [Repository layout](#-repository-layout)
- [Quick start](#-quick-start)
- [The HotpotQA benchmark (and a note on fairness)](#-the-hotpotqa-benchmark-and-a-note-on-fairness)
- [Current status and roadmap](#-current-status-and-roadmap)
- [Conceptual grounding](#-conceptual-grounding)
- [References](#-references)
- [License](#-license)

---

## Why this work matters

In most RAG systems, vector retrieval is the primary substrate. That works well when the question is essentially *"what text is semantically similar to this query?"*

But many high-value governance, audit, and evaluation questions are **structure** problems, not just **similarity** problems:

- what evidence *supports* this claim?
- which control *maps to* this requirement?
- what related clause should surface even if it is not the nearest chunk in embedding space?
- what happens *downstream* if monitoring detects drift?
- did an agent follow a *valid* reasoning/tool path?
- can we *inspect an evidence path* rather than just trust a similarity score?

A geometric knowledge network is one practical infrastructure pattern for making the structure of a task explicit, so these questions become answerable and auditable.

---

## What is a Geometric Knowledge Network here?

In this project a GKN is a **hybrid representation** with four conceptual layers (only the first two are implemented in the MVP):

| Layer | What it represents | Example | MVP status |
|---|---|---|---|
| **Symbolic graph** | Entities and typed relationships | `Requirement —MAPS_TO→ Control` | ✅ implemented (NetworkX) |
| **Geometric layer** | Embeddings, distances, neighborhoods | similar requirements cluster together | ✅ implemented (FAISS / TF-IDF) |
| **Constraint layer** | Rules and directional dependencies | *a finding must be supported by evidence* | 🔜 planned |
| **Dynamic layer** | Sequences, actions, agent traces | `prompt → retrieval → tool → answer → eval` | 🔜 planned |

"Geometric" here is interpreted pragmatically: **embedding space + graph structure + retrieval paths** — not (yet) non-Euclidean KG embeddings or graph neural networks. Those are explicit later-phase options, not MVP requirements.

---

## GKN vs. a traditional vector store

<p align="center">
  <img src="docs/images/baseline_vs_hybrid.png" alt="Baseline vs hybrid retrieval" width="900">
</p>

| Dimension | Vector-store RAG | GKN-enhanced RAG |
|---|---|---|
| Core retrieval signal | Embedding similarity | Embedding similarity + graph structure |
| Best at | Semantically similar chunks | Similar chunks **plus** linked/related context |
| Multi-hop support | Limited | Stronger via graph expansion |
| Explainability | Similarity score only | Inspectable nodes, edges, and evidence paths |
| Traceability | Often chunk-level only | Document → chunk → entity → related chunk paths |
| Failure mode | Misses structurally relevant but distant chunks | Can recover linked chunks through the network |

> **Design distinction:** a vector store optimizes neighborhood in embedding space; GKN adds explicit relational structure that can surface relevant chunks even when they are not the nearest semantic match.

---

## 🏗️ Architecture

The recommended stack keeps source documents as the source of truth and layers structure on top:

```text
Raw Documents
   │
   ▼
Chunking (stable evidence units + metadata)
   │
   ├─────────────► Vector Index ───────► Baseline Retrieval
   ▼
Typed Entity / Relation Extraction (heuristic now; LLM-assisted later)
   │
   ▼
Geometric Knowledge Network  (NetworkX; Neo4j later)
   │
   ▼
Hybrid Retrieval  (vector hits + graph expansion + re-ranking)
   │
   ▼
Evaluation + Path Explanations + Reports + Figures + Saved Artifacts
```

**MVP component choices:** parsing (`pypdf`, `python-docx`), embeddings (OpenAI/Azure or local `sentence-transformers`), vector store (FAISS; TF-IDF fallback), graph store (NetworkX), orchestration (plain Python), evaluation (pandas + rule checks; LLM-as-judge later).

---

## 🧩 Graph schema (initial)

The schema is intentionally **small** in Phase 1 — broad schemas create extraction noise and governance burden. The reference (governance) schema:

**Node types:** `Document`, `Chunk`, `Requirement`, `Control`, `Evidence`, `Incident`/`Finding`, `Concept` (with `Model/AISystem`, `ValidationStep`, `AgentAction` planned).

**Relationship types:** `CONTAINS`, `MENTIONS`, `REQUIRES`, `MAPS_TO`, `SUPPORTS`, `CONTRADICTS`, `DEPENDS_ON`, `CITES` (and `GENERATED_BY`, `PRECEDES` for traces).

**Extraction principles (target):** every extracted node/edge carries `source_chunk_id`, a confidence score, and provenance; no relation is inferred unless explicitly supported by the source chunk. (The current MVP uses lightweight heuristic extraction; LLM-assisted extraction with provenance is a planned upgrade.)

Use-case adapters (e.g. HotpotQA) define their **own** node/edge vocabulary — for HotpotQA: `Document`, `Chunk`, `TitleEntity`, `NamedEntity`, `Concept`, and corpus-derived `SHARES_ENTITY` bridge edges.

---

## 🔢 Theoretical formulation

At a high level, hybrid retrieval combines a semantic score and a graph-derived bonus:

```text
s_hyb(q, c) = s_vec(q, c) + b(c)
```

- `s_vec(q, c)` — semantic similarity between query and chunk
- `b(c)` — a bonus from the chunk's typed structural neighborhood and graph expansion

<p align="center">
  <img src="docs/images/math_semantic_vs_structural.png" alt="Semantic vs structural closeness" width="850">
</p>

A fuller treatment (graph construction, structural closeness, worked examples) is in
[docs/mathematical_formulation.md](docs/mathematical_formulation.md).

**Example multi-hop flow:** baseline retrieval finds a chunk stating *"annual review is required"*; the graph links that requirement to `Control C-101`; another chunk mentioning `C-101` is pulled in via graph expansion even though it was not a top semantic match:

```text
Chunk A → Requirement → Control C-101 → Chunk B
```

---

## 🎯 Use cases

The framework is built around concrete use cases rather than a generic "world graph":

| Use case | Example question | Where graph structure helps |
|---|---|---|
| **AI governance / model risk** (flagship) | *What does the policy require for validation evidence?* | requirement ↔ control ↔ evidence mapping, traceable support paths |
| **HotpotQA multi-hop QA** (benchmark) | *Which film directed by X is a horror film?* | bridging across articles via shared entities |
| Evidence traceability | *Which source supports this conclusion?* | document → chunk → requirement → answer path |
| Hallucination / support checking | *Is this statement supported by the corpus?* | compare claims to evidence nodes and citation paths |
| Agent-trace evaluation *(planned)* | *Did the agent use the correct tool sequence?* | execution represented as a graph trajectory |

---

## 🗂️ Repository layout

```text
geometric_knowledge_network/
  README.md
  LICENSE
  requirements.txt
  data/
    sample_docs/            # synthetic governance corpus
    eval_queries.json       # governance benchmark queries
    hotpot_train_v1.1.json  # HotpotQA (downloaded on demand; git-ignored)
  notebooks/
    99_end_to_end_demo.ipynb                   # governance demo
    11_hotpotqa_knowledge_network_demo.ipynb   # HotpotQA demo (fair benchmark)
    10_hotpotqa_gkn_benchmark.ipynb            # compact HotpotQA benchmark
  src/geometric_knowledge_network/
    config.py               # env-driven configuration
    schema.py               # node/edge type enums
    ingest.py               # parsing + chunking
    extraction.py           # governance heuristic extraction
    vector_store.py         # TF-IDF baseline + embedding store (FAISS, cached)
    graph_builder.py        # governance knowledge network
    hotpotqa_loader.py      # HotpotQA load + auto-download/cache
    hotpotqa_graph.py       # HotpotQA graph (corpus-only, leakage-free)
    hotpotqa_relevance.py   # supporting-title relevance (eval only)
    hotpotqa_benchmark.py   # baseline vs hybrid runner
    hybrid_retriever.py     # vector + graph expansion + re-ranking
    path_explainer.py       # graph path explanations
    evaluation.py           # hit-rate / recall@k / precision@k / MRR
    visualization.py, reporting.py
  tests/
    test_smoke.py           # governance pipeline
    test_hotpotqa.py        # HotpotQA pipeline + leakage guard
```

---

## 🚀 Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` for the HotpotQA / embedding settings (all optional; sensible defaults apply):

```ini
# Embeddings: "local" (sentence-transformers), "small", or "large" (OpenAI/Azure)
EMBEDDING_CHOICE=local
# OPENAI_API_KEY=...                # only for cloud embeddings
# OPENAI_BASE_URL=...               # optional (Azure / gateway)
HOTPOTQA_SAMPLE_SIZE=25
HOTPOTQA_RANDOM_SEED=42
FORCE_REBUILD_VECTOR_STORE=false
```

Then launch Jupyter and open a demo:

```bash
jupyter notebook
```

- **Governance demo:** `notebooks/99_end_to_end_demo.ipynb`
- **HotpotQA demo:** `notebooks/11_hotpotqa_knowledge_network_demo.ipynb`

The notebooks are intentionally code-light; implementation lives in `src/`.

**Run the tests** (dependency-light — they use the TF-IDF path, so no faiss/openai/dataset needed):

```bash
PYTHONPATH=src python -m pytest tests/        # requires: pip install pytest
```

---

## 🧪 The HotpotQA benchmark (and a note on fairness)

[HotpotQA](https://hotpotqa.github.io/) is a multi-hop QA dataset where answering requires bridging across multiple articles — a natural fit for testing graph-augmented retrieval.

**Automatic data handling.** The dataset is **downloaded and cached on first use** (default `data/hotpot_train_v1.1.json`, ~570 MB, git-ignored). The loader tries the official CMU mirror first and **falls back to the Hugging Face mirror** automatically if CMU is unreachable, converting it to the canonical HotpotQA schema. Subsequent runs load the local copy directly.

**Leakage-free comparison.** The retrieval graph is built from **observable corpus text only**. The gold supporting facts are used *exclusively* to define evaluation relevance — they are **never** encoded into the graph the retriever uses. (An earlier version added `SupportingFact` nodes / `SUPPORTS` edges from the gold labels and rewarded the retriever for landing near them, which leaks the answer key.) As a result the baseline-vs-hybrid numbers are honest, and the hybrid retriever must *earn* any gain rather than inherit it from leaked labels. Expect the hybrid to sometimes **tie or trail** the baseline today; that is the signal that drives the next round of graph-scoring improvements.

---

## ⚠️ Current status and roadmap

This is an MVP focused on a single question: **does a lightweight knowledge network measurably improve retrieval, traceability, and evaluation over vector-only RAG?**

**Current limitations**
- heuristic extraction (not yet LLM-assisted with provenance/confidence)
- title-level (not sentence-level) HotpotQA relevance
- graph scoring is simple; expansion does not yet reliably beat the baseline
- no constraint layer, agent-trace layer, or answer synthesis / citation validation yet

**Near-term**
- generalize into a reusable **GKN core** + thin **use-case adapters** (HotpotQA, governance, …)
- sentence-level relevance + more query-aware graph scoring
- LLM-assisted extraction with `source_chunk_id` + confidence + prompt-version provenance
- cleaner evaluation reports and artifact export

**Longer-term options** *(only if the lightweight graph proves its value first)*
- Neo4j migration (Cypher, visualization, scale)
- graph / knowledge-graph embeddings; non-Euclidean (hyperbolic) embeddings for hierarchy
- temporal and causal graph layers
- graph neural networks; interactive Streamlit/Gradio review UI; CI/CD evaluation

---

## 🧠 Conceptual grounding

- **GraphRAG** — graphs improve retrieval/summarization over private corpora when questions need more than local similarity
- **Knowledge graph embeddings** — relations as geometric operations, not just text labels
- **Geometric deep learning** — structured / non-Euclidean representations matter when data has relational topology
- **Geometry discovery** (Agus Sudjianto) — learning systems succeed by discovering structure, similarity, direction, and path, not only by raw matching

---

## 📖 References

- Agus Sudjianto, *Learning as Geometry Discovery* (2026) — https://agussudjianto.substack.com/p/learning-as-geometry-discovery
- Agus Sudjianto, *What Learning Is Geometry Discovery Actually Means* (2026) — https://agussudjianto.substack.com/p/what-learning-is-geometry-discovery
- Microsoft Research, *GraphRAG* (2024) — https://www.microsoft.com/en-us/research/project/graphrag/
- Darren Edge et al., *A Graph RAG Approach to Query-Focused Summarization* (2024) — https://arxiv.org/abs/2404.16130
- Neo4j, *LLM Knowledge Graph Builder* — https://neo4j.com/labs/genai-ecosystem/llm-graph-builder/
- Chengjin Xu et al., *Knowledge Graph Embeddings in Geometric Algebras* (COLING 2020) — https://aclanthology.org/2020.coling-main.46/
- ACM Computing Surveys, *Knowledge Graph Embedding: A Survey from the Perspective of Representation Spaces* — https://dl.acm.org/doi/10.1145/3643806
- Yuhan Liu et al., *Unifying Geometry KG Embedding with Optimal Transport* (TheWebConf 2024) — https://openreview.net/forum?id=v9H7e0ShNN
- Michael M. Bronstein et al., *Geometric deep learning: going beyond Euclidean data* (2016) — https://arxiv.org/abs/1611.08097
- HotpotQA — https://hotpotqa.github.io/
- `zjukg/KG-LLM-Papers` — https://github.com/zjukg/KG-LLM-Papers

---

## 📄 License

MIT — see [LICENSE](LICENSE).

> **Status:** early-stage local research MVP. The plan and schema here are initial and will be updated, enhanced, and adjusted as research progresses.
