<div align="center">

# 🌐 Geometric Knowledge Network

**Augment vector retrieval with a lightweight, inspectable knowledge graph — for document-grounded reasoning, evidence traceability, and AI evaluation.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Early%20MVP-orange.svg)]()
[![Graph: NetworkX](https://img.shields.io/badge/Graph-NetworkX-2C3E50.svg)](https://networkx.org/)
[![Vectors: FAISS](https://img.shields.io/badge/Vectors-FAISS-005571.svg)](https://faiss.ai/)
[![Benchmark: HotpotQA](https://img.shields.io/badge/Benchmark-HotpotQA-412991.svg)](https://hotpotqa.github.io/)

*hybrid retrieval · multi-hop graph expansion · typed evidence paths · governance & HotpotQA adapters · local-first*

</div>

---

> **Core idea:** a vector store tells us what is *semantically close*; a geometric knowledge network adds what is *structurally connected* — supporting, dependent, mapped, or implicated — and makes the evidence path **visible**.

---

## 📖 Overview

A full enterprise-scale knowledge graph is expensive to build and hard to govern. This project takes the pragmatic middle path:

> **raw documents + vector embeddings + a lightweight knowledge graph + traces + LLM orchestration.**

This delivers much of the value of graph-geometric reasoning — multi-hop retrieval, explicit evidence paths, structural relevance — **without** a universal world model, a graph database, or graph neural networks up front. The graph is an *indexed, reviewable layer over the source documents*, not the ultimate source of truth.

The framework is built **general-purpose first, then specialized per use case**. The two reference adapters today are **AI governance / model-risk documents** and the **HotpotQA** multi-hop QA benchmark.

> ⚠️ **Status & scope:** early-stage, evolving research MVP. The architecture and schema below are an *initial plan* and will change as experiments accumulate. Treat design choices and numbers as provisional.

---

## 🧭 At a Glance

```
┌──────────────┬─────────────────┬─────────────────────┬──────────────────────┬───────────────────┐
│  DOCUMENTS   │   VECTOR INDEX  │   KNOWLEDGE GRAPH   │   HYBRID RETRIEVAL   │    EVALUATION     │
│              │                 │                     │                      │                   │
│ parse +      │ FAISS / TF-IDF  │ NetworkX, typed     │ vector hits          │ hit-rate, recall  │
│ chunk into   │ cached          │ entities & edges,   │ + graph expansion    │ precision, MRR    │
│ stable units │ embeddings,     │ corpus-grounded,    │ + re-ranking         │ + inspectable     │
│ + metadata   │ semantic search │ leakage-free        │                      │ evidence paths    │
└──────────────┴─────────────────┴─────────────────────┴──────────────────────┴───────────────────┘
```

---

## 🤔 Why This Matters

In most RAG systems, vector retrieval is the primary substrate — great when the question is essentially *"what text is semantically similar to this query?"*

But many high-value governance, audit, and evaluation questions are **structure** problems, not just **similarity** problems:

| Question | Vector-only | GKN-enhanced |
|---|:---:|:---:|
| What is **similar** to this query? | ✅ Strong | ✅ Strong |
| What **supports** this claim? | ⚠️ Weak | ✅ via `SUPPORTS` / `CITES` paths |
| What **maps to** this requirement? | ⚠️ Weak | ✅ via `MAPS_TO` edges |
| What is the **multi-hop** answer across articles? | ⚠️ Limited | ✅ via graph expansion |
| **Why** was this chunk surfaced? | ❌ Score only | ✅ inspectable node/edge path |

> **Design distinction:** a vector store optimizes neighborhood in *embedding space*; GKN adds explicit relational *structure* that can surface relevant chunks even when they are not the nearest semantic match.

<p align="center">
  <img src="docs/images/baseline_vs_hybrid.png" alt="Baseline vs hybrid retrieval" width="880">
</p>

---

## 🧱 What Is a Geometric Knowledge Network Here?

A GKN is a **hybrid representation** with four conceptual layers — only the first two are implemented in the MVP:

| Layer | What it represents | Example | Status |
|---|---|---|:---:|
| **Symbolic graph** | Entities and typed relationships | `Requirement —MAPS_TO→ Control` | ✅ |
| **Geometric layer** | Embeddings, distances, neighborhoods | similar requirements cluster | ✅ |
| **Constraint layer** | Rules and directional dependencies | *a finding must be supported by evidence* | 🔜 |
| **Dynamic layer** | Sequences, actions, agent traces | `prompt → retrieval → tool → answer → eval` | 🔜 |

"Geometric" is interpreted pragmatically — **embedding space + graph structure + retrieval paths** — not (yet) non-Euclidean KG embeddings or graph neural networks. Those are explicit later-phase options, not MVP requirements.

---

## 🏗️ Architecture

```text
Raw Documents
   │
   ▼
Chunking ─ stable evidence units + metadata
   │
   ├──────────────► Vector Index ──────────► Baseline Retrieval
   ▼
Typed Entity / Relation Extraction ─ heuristic now; LLM-assisted later
   │
   ▼
Geometric Knowledge Network ─ NetworkX (Neo4j later)
   │
   ▼
Hybrid Retrieval ─ vector hits + graph expansion + re-ranking
   │
   ▼
Evaluation + Path Explanations + Reports + Figures + Saved Artifacts
```

**MVP component choices:** parsing (`pypdf`, `python-docx`) · embeddings (OpenAI / Azure or local `sentence-transformers`) · vector store (FAISS, with a dependency-light TF-IDF fallback) · graph store (NetworkX) · orchestration (plain Python) · evaluation (pandas + rule checks; LLM-as-judge later).

---

## 🧩 Graph Schema (initial)

The schema is intentionally **small** in Phase 1 — broad schemas create extraction noise and governance burden.

**Reference (governance) node types**
`Document` · `Chunk` · `Requirement` · `Control` · `Evidence` · `Incident`/`Finding` · `Concept`
*(planned: `Model`/`AISystem`, `ValidationStep`, `AgentAction`)*

**Relationship types**
`CONTAINS` · `MENTIONS` · `REQUIRES` · `MAPS_TO` · `SUPPORTS` · `CONTRADICTS` · `DEPENDS_ON` · `CITES`
*(planned: `GENERATED_BY`, `PRECEDES` for traces)*

> **Extraction principle (target):** every node/edge carries `source_chunk_id`, a confidence score, and provenance; no relation is inferred unless explicitly supported by the source chunk. *(The MVP uses lightweight heuristic extraction; LLM-assisted extraction with provenance is a planned upgrade.)*

Use-case adapters define their own vocabulary — for **HotpotQA**: `Document` · `Chunk` · `TitleEntity` · `NamedEntity` · `Concept`, plus corpus-derived `SHARES_ENTITY` bridge edges.

---

## 🔢 Theoretical Formulation

Hybrid retrieval combines a semantic score and a graph-derived bonus:

```text
s_hyb(q, c) = s_vec(q, c) + b(c)
```

- `s_vec(q, c)` — semantic similarity between query and chunk
- `b(c)` — a bonus from the chunk's typed structural neighborhood and graph expansion

<p align="center">
  <img src="docs/images/math_semantic_vs_structural.png" alt="Semantic vs structural closeness" width="820">
</p>

**Example multi-hop flow** — baseline finds a chunk stating *"annual review is required"*; the graph links that requirement to `Control C-101`; a chunk mentioning `C-101` is then pulled in via expansion, even though it was not a top semantic match:

```text
Chunk A → Requirement → Control C-101 → Chunk B
```

📐 Full treatment (graph construction, structural closeness, worked examples): [docs/mathematical_formulation.md](docs/mathematical_formulation.md)

---

## 🎯 Use Cases

| Use case | Example question | Where structure helps | Status |
|---|---|---|:---:|
| **AI governance / model risk** *(flagship)* | *What does the policy require for validation evidence?* | requirement ↔ control ↔ evidence mapping | ✅ demo |
| **HotpotQA multi-hop QA** *(benchmark)* | *Which film directed by X is a horror film?* | bridging across articles via shared entities | ✅ demo |
| Evidence traceability | *Which source supports this conclusion?* | document → chunk → requirement → answer path | ✅ via paths |
| Hallucination / support checking | *Is this statement supported by the corpus?* | compare claims to evidence nodes & citation paths | 🔜 |
| Agent-trace evaluation | *Did the agent use the correct tool sequence?* | execution as a graph trajectory | 🔜 |

---

## 🗂️ Repository Structure

```text
geometric_knowledge_network/
│
├── data/
│   ├── sample_docs/            # synthetic governance corpus
│   ├── eval_queries.json       # governance benchmark queries
│   └── hotpot_train_v1.1.json  # HotpotQA — downloaded on demand   [git-ignored]
│
├── notebooks/
│   ├── 99_end_to_end_demo.ipynb                  # governance demo
│   └── 11_hotpotqa_knowledge_network_demo.ipynb  # HotpotQA demo (fair benchmark)
│
├── src/geometric_knowledge_network/
│   ├── config.py               # env-driven configuration
│   ├── schema.py               # node / edge type enums
│   ├── ingest.py               # parsing + chunking
│   ├── extraction.py           # governance heuristic extraction
│   ├── vector_store.py         # TF-IDF baseline + embedding store (FAISS, cached)
│   ├── graph_builder.py        # governance knowledge network
│   ├── hotpotqa_loader.py      # HotpotQA load + auto-download / cache
│   ├── hotpotqa_graph.py       # HotpotQA graph (corpus-only, leakage-free)
│   ├── hotpotqa_relevance.py   # supporting-title relevance (eval only)
│   ├── hotpotqa_benchmark.py   # baseline vs hybrid runner
│   ├── hybrid_retriever.py     # vector + graph expansion + re-ranking
│   ├── path_explainer.py       # graph path explanations
│   ├── evaluation.py           # hit-rate / recall@k / precision@k / MRR
│   └── visualization.py, reporting.py
│
└── tests/
    ├── test_smoke.py           # governance pipeline
    └── test_hotpotqa.py        # HotpotQA pipeline + leakage guard
```

---

## 🚀 Quick Start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` (all optional — sensible defaults apply):

```ini
# Embeddings: "local" (sentence-transformers), "small", or "large" (OpenAI/Azure)
EMBEDDING_CHOICE=local
# OPENAI_API_KEY=...                # only for cloud embeddings
# OPENAI_BASE_URL=...               # optional (Azure / gateway)
HOTPOTQA_SAMPLE_SIZE=25
HOTPOTQA_RANDOM_SEED=42
FORCE_REBUILD_VECTOR_STORE=false
```

Launch Jupyter and open a demo:

```bash
jupyter notebook
```

| Notebook | What it shows |
|---|---|
| `notebooks/99_end_to_end_demo.ipynb` | Governance corpus: baseline vs GKN-enhanced retrieval |
| `notebooks/11_hotpotqa_knowledge_network_demo.ipynb` | HotpotQA: fair, leakage-free benchmark with path explanations |

The notebooks are deliberately **code-light** — implementation lives in `src/`.

**Run the tests** *(dependency-light — TF-IDF path, no faiss/openai/dataset needed):*

```bash
pip install pytest
PYTHONPATH=src python -m pytest tests/
```

---

## 🧪 The HotpotQA Benchmark — and a Note on Fairness

[HotpotQA](https://hotpotqa.github.io/) is a multi-hop QA dataset where answering requires bridging across multiple articles — a natural fit for graph-augmented retrieval.

**🔄 Automatic data handling.** The dataset is **downloaded and cached on first use** (default `data/hotpot_train_v1.1.json`, ~570 MB, git-ignored). The loader tries the official CMU mirror first and **falls back to the Hugging Face mirror** automatically if CMU is unreachable, converting it to the canonical HotpotQA schema. Later runs load the local copy directly.

**🛡️ Leakage-free comparison.** The retrieval graph is built from **observable corpus text only**. Gold supporting facts are used *exclusively* to define evaluation relevance — they are **never** encoded into the graph the retriever uses.

> An earlier version added `SupportingFact` nodes / `SUPPORTS` edges from the gold labels and rewarded the retriever for landing near them — which leaks the answer key. With that removed, the baseline-vs-hybrid numbers are honest: the hybrid retriever must *earn* any gain. Expect it to sometimes **tie or trail** the baseline today — that is exactly the signal that drives the next round of graph-scoring improvements.

---

## 🗺️ Roadmap

This MVP exists to answer one question: **does a lightweight knowledge network measurably improve retrieval, traceability, and evaluation over vector-only RAG?**

**Current limitations**
- heuristic extraction (not yet LLM-assisted with provenance/confidence)
- title-level (not sentence-level) HotpotQA relevance
- simple graph scoring — expansion does not yet reliably beat the baseline
- no constraint layer, agent-trace layer, or answer synthesis / citation validation yet

**Near-term**
- 🔜 generalize into a reusable **GKN core** + thin **use-case adapters**
- 🔜 sentence-level relevance + more query-aware graph scoring
- 🔜 LLM-assisted extraction with `source_chunk_id` + confidence + prompt-version provenance
- 🔜 cleaner evaluation reports and artifact export

**Longer-term** *(only if the lightweight graph proves its value first)*
- Neo4j migration (Cypher, visualization, scale)
- graph / non-Euclidean (hyperbolic) embeddings for hierarchy
- temporal and causal graph layers
- graph neural networks · interactive review UI · CI/CD evaluation

---

## 🧠 Conceptual Grounding

- **GraphRAG** — graphs improve retrieval/summarization over private corpora when questions need more than local similarity
- **Knowledge graph embeddings** — relations as geometric operations, not just text labels
- **Geometric deep learning** — structured / non-Euclidean representations matter when data has relational topology
- **Geometry discovery** (Agus Sudjianto) — learning systems succeed by discovering structure, similarity, direction, and path, not only by raw matching

---

## 📚 References

- Agus Sudjianto — *Learning as Geometry Discovery* (2026) · [link](https://agussudjianto.substack.com/p/learning-as-geometry-discovery)
- Agus Sudjianto — *What Learning Is Geometry Discovery Actually Means* (2026) · [link](https://agussudjianto.substack.com/p/what-learning-is-geometry-discovery)
- Microsoft Research — *GraphRAG* (2024) · [link](https://www.microsoft.com/en-us/research/project/graphrag/)
- Darren Edge et al. — *A Graph RAG Approach to Query-Focused Summarization* (2024) · [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)
- Neo4j — *LLM Knowledge Graph Builder* · [link](https://neo4j.com/labs/genai-ecosystem/llm-graph-builder/)
- Chengjin Xu et al. — *Knowledge Graph Embeddings in Geometric Algebras* (COLING 2020) · [link](https://aclanthology.org/2020.coling-main.46/)
- ACM Computing Surveys — *KG Embedding: A Survey from the Perspective of Representation Spaces* · [link](https://dl.acm.org/doi/10.1145/3643806)
- Yuhan Liu et al. — *Unifying Geometry KG Embedding with Optimal Transport* (TheWebConf 2024) · [link](https://openreview.net/forum?id=v9H7e0ShNN)
- Michael M. Bronstein et al. — *Geometric Deep Learning* (2016) · [arXiv:1611.08097](https://arxiv.org/abs/1611.08097)
- HotpotQA · [hotpotqa.github.io](https://hotpotqa.github.io/) — `zjukg/KG-LLM-Papers` · [repo](https://github.com/zjukg/KG-LLM-Papers)

---

<div align="center">

**📄 License:** [MIT](LICENSE)  ·  **Status:** early-stage local research MVP

*The plan and schema here are initial — expect them to be updated, enhanced, and adjusted as research progresses.*

</div>
