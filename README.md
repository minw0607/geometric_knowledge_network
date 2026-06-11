# Geometric Knowledge Network

> A modular, local-first framework for enhancing RAG retrieval with a document-grounded geometric knowledge network.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository explores a practical question at the intersection of **RAG**, **knowledge graphs**, and **AI evaluation**:

> **Can retrieval quality, traceability, and multi-hop grounding improve when a vector store is augmented with a lightweight geometric knowledge network?**

The answer pursued here is not to replace embeddings, but to **complement** them.

Traditional vector-store retrieval is very effective at semantic similarity. However, many real document-grounded workflows require more than semantic closeness. They also require:

- explicit support paths from source text to answer
- mapping between requirements, controls, evidence, and incidents
- multi-hop retrieval over related concepts
- visibility into why a chunk was surfaced
- a structured layer for evaluation and governance

This repository implements a **Geometric Knowledge Network (GKN)** as a lightweight enhancement layer over baseline RAG retrieval.

---

## Why this work matters

In many RAG systems, vector retrieval is treated as the primary retrieval substrate. That works well when the question is mainly:

- “what text is semantically similar to this query?”

But many high-value enterprise and governance questions are different:

- what evidence supports this statement?
- which control maps to this requirement?
- what related policy clause should be surfaced even if it is not the nearest chunk in embedding space?
- what happens downstream if monitoring identifies drift?
- can we inspect an evidence path rather than just a similarity score?

These are not only **similarity** problems. They are also **structure** problems.

That is the motivation for this repository.

---

## What is a Geometric Knowledge Network in this repo?

In this project, a **Geometric Knowledge Network** is a hybrid retrieval structure that combines:

1. **Embedding geometry**  
   Semantic neighborhoods among chunks using vector representations.

2. **Graph topology**  
   Explicit typed relationships across documents, chunks, requirements, controls, evidence, incidents, and concepts.

3. **Retrieval trajectories**  
   A query can move from vector hits to graph-linked neighbors to a more grounded evidence package.

This is aligned with the broader idea that useful AI systems often need both:
- a **geometric view** of similarity and neighborhood, and
- a **symbolic/structural view** of relationships and constraints.

For the MVP, “geometric” is implemented pragmatically through **embedding space + graph structure + retrieval paths**, not through advanced non-Euclidean KG embeddings or graph neural networks.

---

## How GKN differs from a traditional vector store in RAG

| Dimension | Traditional vector-store RAG | GKN-enhanced RAG |
|---|---|---|
| Core retrieval signal | Embedding similarity | Embedding similarity + graph structure |
| Best at | Semantically similar chunks | Similar chunks plus related evidence and linked context |
| Multi-hop support | Limited | Stronger through graph expansion |
| Explainability | Similarity score only | Inspectable nodes, edges, and evidence neighborhoods |
| Traceability | Often chunk-level only | Document -> chunk -> entity/type -> related chunk paths |
| Governance usefulness | Moderate | Higher for support mapping, evaluation, and review |
| Failure mode | Misses structurally relevant but semantically distant chunks | Can recover linked chunks through network expansion |

The key design idea is:

> **A vector store tells us what is close; a knowledge network helps tell us what is connected, dependent, supporting, or implicated.**

---

## Design principles

This repository follows a deliberately pragmatic architecture.

- **Local-first**: no enterprise infrastructure required for the MVP
- **Modular**: notebooks are orchestration layers; implementation lives in `src/`
- **Document-grounded**: graph nodes and edges remain tied to source text
- **Evaluation-first**: the repository is built to compare baseline and hybrid retrieval
- **Inspectable**: outputs, graph summaries, and reports should be saved locally
- **Incremental**: start with lightweight typed heuristics before adding heavier extraction methods

---

## Current MVP capabilities

The repository currently includes:

- sample governance / validation / control documents in `data/sample_docs/`
- document ingestion and chunking
- a baseline vector retriever
- heuristic typed extraction for:
  - requirements
  - controls
  - evidence
  - incidents
  - generic concepts
- a lightweight NetworkX-based knowledge network
- hybrid retrieval with graph-aware candidate expansion
- a benchmark query set in `data/eval_queries.json`
- a demo notebook for baseline vs hybrid exploration
- local artifact saving support for results, reports, figures, and graph summaries

---

## Architecture

```text
Raw Documents
   |
   v
Chunking
   |
   v
Vector Index -------------------------------> Baseline Retrieval
   |
   v
Typed Heuristic Extraction
   |
   v
Geometric Knowledge Network
(Document / Chunk / Requirement / Control / Evidence / Incident / Concept)
   |
   v
Hybrid Retrieval
(Vector Hits + Graph Expansion + Re-ranking)
   |
   v
Evaluation + Reports + Figures + Saved Artifacts
```

---

## Why the sample docs are synthetic

For the first demo, the repository uses a small synthetic-but-realistic corpus rather than a public benchmark-only dataset.

This choice is intentional.

Many public RAG datasets are optimized for answer correctness or retrieval relevance, but they do not always expose the kinds of relationships that make graph augmentation useful. The included documents were designed to create:

- cross-document references
- requirement-control linkages
- evidence-bearing statements
- monitoring and drift relationships
- incident escalation patterns
- multi-hop retrieval opportunities

That makes the repository better suited for demonstrating the value of GKN-enhanced retrieval.

---

## Repository layout

```text
geometric_knowledge_network/
  README.md
  LICENSE
  requirements.txt
  data/
    sample_docs/
    eval_queries.json
  notebooks/
    99_end_to_end_demo.ipynb
  src/geometric_knowledge_network/
    __init__.py
    config.py
    schema.py
    ingest.py
    extraction.py
    vector_store.py
    graph_builder.py
    hybrid_retriever.py
    evaluation.py
    visualization.py
    reporting.py
  tests/
    test_smoke.py
```

---

## Quick start

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Launch Jupyter

```powershell
jupyter notebook
```

### 3. Run the demo notebook

Open:

- `notebooks/99_end_to_end_demo.ipynb`

The notebook is intentionally code-light and delegates implementation to the Python package in `src/`.

---

## What the demo notebook is intended to show

The notebook walks through:

1. loading the sample corpus
2. chunking the documents
3. building a baseline vector index
4. building the knowledge network
5. running baseline retrieval
6. running graph-enhanced hybrid retrieval
7. comparing retrieved chunks
8. visualizing a local graph neighborhood
9. evaluating over a small benchmark query set
10. saving results and reports locally

---

## Local outputs and artifacts

The repository is designed to save outputs locally so results are inspectable and reproducible.

Typical artifact folders include:

- `artifacts/results/`
- `artifacts/reports/`
- `artifacts/figures/`
- `artifacts/graph/`

These can contain:

- retrieval result snapshots
- evaluation summaries
- graph summaries
- exported figures
- comparison tables

---

## Current limitations

This is an MVP and not yet a production system.

Current limitations include:

- heuristic extraction rather than high-accuracy structured extraction
- TF-IDF baseline retrieval rather than stronger embedding models
- lightweight graph semantics rather than deeply curated ontology design
- limited answer synthesis and citation validation
- small synthetic evaluation set
- no public benchmark adapter yet

These limitations are intentional for the current phase: the focus is to validate whether a lightweight knowledge network can produce measurable value over vector-only retrieval.

---

## Planned enhancement path

Near-term improvements:

- stronger chunking strategy
- richer typed schema and edge semantics
- improved graph expansion and re-ranking
- cleaner evaluation reports and artifact export
- robust notebook rendering and GitHub presentation

Longer-term options:

- local embedding model upgrades
- LLM-assisted extraction with provenance and confidence
- temporal graph support
- graph embeddings / knowledge graph embeddings
- Neo4j migration for queryability and visualization
- public benchmark adapters

---

## Conceptual grounding

This repository is inspired by a practical interpretation of several overlapping ideas:

- **GraphRAG**: graphs can improve retrieval and summarization over private corpora when questions require more than local similarity
- **Knowledge graph embeddings**: relations can be treated as structured geometric operations, not just text labels
- **Geometric deep learning**: structured and non-Euclidean representations matter when data has relational topology
- **Geometry discovery**: useful learning systems often discover structure, similarity, transformation, and path rather than only optimize raw matching

This repo takes those ideas and applies them in a pragmatic local MVP for document-grounded RAG.

---

## References and related resources

### Core conceptual and practical references

- Agus Sudjianto, *Learning as Geometry Discovery* (2026)  
  https://agussudjianto.substack.com/p/learning-as-geometry-discovery

- Agus Sudjianto, *What Learning Is Geometry Discovery Actually Means* (2026)  
  https://agussudjianto.substack.com/p/what-learning-is-geometry-discovery

- Microsoft Research, *GraphRAG: Unlocking LLM discovery on narrative private data* (2024)  
  https://www.microsoft.com/en-us/research/project/graphrag/

- Darren Edge et al., *A Graph RAG Approach to Query-Focused Summarization* (2024)  
  https://arxiv.org/abs/2404.16130

- Microsoft GraphRAG documentation  
  https://microsoft.github.io/graphrag/

- Neo4j, *LLM Knowledge Graph Builder*  
  https://neo4j.com/labs/genai-ecosystem/llm-graph-builder/

- Neo4j, *Generative AI - Ground LLMs with Knowledge Graphs*  
  https://neo4j.com/generativeai/

### Knowledge graph geometry and embeddings

- Chengjin Xu et al., *Knowledge Graph Embeddings in Geometric Algebras* (COLING 2020)  
  https://aclanthology.org/2020.coling-main.46/

- ACM Computing Surveys, *Knowledge Graph Embedding: A Survey from the Perspective of Representation Spaces*  
  https://dl.acm.org/doi/10.1145/3643806

- Yuhan Liu et al., *Unifying Geometry Knowledge Graph Embedding with Optimal Transport* (TheWebConf 2024)  
  https://openreview.net/forum?id=v9H7e0ShNN

### Broader theoretical background

- Michael M. Bronstein et al., *Geometric deep learning: going beyond Euclidean data* (2016)  
  https://arxiv.org/abs/1611.08097

### Ongoing literature tracking

- `zjukg/KG-LLM-Papers` repository  
  https://github.com/zjukg/KG-LLM-Papers

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Status

Early-stage local MVP under active development, focused on building a credible and inspectable comparison between baseline vector retrieval and GKN-enhanced retrieval.
