# Geometric Knowledge Network

> A lightweight, document-grounded graph enhancement layer for Retrieval-Augmented Generation (RAG).

This repository explores how a **geometric knowledge network** can improve RAG retrieval beyond vector similarity alone by combining:

- **embedding geometry** for semantic similarity
- **graph topology** for explicit relationships among documents, chunks, and concepts
- **evidence paths** for retrieval traceability and multi-hop grounding

The project is intentionally designed as a **modular, local-first MVP**:
- notebooks stay thin
- implementation lives in reusable Python modules under `src/`
- sample documents are included for reproducible demos
- the main question is measurable: **does graph augmentation improve retrieval quality over vector-only RAG?**

---

## Why this repo exists

Vector stores are excellent at finding semantically similar chunks, but similarity alone is often not enough.

In many RAG workflows, we also care about:
- which evidence supports a claim
- which requirement maps to which control
- what related chunk should be surfaced through a dependency or concept link
- whether retrieval supports a multi-hop reasoning path
- how to inspect and explain why a chunk was retrieved

This repo builds a lightweight **knowledge network** over source documents to complement vector retrieval rather than replace it.

---

## What “geometric” means here

In this repository, **geometric** does not mean a full research implementation of non-Euclidean knowledge graph embeddings or graph neural networks.

For the MVP, it means combining three kinds of structure:

1. **Embedding space**  
   Chunks are represented in a semantic similarity space.

2. **Graph structure**  
   Documents, chunks, and extracted concepts are connected through explicit edges.

3. **Retrieval trajectories**  
   Queries can move from vector hits to graph neighbors to more grounded evidence packages.

This gives a practical, inspectable version of “geometry + structure” for RAG enhancement.

---

## Current MVP scope

The initial repository includes:

- sample governance / policy style documents in `data/sample_docs/`
- document ingestion and chunking
- a baseline vector retriever
- lightweight concept extraction
- a NetworkX-based knowledge network
- a hybrid retriever that applies graph-aware score boosting
- a lightweight end-to-end demo notebook

---

## Architecture at a glance

```text
Documents
   |
   v
Chunking
   |
   v
Vector Index ---------------------------> Baseline Retrieval
   |
   v
Concept Extraction
   |
   v
Knowledge Network (Document / Chunk / Concept graph)
   |
   v
Hybrid Retrieval (vector score + graph bonus + evidence context)
```

---

## Repository principles

- **Local-first**: no external infrastructure required for the MVP
- **Modular**: notebooks should orchestrate, not contain heavy logic
- **Document-grounded**: graph nodes and edges stay tied to source text
- **Evaluation-first**: baseline and hybrid retrieval should be compared directly
- **Pragmatic**: this is a lightweight graph enhancement, not a full enterprise ontology platform

---

## Why the sample docs are synthetic

For the first demo, the repository uses a small synthetic-but-realistic corpus instead of a public benchmark dataset.

That choice is intentional.

Many public RAG datasets are optimized for question answering, but they do not always expose the kinds of structure that make graph augmentation valuable. The included sample documents were chosen to create:

- cross-document references
- requirements and controls
- evidence-like statements
- monitoring / validation relationships
- multi-hop retrieval opportunities

This makes the repository better suited for demonstrating the value of graph-enhanced retrieval.

Public benchmark adapters can be added later.

---

## Repository layout

```text
geometric_knowledge_network/
  README.md
  requirements.txt
  data/
    sample_docs/
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

### 2. Open the demo notebook

Launch Jupyter and run:

- `notebooks/99_end_to_end_demo.ipynb`

The notebook is intentionally code-light and uses the Python package in `src/` for the actual implementation.

---

## Current workflow

1. Load sample documents
2. Chunk documents into retrievable units
3. Build a vector index over chunks
4. Build a lightweight knowledge network from extracted concepts
5. Run baseline vector retrieval
6. Run hybrid retrieval with graph-aware boosting
7. Inspect the returned chunks and graph neighborhood

---

## Current limitations

This is an MVP and not yet a production system.

Current limitations include:
- concept extraction is heuristic and intentionally simple
- retrieval uses TF-IDF rather than stronger embedding models
- graph relations are lightweight and not yet schema-rich
- answer generation and citation evaluation are still minimal
- no public benchmark integration yet

---

## Planned next steps

- richer typed schema such as Requirement / Control / Evidence
- optional embedding model upgrades
- stronger graph expansion and re-ranking logic
- benchmark queries and retrieval comparison metrics
- public RAG benchmark adapters
- answer grounding and citation evaluation
- improved visualizations and exported artifacts

---

## Project goal for the next milestone

The next major milestone is a **credible demo comparison** between:

- **baseline vector retrieval**, and
- **vector retrieval enhanced by the geometric knowledge network**

using a small evaluation set over the included sample documents.

---

## Status

Early-stage local MVP under active development.
