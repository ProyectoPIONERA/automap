# Validation — Reproducing the Pipeline Comparison

This document explains how to reproduce the quality evaluation of the **Qwen 2.5 14B Instruct**-based KG pipeline against the results published by Hofer et al. (ESWC 2024).

---

## Overview

The evaluation script `compare_my_pipeline.py` is a **read-only extension** of the official evaluation harness from the Hofer et al. paper. It imports the original, unmodified metric class:

> [`kg_core/metrics/metrics.py → RML_Evaluation`](https://github.com/Vehnem/kg-pipeline/blob/main/kg_core/metrics/metrics.py)

No code in `metrics.py` was altered. The script adds only a side-by-side comparison table and a fuzzy subject-matching helper that is independent of the original metric logic.

---

## Reference Paper

**Hofer et al. (2024)** — *"Towards a Modular Approach for Constructing Knowledge Graphs with LLMs"*  
ESWC 2024 — Supplementary material and generated triples:  
🔗 https://akswnc7.informatik.uni-leipzig.de/~mhofer/paper_supplements/eswc24/kgc/

---

## Repository Layout (relevant to this validation)

```
your-repo/
├── data/
│   ├── input/
│   │   └── hofer-imdb/
│   │       ├── ontology.ttl                  # Adopted from Hofer et al.
│   │       └── diamond/
│   │           ├── diamonds-reference.nt     # Gold standard (from Hofer et al.)
│   │           └── movie_data.csv            # Hofer's JSON converted to flat CSV
│   │                                         # (our pipeline's input format)
│   └── output/
│       └── run_20260527_123946/
│           └── knowledge_graph.nt            # Output of our pipeline (Qwen 2.5 14B)
└── compare_my_pipeline.py                    # This validation script
```

The `kg-pipeline` repository (Hofer et al.) is a **separate dependency** cloned alongside your repo. See *Setup* below.

---

## What is Being Compared

| Pipeline | Model | Input format | KG file |
|---|---|---|---|
| Hofer et al. | GPT-4 (0125-preview) | JSON | [`gpt-4-0125-preview_101.ttl`](https://akswnc7.informatik.uni-leipzig.de/~mhofer/paper_supplements/eswc24/kgc/results/generated_triples.nt/gpt-4-0125-preview_101.ttl.nt) |
| Hofer et al. | Claude 3 Opus | JSON | [`claude-3-opus-20240229_12.ttl`](https://akswnc7.informatik.uni-leipzig.de/~mhofer/paper_supplements/eswc24/kgc/results/generated_triples.nt/claude-3-opus-20240229_12.ttl.nt) |
| **Ours** | **Qwen 2.5 14B Instruct** | **CSV** | `data/output/run_20260527_123946/knowledge_graph.nt` |

All three are evaluated against the **same gold standard**:  
`data/input/hofer-imdb/diamond/diamonds-reference.nt`  
(also available at: https://github.com/Vehnem/kg-pipeline/tree/main/experiments/llm4rml/diamonds-json)

> **Important note on input format:**  
> Hofer et al. use custom nested JSON as pipeline input. Our pipeline accepts flat CSV.  
> To enable a fair comparison on the *same data*, we converted Hofer's JSON to a flat CSV:  
> `data/input/hofer-imdb/diamond/movie_data.csv`  
> The ontology used is identical: `data/input/hofer-imdb/ontology.ttl`  
> (also at: https://github.com/Vehnem/kg-pipeline/blob/main/experiments/llm4rml/ontology.ttl)

---

## Setup

### 1. Clone the Hofer et al. evaluation framework

```bash
git clone https://github.com/Vehnem/kg-pipeline.git
cd kg-pipeline
```

The commit used for this evaluation:

```
git checkout eb3a211d07a1a710f31294f353b780e794cedad9
```

### 2. Install dependencies

Using **Poetry** (recommended, matches the original repo):

```bash
poetry install
poetry shell
```

Or with pip:

```bash
pip install rdflib scikit-learn
```

> `torch` is **not required** for this evaluation. If you run out of disk space during `poetry install`, you can skip torch — the comparison script only needs `rdflib` and `scikit-learn`.

### 3. Download the Hofer et al. generated triples

Place them inside the `kg-pipeline` repo under `experiments/llm4rml/diamonds-json/`:

```bash
# GPT-4
curl -o experiments/llm4rml/diamonds-json/gpt-4-0125-preview_101.ttl \
  "https://akswnc7.informatik.uni-leipzig.de/~mhofer/paper_supplements/eswc24/kgc/results/generated_triples/gpt-4-0125-preview_101.ttl"

# Claude 3 Opus
curl -o experiments/llm4rml/diamonds-json/claude-3-opus-20240229_12.ttl \
  "https://akswnc7.informatik.uni-leipzig.de/~mhofer/paper_supplements/eswc24/kgc/results/generated_triples/claude-3-opus-20240229_12.ttl"
```

### 4. Place your KG output

Copy or symlink your pipeline output into the expected location:

```bash
# Option A — copy
cp data/output/run_20260527_123946/knowledge_graph.nt \
   /path/to/kg-pipeline/target/your_knowledge_graph.nt

# Option B — edit the path directly in compare_my_pipeline.py
YOUR_KG = "data/output/run_20260527_123946/knowledge_graph.nt"
```

### 5. Copy the validation script

```bash
cp compare_my_pipeline.py /path/to/kg-pipeline/compare_my_pipeline.py
```

---

## Running the Validation

From the **root of the `kg-pipeline` repo**:

```bash
python compare_my_pipeline.py
```

Expected output structure:

```
===========================================================================
  SIDE-BY-SIDE COMPARISON  (F1 scores)
===========================================================================
  Metric                               Research      Yours   Delta
---------------------------------------------------------------------------
  Exact triple match                          0          0
  Subject URIs (exact)                        0          0
  Subject IDs (fuzzy, IRI-agnostic)       1.000      1.000  (+0.000)
  Class assignments (rdf:type)            0.846      0.846  (+0.000)
  Unique classes used                     0.800      0.800  (+0.000)
  Predicate usage                         0.667      0.655  (-0.011)
  Unique predicates used                  0.693      0.516  (-0.177)
  Literal values                          0.833      0.648  (-0.185)
  URI object values                       0.489      0.489  (+0.000)
---------------------------------------------------------------------------
  AVERAGE F1                              0.592      0.550  (-0.042)
===========================================================================

  ADDITIONAL STATS
---------------------------------------------------------------------------
  All 10 persons present (1=yes)              1          1
  All 4 actors present  (1=yes)               1          1
  Person entity count                        20         10
  Actor entity count                          8          4
  Persons typed as dbo:Person                10         10
  Actors typed as dbo:Actor                   0          0
  All ref predicates mapped (1=yes)           0          0
  Only ref predicates used (1=yes)            0          0
===========================================================================

  Research pipeline total triples : 56
  Your pipeline total triples     : 67
  Gold standard triples           : 49

✓ Full results saved to target/comparison_results.json
```

Full numeric results are written to `target/comparison_results.json`.

---

## URI Conventions

| Graph | Base IRI |
|---|---|
| Gold standard (`diamonds-reference.nt`) | `http://dbpedia.org/ontology/` |
| Hofer et al. generated triples | `http://mykg.org/resource/` |
| Our pipeline output | `http://mykg.org/resource/` |

The metrics in `RML_Evaluation.additionalStats()` check specifically for the `http://mykg.org/resource/` prefix to count person/actor entity coverage. Your pipeline **must** use this base IRI for those counts to be valid.

---

## Metrics Explained

All metrics are computed by `RML_Evaluation` from the original `metrics.py`:

| Metric | What it measures |
|---|---|
| **Exact triple match** | Full `(subject, predicate, object)` string equality |
| **Subject URIs (exact)** | IRI equality of subject nodes |
| **Subject IDs (fuzzy)** | IRI-agnostic: matches only the `nm*/tt*` token, ignoring base IRI differences |
| **Class assignments** | All `rdf:type` triples (multiset) |
| **Unique classes** | Set of distinct `rdf:type` object values used |
| **Predicate usage** | All predicate occurrences (multiset) |
| **Unique predicates** | Set of distinct predicates used |
| **Literal values** | All `rdf:Literal` object values |
| **URI object values** | All `rdf:Resource` (URI) object values |
| **Additional stats** | Coverage of the 10 IMDB person IDs and 4 actor IDs, typed correctly |

All scores are reported as **Precision / Recall / F1**.

---

## Citing

If you use this validation approach, please cite the original paper:

```bibtex
@inproceedings{hofer2024llm4kgc,
  title     = {Towards a Modular Approach for Constructing Knowledge Graphs with LLMs},
  author    = {Hofer, Markus and others},
  booktitle = {ESWC 2024},
  year      = {2024}
}
```

