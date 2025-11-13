# AUTOMAP

Automatic mapping generation pipeline.

## TODO + IDEAS

- Añadir referencia del mapping en la evaluación. Poder ver la parte del mapping que ha generado las tripletas que han fallado.

## Table of Contents

- [Requirements](#requirements)
- [Running Experiments](#running-experiments)
- Pipeline
  - [Preprocess](#preprocess)
  - [Graph Evaluation](#graph-evaluation)
  - [Postprocess](#postprocess)

## Requirements

### System Requirements

- **Python**: 3.10.x (strictly)
- **Conda**: For environment and PyTorch GPU management
- **Poetry**: 2.2.1+ for Python dependency management
- **CUDA**: 12.1 (for GPU support)

### Python Dependencies

The project uses a hybrid dependency management approach:

- **Conda** (`environment.yml`): Manages PyTorch with GPU support
- **Poetry** (`pyproject.toml`): Manages all other Python packages

#### Core Dependencies

```python
# RDF and mapping tools
rdflib = ">=7.2.1,<8.0.0"       # RDF graph manipulation
yatter = ">=2.0.3,<3.0.0"        # YARRRML to RML conversion

# Data processing
pandas = ">=2.3.3,<3.0.0"        # Data manipulation
pyyaml = ">=6.0.3,<7.0.0"        # YAML configuration files
lxml = ">=6.0.2,<7.0.0"          # XML processing

# Machine Learning (managed by conda)
torch = "==2.5.1"                # PyTorch with CUDA support
torchaudio = "==2.5.1"           # Audio processing
torchvision = "==0.20.1"         # Vision processing
accelerate = ">=1.11.0,<2.0.0"   # Distributed training
transformers = ">=4.57.1,<5.0.0" # Transformer models
```

> **Important**: PyTorch is installed via Conda to ensure proper GPU support (CUDA 12.1). It's also declared in `pyproject.toml` to prevent Poetry from updating it to an incompatible CPU-only version.

### External Tools

- **RMLMapper**: Version 8.0.0-r378-all
  - Download: [RMLMapper v8.0.0](https://github.com/RMLio/rmlmapper-java/releases/download/v8.0.0/rmlmapper-8.0.0-r378-all.jar)
  - Location: Must be placed in `resources/rmlmapper-8.0.0-r378-all.jar`
  - Purpose: Executes RML mappings to generate RDF graphs

## Installation

### Local Installation (Standard)

1. **Create and activate the Conda environment**:
   ```bash
   # Create environment with Python 3.10
   conda env create -f environment.yml -p ./.venv
   
   # Activate the environment
   conda activate ./.venv
   ```

2. **Install Python dependencies with Poetry**:
   ```bash
   poetry install
   ```

3. **Download RMLMapper**:
   ```bash
   # Create resources directory if it doesn't exist
   mkdir -p resources
   
   # Download RMLMapper
   wget -O resources/rmlmapper-8.0.0-r378-all.jar \
     https://github.com/RMLio/rmlmapper-java/releases/download/v8.0.0/rmlmapper-8.0.0-r378-all.jar
   ```

### HPC Installation (Cesvima)

For HPC environments with GPU nodes:

1. **Install PyTorch with GPU support**:
   ```bash
   # Submit installation job to GPU node
   sbatch scripts/install.sh
   ```

2. **Install remaining dependencies**:
   ```bash
   # Activate the environment
   conda activate ./.venv
   
   poetry install
   ```

3. **Download RMLMapper** (same as local installation):
   ```bash
   mkdir -p resources
   wget -O resources/rmlmapper-8.0.0-r378-all.jar \
     https://github.com/RMLio/rmlmapper-java/releases/download/v8.0.0/rmlmapper-8.0.0-r378-all.jar
   ```

### Verification

To verify your installation:

```bash
# Check Python version
python --version  # Should show Python 3.10.x

# Check PyTorch GPU support
python scripts/test_torch.py

# Check Poetry packages
poetry show
```

## 🚀 Running Experiments

Bash scripts are provided to run complete experiments with a single command, automating the entire pipeline from mapping generation to evaluation.

A template is available at `resources/exp_bin_template.sh`. To use it:

1. Copy the template to your dataset's `bin/` directory
2. Configure only these variables:
   ```bash
   scenario="<your_scenario>/<your_subscenario>/<...>"
   exp="<your_experiment_name>"
   ```
3. Run the script - everything else is handled automatically

The script will create an experiment directory at `datasets/<dataset>/exps/<scenario>_<experiment_name>/` with the data files linked from `data/`.

## 🔧 Preprocess

blablabla

## 🕸️ Graph Evaluation

A Python module for evaluating RDF graphs against reference ontologies with comprehensive metrics.

> Modified version of the evaluation code from [kg-pipeline](https://github.com/Vehnem/kg-pipeline), used in the paper ["Towards self-configuring Knowledge Graph Construction Pipelines using LLMs - A Case Study with RML"](https://ceur-ws.org/Vol-3718/paper6.pdf)

### Overview

This project provides both a command-line tool and a Python API for comparing RDF graphs. It computes precision, recall, and F1-scores across multiple dimensions including triples, subjects, predicates, objects, and classes, with support for hierarchy-aware scoring.

### Usage

#### Command Line

Use the `compute_metrics.py` script to evaluate graphs from the command line:

```bash
python compute_metrics.py --config config.yaml --gold_graph gold.nt < predicted.nt
```

The script reads the predicted graph from stdin and outputs a comprehensive JSON object with evaluation results. The output includes multiple evaluation dimensions:

- **Basic metrics**: Precision, recall, and F1-scores for triples, subjects, classes, predicates, and objects
- **Unique element metrics**: Evaluation of unique classes, predicates, and property-object pairs
- **Datatype validation**: Property-datatype combinations for literals
- **Entity coverage**: Coverage of expected entities by type
- **Hierarchy-aware scores**: Semantic similarity using ontology hierarchies for classes and properties
- **Detailed predicate analysis**: Usage statistics and correctness metrics for each configured predicate

Each metric includes true positives (tp), false positives (fp), false negatives (fn), and computed scores, providing a complete picture of how well the predicted graph matches the reference graph.

#### Python API

Import and use the `grapheval` module directly in your code:

```python
from rdflib import Graph
from grapheval import GraphEvaluator
from grapheval.config import Config

# Load graphs
test_graph = Graph().parse('predicted.nt', format='nt')
reference_graph = Graph().parse('gold.nt', format='nt')

# Evaluate with configuration
config = Config('config.yaml')
evaluator = GraphEvaluator(test_graph, reference_graph, config)

# Get all metrics
results = evaluator.evaluate_all()
print(f"F1 Score: {results['triples']['f1']:.4f}")

# Or get just a summary
summary = evaluator.get_summary()
```

### Configuration

The evaluation requires a YAML configuration file that specifies the ontology and evaluation parameters. Example structure:

```yaml
# Path to the ontology file
ontology_file: "/path/to/ontology.ttl"

# Base IRI for resources
base_iri: "http://mykg.org/resource/"

# Namespaces for predicates
namespaces:
  "dbo": "http://dbpedia.org/ontology/"

# Predicates to evaluate (using namespace prefixes)
predicates_to_evaluate:
  "dbo":
    - "starring"
    - "director"
    - "title"

# Entity IDs by type for validation
ids_by_type:
  "http://dbpedia.org/ontology/Film":
    - "tt0167423"
  "http://dbpedia.org/ontology/Person":
    - "nm0000002"
```

## ⚙️ Postprocess

### `map2map`

`map2map` converts mapping files to the RML format. When provided with a
YARRRML file (``.yml``/``.yaml``) it relies on the
[YATTER](https://github.com/RMLio/yatter) CLI to do the conversion.  Existing
RML files are copied to the requested destination.

```
python -m src.map2map path/to/mapping.yml -o path/to/mapping.rml.ttl
```

The command prints the location of the generated file to standard output.
Set the ``YATTER_CMD`` environment variable or pass ``--yatter`` to point to a
custom executable when ``yatter`` is not on ``PATH``.

### `map2graph`

`map2graph` executes an RML mapping using the
[`RMLMapper`](https://github.com/RMLio/rmlmapper-java) CLI.  The path to the
mapper is read from the ``RMLMAPPER_JAR`` environment variable unless it is
provided via ``--rmlmapper``.  Use ``--ontology`` and ``--headers`` to provide
paths that can be consumed in the mapping via RMLMapper parameter placeholders
(``@{ontology}`` and ``@{headers}``).

```
python -m src.map2graph path/to/mapping.rml.ttl --ontology path/to/ontology.ttl \
    --headers path/to/headers.csv --rmlmapper /path/to/rmlmapper.jar
```

The generated triples are printed to standard output (one line per line in the
output file) and also written to disk. By default the file is named
``graph.ttl`` in the current working directory; supply ``--output`` to override
this behaviour. If the mapping fails or produces no triples the command exits
with a non-zero status code.


