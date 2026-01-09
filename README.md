# AUTOMAP

**Automatic Knowledge Graph Mapping Generation Pipeline**

AUTOMAP is a comprehensive framework for generating and evaluating mappings automatically. It provides a complete pipeline from data preprocessing to RDF graph generation and evaluation, supporting multiple mapping formats (YARRRML, RML) and featuring advanced evaluation metrics.

---

## Project Status

🚧 **Active Development** - This project is actively maintained and under continuous development.

### Roadmap
- Add mapping reference traceability in evaluation
- Enhanced preprocessing capabilities
- Additional LLM-based mapping generation methods

---

## Table of Contents

- [Features](#-features)
- [Repository Structure](#-repository-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
  - [Running Experiments](#running-experiments)
  - [Preprocess](#preprocess)
  - [Converters](#converters)
  - [RDF Turtle Light](#rdf-turtle-light)
  - [Graph Evaluation](#graph-evaluation)
  - [Postprocess](#postprocess)
- [Acknowledgments](#-acknowledgments)
- [Authors and Contact](#-authors-and-contact)
- [License](#-license)

---

## Features

AUTOMAP provides a complete toolkit for mapping generation and evaluation:

### Core Capabilities
- **Automated Mapping Execution**: Execute RML mappings with RMLMapper integration
- **Ontology Simplification**: Generate minimal, readable Turtle representations from complex OWL/RDFS ontologies
- **Comprehensive Evaluation**: RDF graph evaluation with detailed metrics for in-depth quality analysis
- **GPU Acceleration**: Support for transformer models with CUDA
- **Flexible Pipeline**: Modular architecture for custom workflows

### Pipeline Components
1. **Preprocessing**: Data preparation and configuration
2. **Conversion**: YARRRML ↔ RML transformation
3. **Execution**: RML mapping to RDF graph generation
4. **Evaluation**: Precision, recall, F1-scores across multiple dimensions
5. **Postprocessing**: Results analysis and reporting

---

## Repository Structure

```
automap/
├── automap/                 # Main code package
│   ├── converters/            # Mapping format converters
│   ├── grapheval/             # Graph evaluation metrics
│   │   └── metrics/
│   ├── methods/               # Mapping generation methods
│   │   ├── examples/
│   │   ├── llm_base/
│   │   └── remap/
│   ├── postprocess/           # Results postprocessing (not used yet)
│   ├── preprocess/            # Data preprocessing  (not used yet)
│   ├── rdf_turtle_light/      # Ontology simplification
│   └── utils/                 # Utilities (config, auth, printers)
├── datasets/                # Experiment datasets
│   ├── <dataset_name>/        # Dataset directory
│   │   ├── bin/               # Experiment execution scripts
│   │   ├── data/              # Data organized by scenarios
│   │   │   ├── <scenario1>/   # Scenario 1: specific ontology
│   │   │   │   ├── config.yaml           # Evaluation configuration
│   │   │   │   ├── ontology.ttl          # Target ontology
│   │   │   │   ├── <case1>/              # Case 1: specific data instance
│   │   │   │   │   ├── data.*            # Source data (CSV, JSON, XML, etc.)
│   │   │   │   │   ├── gold_mapping.yml  # Reference mapping (optional)
│   │   │   │   │   └── gold_graph.nt     # Reference RDF graph
│   │   │   │   └── <case2>/              # Case 2: different data, same ontology
│   │   │   │       └── ...
│   │   │   └── <scenario2>/   # Scenario 2: different ontology
│   │   │       └── ...
│   │   └── exps/              # Generated experiment results
│   ├── blinkg/                # Example: BlinkG benchmark
│   ├── imbd/                  # Example: IMDB movies dataset
│   └── PODIO/                 # Example: PODIO dataset
├── resources/               # External resources
│   └── rmlmapper-*.jar        # RMLMapper executable
├── scripts/                 # Installation and job scripts
├── pyproject.toml           # Poetry dependencies
├── environment.yml          # Conda environment (PyTorch GPU)
└── README.md                # This file
```

### Dataset Organization

Datasets must be placed in the `datasets/` directory following this hierarchical structure:

- **Dataset level**: Top-level directory for a collection of related experiments (e.g., `blinkg`, `imbd`)
- **Scenario level** (`data/<scenario>/`): Each scenario represents a **different target ontology** with its own:
  - `ontology.ttl`: The target ontology file
  - `config.yaml`: Evaluation configuration specific to this ontology
- **Case level** (`data/<scenario>/<case>/`): Each case within a scenario represents a **different data instance** for the same ontology, containing:
  - Source data files (CSV, JSON, XML, etc.)
  - `gold_mapping.yml`: Reference mapping (optional, for mapping evaluation)
  - `gold_graph.nt`: Reference RDF graph (for graph evaluation)

This structure allows testing the same ontology (scenario) with different data sources (cases), or different ontologies (scenarios) within the same dataset collection.

---

## Requirements

### System Requirements

- **Python**: 3.10.x (strictly required)
- **Conda**: For environment and PyTorch GPU management
- **Poetry**: 2.2.1+ for Python dependency management
- **CUDA**: 12.1 (for GPU support)
- **Java**: JRE 11+ (for RMLMapper)

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

---

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

---

## Usage Guide

### Running Experiments

Bash scripts are provided to run complete experiments with a single command, automating the entire pipeline from mapping generation to evaluation.

**Dataset Requirements**: Datasets must be placed in the `automap/datasets/` directory following the structure described in [Repository Structure](#repository-structure), with scenarios organized under `data/` subdirectories (e.g., `datasets/<dataset_name>/data/scenario1/`, `datasets/<dataset_name>/data/scenario2/`, etc.).

#### Experiment Template

A template is available at `resources/exp_bin_template.sh`. To use it:

1. Copy the template to your dataset's `bin/` directory
2. Configure only these variables:
   ```bash
   # Line 3. Project path.
   project="$HOME/workspace/automap"

   # Line 39. Dataset path.
   dataset="$project/datasets/<YOUR_DATASET>"

   # Line 43. Data scenarios to execute, sep by blanks.
   scenarios="<YOUR_SCENARIOS>"

   # Line 49. Amount of runs.
   runs="<YOUR_RUNS>"

   # Lines 58 and 59. Method path and method name.
   method="$python $automap/methods/<YOUR_METHOD_PATH>.py"
   method_name="<YOUR_METHOD_NAME>"

   ```
3. Run the script - everything else is handled automatically

The script will create an experiment directory at `datasets/<dataset>/exps/<scenario>_<experiment_name>/` with the data files linked from `data/`.

### Converters

The converters module provides tools to transform between different RDF mapping formats and execute mappings.

### Map2RML - YARRRML to RML Conversion

Converts YARRRML mapping files to RML (Turtle) format using the [YATTER](https://github.com/RMLio/yatter) library.

#### Python API

```python
from automap.converters import Map2RML

# Initialize converter
converter = Map2RML()

# Convert YARRRML string to RML
yarrrml_content = """
prefixes:
  ex: http://example.org/
  
mappings:
  person:
    sources:
      - ['data.csv~csv']
    s: ex:$(id)
    po:
      - [a, ex:Person]
      - [ex:name, $(name)]
"""

rml_mapping = converter(yarrrml_content)
print(rml_mapping)
```

#### Command Line

```bash
# Read YARRRML from stdin, output RML to stdout
cat mapping.yml | python -m automap.converters.map2rml > mapping.rml.ttl
```

## RDF Turtle Light

A module for simplifying OWL/RDFS ontologies into minimal, readable Turtle format optimized for mapping generation.

### Overview

`Onto2LightTTL` reduces full OWL/RDFS ontologies to their essential components by extracting only the information needed for generating RDF mappings:

- Class declarations and hierarchies
- Object and datatype property declarations
- Domain and range definitions
- Labels and comments
- Excludes complex restrictions and blank nodes

### Features

- **Minimal output**: Removes verbose OWL constructs while preserving semantic structure
- **Ordered serialization**: Groups classes, object properties, and datatype properties separately
- **Format support**: Accepts any RDF format supported by RDFLib (Turtle, RDF/XML, N-Triples, etc.)
- **Clean syntax**: Uses `a` instead of `rdf:type` and `prefix` instead of `@prefix`

### Usage

#### Python API

```python
from automap.rdf_turtle_light import Onto2LightTTL

# Initialize converter
converter = Onto2LightTTL(ordered=True)

# Load full ontology
with open("full_ontology.ttl") as f:
    ontology = f.read()

# Convert to light Turtle
light_ttl = converter(ontology, input_format="turtle")

# Save or use the simplified ontology
with open("minimal_ontology.ttl", "w") as f:
    f.write(light_ttl)
```

#### Command Line

```bash
# Read ontology from stdin, output to stdout
cat full_ontology.rdf | python -m automap.rdf_turtle_light --format xml

# Save to file
cat full_ontology.ttl | python -m automap.rdf_turtle_light -o minimal.ttl
```

## Graph Evaluation

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

---

## Acknowledgments

This project builds upon and acknowledges the following:

### Funding

- **PIONERA Project**: This work is part of the PIONERA initiative

### Tools and Libraries

- **RMLMapper**: [RMLio/rmlmapper-java](https://github.com/RMLio/rmlmapper-java) - RML mapping execution
- **YATTER**: [RMLio/yatter](https://github.com/RMLio/yatter) - YARRRML to RML conversion
- **kg-pipeline**: [Vehnem/kg-pipeline](https://github.com/Vehnem/kg-pipeline) - Original evaluation framework
- **RDFLib**: Python library for working with RDF
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library

### Research

The graph evaluation module is a modified version of the code from:
> Vehnem et al., "Towards self-configuring Knowledge Graph Construction Pipelines using LLMs - A Case Study with RML", CEUR-WS Vol-3718, 2023
> [Paper](https://ceur-ws.org/Vol-3718/paper6.pdf)

---

## Authors and Contact

### Main Contributors

- **Carlos Golvano** - Main developer
  - GitHub: [@CarlosGolvano](https://github.com/CarlosGolvano)

### Contact

For questions, suggestions, or collaboration opportunities:
- **Issues**: [GitHub Issues](https://github.com/CarlosGolvano/automap/issues)
- **Email**: carlos.golvano@upm.es

---

## Funding

This work has received funding from the PIONERA project (Enhancing interoperability in data spaces through artificial intelligence), a project funded in the context of the call for Technological Products and Services for Data Spaces of the Ministry for Digital Transformation and Public Administration within the framework of the PRTR funded by the European Union (NextGenerationEU)

<div align="center">
  <img src="Logos financiación.png" alt="Logos financiación" width="900" />
</div>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
