# AUTOMAP

Automatic mapping generation pipeline.

## TODO + IDEAS

- Añadir referencia del mapping en la evaluación. Poder ver la parte del mapping que ha generado las tripletas que han fallado.

## Table of Contents

- [Requirements](#requirements)
- [Running Experiments](#running-experiments)
- Pipeline
  - [Preprocess](#preprocess)
  - [Converters](#converters)
  - [RDF Turtle Light](#rdf-turtle-light)
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

Lipsum

## 🔄 Converters

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

### RML2Graph - Execute RML Mappings

Executes RML mappings using the [RMLMapper](https://github.com/RMLio/rmlmapper-java) to generate RDF graphs.

#### Command Line

The RMLMapper path can be provided via the `RMLMAPPER_JAR` environment variable or the `--rmlmapper` flag. Use `--ontology` and `--headers` to provide paths that can be referenced in the mapping via RMLMapper parameter placeholders (`@{ontology}` and `@{headers}`).

## 📝 RDF Turtle Light

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

#### Example

[...]

### What Gets Filtered Out

[...]

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

[...]
---

### Legacy CLI Tools

The following command-line tools are also available for direct conversion operations:

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


