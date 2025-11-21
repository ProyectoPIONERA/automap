# RDF Turtle Light Transformer

A standalone Python module for converting standard RDF Turtle syntax to simplified "Turtle Light" variations.

## Overview

This module implements the Turtle Light syntax transformation from the [12ShadesOfRDFSyntax](https://github.com/datalogism/12ShadesOfRDFSyntax) research project. It can generate 8 different variations of simplified Turtle syntax by controlling three parameters:

- **datatype**: Keep or remove datatype annotations (`^^xsd:type`)
- **inline**: Single-line vs multiline format
- **factorised**: Subject appears once (factorised) vs repeated on each line (non-factorised)

## Installation

```bash
# Install dependencies
pip install rdflib

# Optional: Make the script executable
chmod +x rdf_turtle_light.py
```

## Quick Start

### As a Python Module (API)

```python
from rdf_turtle_light import simplify_turtle, generate_all_variations

# Standard Turtle input
turtle = """
@prefix ex: <http://example.org/>.
@prefix foaf: <http://xmlns.com/foaf/0.1/>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

ex:homerSimpson a foaf:Person ;
    foaf:name "Homer Simpson"^^xsd:string ;
    ex:birthDate "1987-04-19"^^xsd:date ;
    ex:birthYear "1987"^^xsd:gYear.
"""

# Generate single variation
simplified = simplify_turtle(turtle, datatype=False, inline=True, factorised=True)
print(simplified)
# Output: :homerSimpson a :Person ; :name "Homer Simpson" ; :birthDate "1987-04-19" ; :birthYear "1987".

# Generate all 8 variations
variations = generate_all_variations(turtle)
for name, content in variations.items():
    print(f"\n{name}:\n{content}")
```

### As a Command-Line Tool (CLI)

```bash
# Basic transformation
python rdf_turtle_light.py -i input.ttl -o output.ttl

# Inline, factorised format without datatypes
python rdf_turtle_light.py -i input.ttl -o output.ttl --inline --factorised

# Generate all 8 variations
python rdf_turtle_light.py -i input.ttl --all-variations -d output_variations/

# Process from stdin/stdout (useful for piping)
cat input.ttl | python rdf_turtle_light.py --stdin --inline > output.ttl

# Transform JSON dataset
python rdf_turtle_light.py -i dataset.json -o dataset_light.json --json --inline

# Convert from other RDF formats
python rdf_turtle_light.py -i data.rdf --format xml -o output.ttl --inline
```

## The 8 Turtle Light Variations

| Variation | datatype | inline | factorised | Description |
|-----------|----------|--------|------------|-------------|
| `0datatype_0inline_0facto` | No | No | No | Multiline, non-factorised, no datatypes |
| `0datatype_0inline_1facto` | No | No | Yes | Multiline, factorised, no datatypes |
| `0datatype_1inline_0facto` | No | Yes | No | Inline, non-factorised, no datatypes |
| `0datatype_1inline_1facto` | No | Yes | Yes | **Turtle Ultra Light** - Inline, factorised, no datatypes |
| `1datatype_0inline_0facto` | Yes | No | No | Multiline, non-factorised, with datatypes |
| `1datatype_0inline_1facto` | Yes | No | Yes | Multiline, factorised, with datatypes |
| `1datatype_1inline_0facto` | Yes | Yes | No | Inline, non-factorised, with datatypes |
| `1datatype_1inline_1facto` | Yes | Yes | Yes | Inline, factorised, with datatypes |

### Example Variations

**Original Turtle:**
```turtle
@prefix ex: <http://example.org/>.
@prefix foaf: <http://xmlns.com/foaf/0.1/>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

ex:homerSimpson a foaf:Person ;
    foaf:name "Homer Simpson"^^xsd:string ;
    ex:birthDate "1987-04-19"^^xsd:date .
```

**Variation 1: Multiline, Factorised, No Datatype** (`0datatype_0inline_1facto`)
```turtle
:homerSimpson a :Person ;
   :name "Homer Simpson" ;
   :birthDate "1987-04-19".
```

**Variation 2: Inline, Factorised, No Datatype** (`0datatype_1inline_1facto`) - **Turtle Ultra Light**
```turtle
:homerSimpson a :Person ; :name "Homer Simpson" ; :birthDate "1987-04-19".
```

**Variation 3: Multiline, Non-Factorised, No Datatype** (`0datatype_0inline_0facto`)
```turtle
:homerSimpson a :Person .
:homerSimpson :name "Homer Simpson" .
:homerSimpson :birthDate "1987-04-19" .
```

## API Reference

### Core Functions

#### `simplify_turtle(turtle_input, datatype=False, inline=False, factorised=True)`

Transform standard Turtle RDF to Turtle Light syntax.

**Parameters:**
- `turtle_input` (str): Standard Turtle RDF string
- `datatype` (bool): Keep datatype annotations (default: False)
- `inline` (bool): Use single-line format (default: False)
- `factorised` (bool): Factorised format (default: True)

**Returns:** Simplified Turtle Light string

#### `generate_all_variations(turtle_input)`

Generate all 8 Turtle Light variations.

**Parameters:**
- `turtle_input` (str): Standard Turtle RDF string

**Returns:** Dictionary mapping variation names to transformed strings

#### `transform_file(input_file, output_file=None, datatype=False, inline=False, factorised=True, format='turtle')`

Transform an RDF file to Turtle Light format.

**Parameters:**
- `input_file` (str|Path): Input file path
- `output_file` (str|Path, optional): Output file path
- `datatype` (bool): Keep datatypes (default: False)
- `inline` (bool): Inline format (default: False)
- `factorised` (bool): Factorised format (default: True)
- `format` (str): Input format: 'turtle', 'xml', 'json-ld', 'ntriples' (default: 'turtle')

**Returns:** Transformed string

#### `transform_json_dataset(input_json, output_json, triples_field='triples', datatype=False, inline=False, factorised=True)`

Transform a JSON dataset file containing RDF triples.

**Parameters:**
- `input_json` (str|Path): Input JSON file
- `output_json` (str|Path): Output JSON file
- `triples_field` (str): Field containing triples (default: 'triples')
- `datatype` (bool): Keep datatypes (default: False)
- `inline` (bool): Inline format (default: False)
- `factorised` (bool): Factorised format (default: True)

## CLI Reference

```
usage: rdf_turtle_light.py [-h] [-i INPUT] [-o OUTPUT] [--stdin]
                          [--format {turtle,xml,json-ld,ntriples,n3}]
                          [--json] [--triples-field TRIPLES_FIELD]
                          [--datatype] [--inline] [--factorised]
                          [--non-factorised] [--all-variations]
                          [-d OUTPUT_DIR]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input RDF file path
  -o OUTPUT, --output OUTPUT
                        Output file path (if not specified, prints to stdout)
  --stdin               Read input from stdin
  --format {turtle,xml,json-ld,ntriples,n3}
                        Input RDF format (default: turtle)
  --json                Process JSON dataset file
  --triples-field TRIPLES_FIELD
                        Field name containing triples in JSON (default: triples)
  --datatype            Keep datatype annotations (^^xsd:type)
  --inline              Use inline (single-line) format
  --factorised          Use factorised format (subject appears once)
  --non-factorised      Use non-factorised format (subject repeats)
  --all-variations      Generate all 8 Turtle Light variations
  -d OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory for all variations
```

## Use Cases

### Bash Script Integration

```bash
#!/bin/bash
# Process all Turtle files in a directory

for file in data/*.ttl; do
    echo "Processing $file"
    python rdf_turtle_light.py -i "$file" -o "output/$(basename $file)" --inline --factorised
done
```

### Batch Processing

```python
from pathlib import Path
from rdf_turtle_light import transform_file

input_dir = Path("data/rdf_files")
output_dir = Path("data/turtle_light")
output_dir.mkdir(exist_ok=True)

for rdf_file in input_dir.glob("*.ttl"):
    transform_file(
        rdf_file,
        output_dir / rdf_file.name,
        datatype=False,
        inline=True,
        factorised=True
    )
```

### Research/Experimentation

```python
from rdf_turtle_light import generate_all_variations
import json

# Load dataset
with open("my_dataset.json") as f:
    data = json.load(f)

# Generate all variations for analysis
for item in data:
    variations = generate_all_variations(item["triples"])
    item["variations"] = variations

# Save augmented dataset
with open("dataset_with_variations.json", "w") as f:
    json.dump(data, f)
```

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@InProceedings{10.1007/978-3-031-78952-6_8,
author="Ringwald, C{\'e}lian and Gandon, Fabien and Faron, Catherine and Michel, Franck and Akl, Hanna Abi",
title="12 Shades of RDF: Impact of Syntaxes on Data Extraction with Language Models",
booktitle="The Semantic Web: ESWC 2024 Satellite Events",
year="2025",
publisher="Springer Nature Switzerland",
pages="81--91"
}
```

## License

Extracted from the [12ShadesOfRDFSyntax](https://github.com/datalogism/12ShadesOfRDFSyntax) project.

## Related Resources

- Original research repository: https://github.com/datalogism/12ShadesOfRDFSyntax
- WandB experiments: https://wandb.ai/celian-ringwald/12ShadesOfRDF
