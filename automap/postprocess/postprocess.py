from typing import List
from map2rml import Map2RML
from pathlib import Path


def _parse_argv():
    import argparse

    parser = argparse.ArgumentParser(description="Postprocess YARRRML to RDF graph.")
    parser.add_argument(
        "exp_path", type=str,
        help="Path to the experiment directory."
    )
    parser.add_argument(
        "--mapping_name", type=str,
        help="Name of the input mapping file.", default="mapping.yml"
    )

    return parser.parse_args()


def postprocess(yarrrml_content: str, metadata: dict = None) -> str:
    map2rml = Map2RML()
    rml_mapping = map2rml(yarrrml_content)

    status = 1 if not rml_mapping else 0
    # add_metadata(metadata, 'mapping_status', status)

    return rml_mapping


if __name__ == "__main__":
    """Main CLI entry point."""
    args = _parse_argv()

    mapping_path = Path(args.exp_path) / args.mapping_name
    output_path = Path(args.exp_path) / "rml_mapping.ttl"

    with open(mapping_path, 'r') as f:
        yarrrml_content = f.read()

    metadata = {}
    rml_mapping = postprocess(yarrrml_content, metadata=metadata)

    with open(output_path, 'w') as f:
        f.write(rml_mapping)

    print(metadata)
