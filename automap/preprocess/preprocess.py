from argparse import ArgumentParser
from pathlib import Path


def preprocess_data(ontology_path: Path):
    return ontology_path


def _parse_argv():
    parser = ArgumentParser(description="Preprocess data for training.")
    parser.add_argument(
        "--ontology",
        type=str,
        required=True,
        help="Path to the ontology file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_argv()
    preprocess_data()
