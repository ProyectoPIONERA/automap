from pathlib import Path
from typing import List


def find_file(directory: Path, filenames: List[str]):
    files = [file for filename in filenames for file in directory.glob(filename)]
    if not files:
        raise FileNotFoundError(f"No {filenames} file found in the specified directory.")
    elif len(files) > 1:
        raise FileExistsError(f"Multiple {filenames} files found in the specified directory.")
    return files[0]


def find_yarml_file(directory: Path):
    return find_file(directory, ["mapping.yml", "mapping.yaml"])


def find_metadata_file(directory: Path):
    return find_file(directory, ["metadata.json"])
