# TODO: Revisar porque esto lo ha generado Clause Sonnet
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _escape_property_value(value: Path) -> str:
    """Escape ``value`` so it can be written to a Java ``.properties`` file."""

    text = str(value)
    text = text.replace("\\", "\\\\")
    return text.replace("\n", "\\n")


def _validate_path(path: Path | None, description: str) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} '{resolved}' does not exist")
    return resolved


def _get_rmlmapper_path(arg: Path | None) -> Path:
    if arg is not None:
        mapper = arg
    else:
        env = os.getenv("RMLMAPPER_JAR")
        if not env:
            raise RuntimeError(
                "Path to RMLMapper jar is required. Provide --rmlmapper or set "
                "RMLMAPPER_JAR"
            )
        mapper = Path(env)
    mapper = mapper.resolve()
    if not mapper.exists():
        raise FileNotFoundError(f"RMLMapper jar '{mapper}' does not exist")
    return mapper


def _run_mapper(
    mapper: Path,
    mapping: Path,
    destination: Path,
    serialization: str,
    parameters: Path | None,
    *,
    working_directory: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        "java",
        "-jar",
        str(mapper),
        "-m",
        str(mapping),
        "-o",
        str(destination),
        "-s",
        serialization,
    ]

    if parameters is not None:
        command.extend(["-p", str(parameters)])

    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=working_directory,
    )


def map2graph(
    mapping: Path,
    ontology: Path | None = None,
    headers: Path | None = None,
    output: Path | None = None,
    serialization: str = "turtle",
    rmlmapper: Path | None = None,
) -> Path:
    """Execute the RML ``mapping`` and return the path to the generated graph."""

    mapping_path = _validate_path(mapping, "Mapping file")
    if mapping_path is None:  # pragma: no cover - defensive, arg required
        raise ValueError("A mapping file must be provided")
    ontology = _validate_path(ontology, "Ontology")
    headers = _validate_path(headers, "Headers")

    mapper_path = _get_rmlmapper_path(rmlmapper)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_output = Path(tmpdir) / "graph.ttl"

        parameters_path: Path | None = None
        if ontology or headers:
            parameters_path = Path(tmpdir) / "parameters.properties"
            parameter_lines = []
            if ontology:
                escaped = _escape_property_value(ontology)
                parameter_lines.append(f"ontology={escaped}")
            if headers:
                escaped = _escape_property_value(headers)
                parameter_lines.append(f"headers={escaped}")
            parameters_path.write_text("\n".join(parameter_lines), encoding="utf-8")

        result = _run_mapper(
            mapper_path,
            mapping_path,
            tmp_output,
            serialization,
            parameters_path,
            working_directory=mapping_path.parent,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "RMLMapper failed with exit code "
                f"{result.returncode}: {result.stderr.strip()}"
            )

        if not tmp_output.exists() or tmp_output.stat().st_size == 0:
            raise RuntimeError("Mapping executed but produced no triples")

        if output is None:
            final_output = Path.cwd() / tmp_output.name
        else:
            final_output = Path(output).resolve()
            final_output.parent.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(tmp_output, final_output)

    return final_output


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a knowledge graph from an RML mapping"
    )
    parser.add_argument("mapping", type=Path, help="Path to the RML mapping file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="File to write the generated graph to. Defaults to graph.ttl",
    )
    parser.add_argument("--ontology", type=Path, help="Path to the ontology file")
    parser.add_argument("--headers", type=Path, help="Path to the data headers")
    parser.add_argument(
        "--serialization",
        default="turtle",
        help="Serialization format supported by RMLMapper (default: turtle)",
    )
    parser.add_argument(
        "--rmlmapper",
        type=Path,
        help="Path to the rmlmapper.jar. Overrides RMLMAPPER_JAR",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Do not print the generated graph to stdout",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        graph_path = map2graph(
            mapping=args.mapping,
            ontology=args.ontology,
            headers=args.headers,
            output=args.output,
            serialization=args.serialization,
            rmlmapper=args.rmlmapper,
        )
    except Exception as exc:  # pragma: no cover - thin wrapper
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.no_print:
        with open(graph_path, "r", encoding="utf-8") as graph_file:
            for line in graph_file:
                print(line.rstrip())

    print(graph_path, file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
