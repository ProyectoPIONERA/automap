import sys
import yatter
from ruamel.yaml import YAML


class Map2RML:
    def __init__(self):
        self.yaml = YAML(typ="safe", pure=True)

    def __call__(self, yarrrml_content: str) -> str:
        if not yarrrml_content:
            raise ValueError("Input YARRRML string is empty")

        yaml_content = self.yaml.load(yarrrml_content)

        return yatter.translate(yaml_content)


def main():
    yarrrml_str = sys.stdin.read()

    map2rml = Map2RML()
    output = map2rml(yarrrml_str)
    print(output)


if __name__ == "__main__":
    sys.exit(main())
