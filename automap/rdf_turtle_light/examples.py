#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of the RDF Turtle Light Transformer module.
"""

from rdf_turtle_light import (
    simplify_turtle,
    generate_all_variations,
    transform_file,
    transform_json_dataset
)


def example_basic_transformation():
    """Example 1: Basic transformation with different parameters."""
    print("=" * 80)
    print("Example 1: Basic Transformation")
    print("=" * 80)

    turtle_input = """
@prefix ex: <http://example.org/>.
@prefix foaf: <http://xmlns.com/foaf/0.1/>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

ex:person1 a foaf:Person ;
    foaf:name "Homer Simpson"^^xsd:string ;
    ex:birthDate "1987-04-19"^^xsd:date ;
    ex:birthYear "1987"^^xsd:gYear ;
    ex:alias "Homer J. Simpson"^^xsd:string .
"""

    # Variation 1: Multiline, factorised, no datatype (most simplified)
    print("\n1. Multiline, Factorised, No Datatype:")
    result1 = simplify_turtle(turtle_input, datatype=False, inline=False, factorised=True)
    print(result1)

    # Variation 2: Inline, factorised, no datatype (Turtle Ultra Light)
    print("\n2. Inline, Factorised, No Datatype (Turtle Ultra Light):")
    result2 = simplify_turtle(turtle_input, datatype=False, inline=True, factorised=True)
    print(result2)

    # Variation 3: Multiline, non-factorised, no datatype
    print("\n3. Multiline, Non-Factorised, No Datatype:")
    result3 = simplify_turtle(turtle_input, datatype=False, inline=False, factorised=False)
    print(result3)

    # Variation 4: Multiline, factorised, with datatype
    print("\n4. Multiline, Factorised, With Datatype:")
    result4 = simplify_turtle(turtle_input, datatype=True, inline=False, factorised=True)
    print(result4)


def example_all_variations():
    """Example 2: Generate all 8 variations at once."""
    print("\n" + "=" * 80)
    print("Example 2: All 8 Variations")
    print("=" * 80)

    turtle_input = """
@prefix ex: <http://example.org/>.
@prefix foaf: <http://xmlns.com/foaf/0.1/>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

ex:marieCurie a foaf:Person ;
    foaf:name "Marie Curie"^^xsd:string ;
    ex:birthDate "1867-11-07"^^xsd:date ;
    ex:birthYear "1867"^^xsd:gYear ;
    ex:deathDate "1934-07-04"^^xsd:date ;
    ex:deathYear "1934"^^xsd:gYear .
"""

    variations = generate_all_variations(turtle_input)

    for config_name, transformed in variations.items():
        print(f"\n{config_name}:")
        print("-" * 60)
        print(transformed)


def example_file_transformation():
    """Example 3: Transform files."""
    print("\n" + "=" * 80)
    print("Example 3: File Transformation")
    print("=" * 80)

    # Create a sample Turtle file
    sample_content = """
@prefix ex: <http://example.org/>.
@prefix foaf: <http://xmlns.com/foaf/0.1/>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

ex:albertEinstein a foaf:Person ;
    foaf:name "Albert Einstein"^^xsd:string ;
    ex:birthDate "1879-03-14"^^xsd:date ;
    ex:birthYear "1879"^^xsd:gYear ;
    ex:deathDate "1955-04-18"^^xsd:date .
"""

    # Write sample file
    with open("/tmp/sample_input.ttl", "w") as f:
        f.write(sample_content)

    print("\nTransforming /tmp/sample_input.ttl...")

    # Transform file
    result = transform_file(
        "/tmp/sample_input.ttl",
        "/tmp/sample_output.ttl",
        datatype=False,
        inline=True,
        factorised=True
    )

    print(f"\nOutput saved to /tmp/sample_output.ttl:")
    print(result)


def example_json_dataset():
    """Example 4: Transform JSON dataset."""
    print("\n" + "=" * 80)
    print("Example 4: JSON Dataset Transformation")
    print("=" * 80)

    import json

    # Create sample JSON dataset
    dataset = [
        {
            "entity": "Isaac_Newton",
            "abstract": "Sir Isaac Newton was an English mathematician and physicist.",
            "triples": """
@prefix ex: <http://example.org/>.
@prefix foaf: <http://xmlns.com/foaf/0.1/>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

ex:isaacNewton a foaf:Person ;
    foaf:name "Isaac Newton"^^xsd:string ;
    ex:birthDate "1643-01-04"^^xsd:date ;
    ex:birthYear "1643"^^xsd:gYear .
"""
        },
        {
            "entity": "Ada_Lovelace",
            "abstract": "Ada Lovelace was an English mathematician and writer.",
            "triples": """
@prefix ex: <http://example.org/>.
@prefix foaf: <http://xmlns.com/foaf/0.1/>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

ex:adaLovelace a foaf:Person ;
    foaf:name "Ada Lovelace"^^xsd:string ;
    ex:birthDate "1815-12-10"^^xsd:date ;
    ex:birthYear "1815"^^xsd:gYear .
"""
        }
    ]

    # Save input dataset
    with open("/tmp/sample_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print("\nTransforming JSON dataset...")

    # Transform
    transform_json_dataset(
        "/tmp/sample_dataset.json",
        "/tmp/sample_dataset_light.json",
        datatype=False,
        inline=True,
        factorised=True
    )

    # Load and display result
    with open("/tmp/sample_dataset_light.json") as f:
        result = json.load(f)

    print("\nTransformed dataset:")
    for item in result:
        print(f"\nEntity: {item['entity']}")
        print(f"Triples: {item['triples']}")


def example_programmatic_batch():
    """Example 5: Programmatic batch processing."""
    print("\n" + "=" * 80)
    print("Example 5: Batch Processing Multiple Entities")
    print("=" * 80)

    # Simulate multiple entities
    entities = [
        ("nikolaTesla", "Nikola Tesla", "1856-07-10", "1856", "1943-01-07"),
        ("graceHopper", "Grace Hopper", "1906-12-09", "1906", "1992-01-01"),
        ("alanTuring", "Alan Turing", "1912-06-23", "1912", "1954-06-07"),
    ]

    results = {}

    for identifier, name, birth_date, birth_year, death_date in entities:
        turtle = f"""
@prefix ex: <http://example.org/>.
@prefix foaf: <http://xmlns.com/foaf/0.1/>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

ex:{identifier} a foaf:Person ;
    foaf:name "{name}"^^xsd:string ;
    ex:birthDate "{birth_date}"^^xsd:date ;
    ex:birthYear "{birth_year}"^^xsd:gYear ;
    ex:deathDate "{death_date}"^^xsd:date .
"""

        # Generate Turtle Ultra Light version
        simplified = simplify_turtle(turtle, datatype=False, inline=True, factorised=True)
        results[identifier] = simplified

    print("\nBatch processing results (Turtle Ultra Light):")
    for identifier, simplified in results.items():
        print(f"\n{identifier}:")
        print(simplified)


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("RDF Turtle Light Transformer - Examples")
    print("=" * 80)

    example_basic_transformation()
    example_all_variations()
    example_file_transformation()
    example_json_dataset()
    example_programmatic_batch()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
