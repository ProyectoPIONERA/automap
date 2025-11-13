from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--csv')
    parser.add_argument('--output')
    args = parser.parse_args()

    csv_file = args.csv

    mapping = \
        f"""prefixes:
    ex: "http://example.org/"
    rdfs: "http://www.w3.org/2000/01/rdf-schema#"
    owl: "http://www.w3.org/2002/07/owl#"

sources:
    students: ["/home/carlos/workspace/automap/datasets/blinkg/data/scenario1/1B/student.csv~csv"]
    # students: [/home/carlos/workspace/automap/datasets/blinkg/data/scenario1/1B/student.json~jsonpath, "$.students[*]"]

mappings:
    Student:
        sources:
            - students
        s: ex:$(ID)
        po:
            - [a, ex:Person]
            - [ex:id, $(ID), xsd:integer]
            - [ex:fullname, $(Name), xsd:string]
"""

    with open(args.output, 'w') as f:
        f.write(mapping)
