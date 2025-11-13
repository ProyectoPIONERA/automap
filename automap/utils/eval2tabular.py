from automap.utils import get_in_domain, get_common, print_metrics, print_title
from typing import List


class Eval2Tabular:
    def __init__(self,):
        self.diff_metrics_common = {"subjects_unique": {"test": "test_subjects_unique",
                                                        "reference": "reference_subjects_unique"},
                                    "subjects_fuzzy_unique": {"test": "test_subjects_fuzzy",
                                                              "reference": "reference_subjects_fuzzy"},
                                    "classes_unique": {"test": "test_classes",
                                                       "reference": "reference_classes"},
                                    "predicates_unique": {"test": "test_po",
                                                          "reference": "reference_po"},
                                    "predicate_datatype_range_unique": {"test": "test_p_datatype",
                                                                        "reference": "reference_p_datatype"},
                                    "objects_uris": {"test": "test_uris",
                                                     "reference": "reference_uris"},
                                    "objects_literals": {"test": "test_literals",
                                                         "reference": "reference_literals"},
                                    }
        self.test_key = "test"
        self.reference_key = "reference"

        self.basic_mark = '$ '
        self.details_mark = '# '

    def __call__(self, eval_json: dict, only_common=False, only_in_domain=False):
        if any(value for value in eval_json["errors"].values()):
            for key, value in eval_json["errors"].items():
                print(f"{key}\t{value}")
            return
        if only_common:
            self._print_common(eval_json)
        elif only_in_domain:
            self._print_in_domain(eval_json)
        else:
            self._print_common(eval_json)
            self._print_in_domain(eval_json)

    def _print_common(self, eval_json: dict):
        print_title("COMMON", mark=self.basic_mark)
        print_metrics(get_common(eval_json), mark=self.basic_mark)

        print_title("COMMON - DETAILS", level=2, mark=self.details_mark)
        print_len = 100
        for metric in self.diff_metrics_common:
            metric_test_key = self.diff_metrics_common[metric][self.test_key]
            metric_reference_key = self.diff_metrics_common[metric][self.reference_key]
            if metric_test_key in eval_json[metric] and metric_reference_key in eval_json[metric]:
                test_values = eval_json[metric][metric_test_key]
                reference_values = eval_json[metric][metric_reference_key]
                fp = self._get_fp(test_values, reference_values)
                fn = self._get_fn(test_values, reference_values)
                if len(fp) > 0:
                    print(f"{self.details_mark}{metric} - FP ({len(fp)})")
                    for item in fp:
                        print(f"{self.details_mark}\t{repr(item[:print_len])}{'[...]' if len(item) > print_len else ''}")
                    print(self.details_mark)
                if len(fn) > 0:
                    print(f"{self.details_mark}{metric} - FN ({len(fn)})")
                    for item in fn:
                        print(f"{self.details_mark}\t{repr(item[:print_len])}{'[...]' if len(item) > print_len else ''}")
                    print(self.details_mark)

    def _print_in_domain(self, eval_json: dict):
        print_title("IN DOMAIN", mark=self.basic_mark)
        print_metrics(get_in_domain(eval_json), mark=self.basic_mark)

    def _get_fp(self, test: List, reference: List) -> List:
        return list(set(test) - set(test).intersection(set(reference)))

    def _get_fn(self, test: List, reference: List) -> List:
        return list(set(reference) - set(test).intersection(set(reference)))


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only_common",
        action="store_true",
        help="Evaluate only common metrics.",
    )
    group.add_argument(
        "--only_in_domain",
        action="store_true",
        help="Evaluate only in domain metrics.",
    )
    args = parser.parse_args()

    eval2tabular = Eval2Tabular()
    eval2tabular(json.loads(sys.stdin.read()), only_common=args.only_common, only_in_domain=args.only_in_domain)
