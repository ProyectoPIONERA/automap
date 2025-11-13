def print_title(header: str, level: int = 1, mark: str = ''):
    line_len = 80
    mid_len_0 = (line_len - len(header) - 2) / 2
    mid_len_1 = mid_len_0 if mid_len_0.is_integer() else int(mid_len_0) + 1

    if level == 1:
        char = '='
    elif level == 2:
        char = '-'
    elif level == 3:
        char = '~'
    else:
        char = '`'

    print(mark + '+' + char * (line_len - 2) + '+')
    print(mark + '+' + char * int(mid_len_0) + header + char * int(mid_len_1) + '+')
    print(mark + '+' + char * (line_len - 2) + '+')


def print_metrics(metrics: dict, mark: str = ''):
    # TODO: hardcoded metrics is bad.
    # metrics = ['tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1']
    printable_metrics = ['p', 'r', 'f1']
    exclude = ["errors"]
    blanks = max(len(key) for key in metrics.keys()) + 1

    print(mark + ' ' * blanks + '\t'.join(printable_metrics))
    for key, values in metrics.items():
        if key not in exclude:
            print(mark + key + ' ' * (blanks - len(key)), end="")
            for metric in printable_metrics:
                if metric in values:
                    if metric == printable_metrics[-1]:
                        print(f"{metrics[key][metric]:<.2f}", end="")
                    else:
                        print(f"{metrics[key][metric]:<.2f}\t", end="")
                else:
                    print(f"N/A\t", end="")
            print()


if __name__ == "__main__":
    import sys
    import json

    json_data = json.loads(sys.stdin.read())
    print_title("SUMMARY")
    print_metrics(json_data)
