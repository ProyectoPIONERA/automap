"""
Scoring Functions for Graph Evaluation

This module provides pure mathematical functions for calculating 
evaluation metrics such as precision, recall, and F1-score.
"""


def precision_score(tp: int, fp: int) -> float:
    return tp / (tp + fp) if tp + fp > 0 else 0.0


def recall_score(tp: int, fn: int) -> float:
    return tp / (tp + fn) if tp + fn > 0 else 0.0


def f1_score(tp: int, fp: int, fn: int) -> float:
    precision = precision_score(tp, fp)
    recall = recall_score(tp, fn)
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def calculate_metrics(tp: int, fp: int, fn: int, tn: int = 0) -> dict:
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'p': precision_score(tp, fp),
        'r': recall_score(tp, fn),
        'f1': f1_score(tp, fp, fn)
    }


def overlapping_lists(list1: list, list2: list) -> list:
    """
    Find overlapping elements between two lists using a two-pointer approach.

    This function sorts both lists and finds common elements efficiently.
    Time complexity: O(n log n + m log m + n + m)

    Args:
        list1: First list
        list2: Second list

    Returns:
        list: List of overlapping elements
    """
    sorted1 = sorted(list1)
    sorted2 = sorted(list2)

    overlap = []
    i = 0
    j = 0

    while i < len(sorted1) and j < len(sorted2):
        if sorted1[i] < sorted2[j]:
            i += 1
        elif sorted1[i] > sorted2[j]:
            j += 1
        else:
            overlap.append(sorted1[i])
            i += 1
            j += 1

    return overlap


def average(scores: list) -> float:
    return sum(scores) / len(scores) if len(scores) > 0 else 0.0
