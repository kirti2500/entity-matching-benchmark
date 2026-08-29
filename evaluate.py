"""
evaluate.py

The scoring engine. Takes a dataframe that has BOTH:
  - predicted_cluster  (the algorithm's guess — who does it THINK is who)
  - person_id          (the ground truth — who ACTUALLY is who)

and computes pairwise precision, recall, and F1.

This file is strategy-agnostic — it doesn't care HOW predicted_cluster was
generated (exact match, fuzzy, embeddings), only that it exists. That's
what lets us reuse this exact same scoring logic for all 3 strategies and
get a fair, apples-to-apples comparison.
"""

import time
from itertools import combinations
import pandas as pd


def get_pairs_from_clusters(df: pd.DataFrame, cluster_col: str) -> set:
    """
    Converts a clustering (records grouped under the same id) into the set
    of all record_id pairs implied by that grouping.

    e.g. if records 1, 2, 3 are all in the same cluster, this returns
    {(1,2), (1,3), (2,3)} — every pair is "claimed" as a duplicate pair.
    """
    pairs = set()
    for _, group in df.groupby(cluster_col):
        ids = sorted(group["record_id"].tolist())
        for pair in combinations(ids, 2):
            pairs.add(pair)
    return pairs


def evaluate(df: pd.DataFrame, predicted_col: str = "predicted_cluster") -> dict:
    """
    Computes precision, recall, F1 for a predicted clustering against the
    true person_id clustering.
    """
    predicted_pairs = get_pairs_from_clusters(df, predicted_col)
    true_pairs = get_pairs_from_clusters(df, "person_id")

    true_positives = predicted_pairs & true_pairs
    false_positives = predicted_pairs - true_pairs
    false_negatives = true_pairs - predicted_pairs

    tp, fp, fn = len(true_positives), len(false_positives), len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_strategy(df: pd.DataFrame, match_fn, strategy_name: str) -> dict:
    """
    Times and evaluates a single matching strategy.
    match_fn must take a df and return a df with a 'predicted_cluster' column.
    """
    start = time.time()
    result_df = match_fn(df.copy())
    elapsed = time.time() - start

    scores = evaluate(result_df)
    scores["strategy"] = strategy_name
    scores["runtime_sec"] = round(elapsed, 4)
    return scores


if __name__ == "__main__":
    from strategy_exact import match_exact

    df = pd.read_csv("people_raw.csv")
    scores = run_strategy(df, match_exact, "Exact Match (baseline)")

    print(f"\n{'='*50}")
    print(f"Strategy: {scores['strategy']}")
    print(f"{'='*50}")
    print(f"Precision: {scores['precision']:.2%}  (of predicted matches, how many were right)")
    print(f"Recall:    {scores['recall']:.2%}  (of true duplicates, how many were caught)")
    print(f"F1 score:  {scores['f1']:.4f}")
    print(f"Runtime:   {scores['runtime_sec']}s")
    print(f"\nTrue Positives:  {scores['true_positives']}")
    print(f"False Positives: {scores['false_positives']}")
    print(f"False Negatives: {scores['false_negatives']}")