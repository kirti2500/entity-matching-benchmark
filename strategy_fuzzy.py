"""
strategy_fuzzy.py

Strategy 2 — Fuzzy string matching.

Instead of requiring exact identical strings, we measure similarity using
edit distance (how many character edits separate two strings) via the
rapidfuzz library. Two records are linked if their combined name+phone
similarity score is above a chosen threshold.

We use union-find (same technique as the entity-resolution project) to turn
pairwise "these two are similar enough" decisions into final groups —
if A matches B, and B matches C, then A, B, and C all end up in the same
predicted group, even if A and C weren't directly compared as similar.
"""

import re
import pandas as pd
from rapidfuzz import fuzz


def normalize_phone(phone):
    """Same normalization logic as the original entity-resolution project —
    strip everything but digits, keep last 10."""
    digits = re.sub(r"\D", "", str(phone))
    return digits[-10:] if len(digits) >= 10 else digits


def similarity_score(row_a, row_b):
    """
    Combines name similarity and phone similarity into one score.
    Phone is normalized first so formatting differences (+91, leading 0,
    dashes) don't unfairly tank the similarity score.
    """
    name_sim = fuzz.ratio(str(row_a["full_name"]).lower(), str(row_b["full_name"]).lower())
    phone_a = normalize_phone(row_a["phone"])
    phone_b = normalize_phone(row_b["phone"])
    phone_sim = fuzz.ratio(phone_a, phone_b)

    # weight name and phone equally
    return (name_sim + phone_sim) / 2


class UnionFind:
    """Standard union-find / disjoint-set structure."""
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


def match_fuzzy(df: pd.DataFrame, threshold: float = 85.0) -> pd.DataFrame:
    """
    Compares every pair of records, links any pair scoring above `threshold`
    into the same union-find group, then assigns predicted_cluster ids.

    NOTE: this is an O(n^2) approach — fine for our benchmark size (~400
    records), but a real production system with millions of records would
    need blocking (e.g. only compare records in the same city) to avoid
    comparing every record to every other record. Worth mentioning as a
    known scaling limitation.
    """
    df = df.copy()
    records = df.to_dict("records")
    uf = UnionFind(df["record_id"].tolist())

    n = len(records)
    for i in range(n):
        for j in range(i + 1, n):
            score = similarity_score(records[i], records[j])
            if score >= threshold:
                uf.union(records[i]["record_id"], records[j]["record_id"])

    df["predicted_cluster"] = df["record_id"].apply(uf.find)
    return df


if __name__ == "__main__":
    from evaluate import run_strategy

    df = pd.read_csv("people_raw.csv")
    scores = run_strategy(df, lambda d: match_fuzzy(d, threshold=85.0), "Fuzzy Match (threshold=85)")

    print(f"\n{'='*50}")
    print(f"Strategy: {scores['strategy']}")
    print(f"{'='*50}")
    print(f"Precision: {scores['precision']:.2%}")
    print(f"Recall:    {scores['recall']:.2%}")
    print(f"F1 score:  {scores['f1']:.4f}")
    print(f"Runtime:   {scores['runtime_sec']}s")
    print(f"\nTrue Positives:  {scores['true_positives']}")
    print(f"False Positives: {scores['false_positives']}")
    print(f"False Negatives: {scores['false_negatives']}")