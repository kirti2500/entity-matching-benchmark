"""
strategy_exact.py

Strategy 1 — Naive exact match (the baseline).

Two records are considered the same person ONLY if their full_name AND
phone are identical, character-for-character. No cleanup, no normalization.
This represents what happens when duplicate detection is done carelessly —
e.g. a raw SQL JOIN on unprocessed text fields.

We expect this to have LOW recall (misses typos, format differences) but
HIGH precision (when it does match, it's very likely correct, since exact
identical strings rarely happen by coincidence).
"""

import pandas as pd


def match_exact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a predicted_cluster id to each record. Records with the exact
    same (full_name, phone) pair get the same cluster id.
    """
    df = df.copy()
    # groupby + ngroup() gives each unique (full_name, phone) combo a
    # distinct integer id — this IS the algorithm's "guess" at who's who.
    df["predicted_cluster"] = df.groupby(["full_name", "phone"]).ngroup()
    return df


if __name__ == "__main__":
    df = pd.read_csv("people_raw.csv")
    result = match_exact(df)
    n_predicted_people = result["predicted_cluster"].nunique()
    print(f"Raw records: {len(df)}")
    print(f"True unique people (ground truth, hidden from algorithm): {df['person_id'].nunique()}")
    print(f"Predicted unique people (naive exact match): {n_predicted_people}")