"""
strategy_embedding.py

Strategy 3 - Embedding-based semantic matching.

Instead of comparing strings character-by-character (fuzzy matching) or
requiring exact equality (baseline), we convert each name into a vector
using a pretrained sentence embedding model. Names that are semantically
related - including nicknames the model has seen used interchangeably in
its training data - end up close together in vector space, even if the
strings themselves share almost no characters (e.g. "Vicky" vs "Vikram").

Similarity between two vectors is measured with cosine similarity (ranges
from -1 to 1, where 1 means identical direction / maximally similar).

Phone number similarity still uses simple string comparison since phone
numbers aren't natural language - embeddings wouldn't help there.

Same union-find clustering approach as the fuzzy matching strategy, so the
only real difference between the two scripts is HOW similarity is measured.
"""

import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, well-regarded general-purpose model


def normalize_phone(phone):
    digits = re.sub(r"\D", "", str(phone))
    return digits[-10:] if len(digits) >= 10 else digits


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class UnionFind:
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


def match_embedding(df: pd.DataFrame, threshold: float = 0.80, model=None) -> pd.DataFrame:
    """
    threshold is a cosine similarity cutoff (0 to 1), NOT a percentage like
    the fuzzy strategy's threshold=85 - different similarity methods use
    different scales, worth noting when comparing strategies side by side.
    """
    df = df.copy()

    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    names = df["full_name"].astype(str).tolist()
    name_embeddings = model.encode(names, show_progress_bar=False)

    phones_norm = df["phone"].apply(normalize_phone).tolist()
    record_ids = df["record_id"].tolist()

    uf = UnionFind(record_ids)
    n = len(df)

    for i in range(n):
        for j in range(i + 1, n):
            name_sim = cosine_similarity(name_embeddings[i], name_embeddings[j])
            phone_sim = fuzz.ratio(phones_norm[i], phones_norm[j]) / 100.0

            combined = (name_sim + phone_sim) / 2
            if combined >= threshold:
                uf.union(record_ids[i], record_ids[j])

    df["predicted_cluster"] = df["record_id"].apply(uf.find)
    return df


if __name__ == "__main__":
    from evaluate import run_strategy

    df = pd.read_csv("people_raw.csv")

    print("Loading embedding model (first run downloads it, ~90MB)...")
    model = SentenceTransformer(MODEL_NAME)

    scores = run_strategy(
        df,
        lambda d: match_embedding(d, threshold=0.80, model=model),
        "Embedding Match (threshold=0.80)"
    )

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