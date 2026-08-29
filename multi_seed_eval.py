"""
multi_seed_eval.py

Robustness check: runs the fuzzy-vs-embedding comparison across MANY
independently generated random datasets (different seeds), not just one.

Why this matters: a single result could be a coincidence of that specific
random dataset. If fuzzy matching wins consistently across many
independent random draws, that's real evidence of a genuine pattern. If
results vary wildly between runs, that's important to know and report
honestly too - a benchmark's credibility depends on this check, not just
on one favorable run.

Reports mean +/- standard deviation for precision, recall, and F1 for
both strategies at their previously-found best thresholds (fuzzy=85,
embedding=0.80).
"""

import numpy as np
import pandas as pd

from generate_data import generate_dataset
from evaluate import evaluate
from strategy_fuzzy import match_fuzzy
from strategy_embedding import match_embedding
from sentence_transformers import SentenceTransformer


def run_multi_seed(n_runs=10, fuzzy_threshold=85, embedding_threshold=0.80):
    print(f"Loading embedding model once (reused across all {n_runs} runs)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    fuzzy_results = []
    embedding_results = []

    for seed in range(n_runs):
        print(f"\n--- Run {seed + 1}/{n_runs} (seed={seed}) ---")

        # temporarily override the module-level seed so each run gets a
        # genuinely different random dataset
        import generate_data
        generate_data.random.seed(seed)
        df = generate_dataset(n_people=200, max_variants=3, out_path=f"_temp_seed{seed}.csv")

        fuzzy_df = match_fuzzy(df.copy(), threshold=fuzzy_threshold)
        fuzzy_scores = evaluate(fuzzy_df)
        fuzzy_results.append(fuzzy_scores)
        print(f"  fuzzy:     precision={fuzzy_scores['precision']:.3f}, recall={fuzzy_scores['recall']:.3f}, f1={fuzzy_scores['f1']:.3f}")

        emb_df = match_embedding(df.copy(), threshold=embedding_threshold, model=model)
        emb_scores = evaluate(emb_df)
        embedding_results.append(emb_scores)
        print(f"  embedding: precision={emb_scores['precision']:.3f}, recall={emb_scores['recall']:.3f}, f1={emb_scores['f1']:.3f}")

    return pd.DataFrame(fuzzy_results), pd.DataFrame(embedding_results)


def summarize(df, name):
    print(f"\n{'='*55}")
    print(f"{name} — across {len(df)} independent random datasets")
    print(f"{'='*55}")
    for metric in ["precision", "recall", "f1"]:
        mean = df[metric].mean()
        std = df[metric].std()
        print(f"  {metric:10s}: {mean:.4f} ± {std:.4f}   (min={df[metric].min():.4f}, max={df[metric].max():.4f})")


if __name__ == "__main__":
    fuzzy_df, embedding_df = run_multi_seed(n_runs=10)

    summarize(fuzzy_df, "FUZZY MATCHING (threshold=85)")
    summarize(embedding_df, "EMBEDDING MATCHING (threshold=0.80)")

    fuzzy_df.to_csv("multiseed_fuzzy_results.csv", index=False)
    embedding_df.to_csv("multiseed_embedding_results.csv", index=False)
    print("\nSaved detailed per-run results to multiseed_fuzzy_results.csv and multiseed_embedding_results.csv")

    # cleanup temp files
    import glob, os
    for f in glob.glob("_temp_seed*.csv"):
        os.remove(f)