"""
sweep_thresholds.py

Runs fuzzy matching and embedding matching across a RANGE of thresholds,
not just one, and records precision/recall/F1 at each point.

Why this matters: reporting a single number (e.g. "fuzzy matching got 97%
precision") could be a lucky pick of threshold. Sweeping the full range and
plotting it shows whether one strategy is *genuinely* better across the
board, or if the earlier result was a coincidence of the exact threshold
chosen. This is the standard way precision/recall tradeoffs are reported
in real record-linkage and information-retrieval research.

Outputs:
  - sweep_results.csv   (every strategy/threshold/precision/recall/f1 row)
  - tradeoff_curve.png  (visual precision-recall curve for both strategies)
"""

import pandas as pd
import matplotlib.pyplot as plt

from evaluate import evaluate
from strategy_fuzzy import match_fuzzy
from strategy_exact import match_exact


def sweep_fuzzy(df, thresholds):
    rows = []
    for t in thresholds:
        result_df = match_fuzzy(df.copy(), threshold=t)
        scores = evaluate(result_df)
        scores["strategy"] = "fuzzy"
        scores["threshold"] = t
        rows.append(scores)
        print(f"  fuzzy @ threshold={t}: precision={scores['precision']:.3f}, recall={scores['recall']:.3f}")
    return rows


def sweep_embedding(df, thresholds, model):
    from strategy_embedding import match_embedding
    rows = []
    for t in thresholds:
        result_df = match_embedding(df.copy(), threshold=t, model=model)
        scores = evaluate(result_df)
        scores["strategy"] = "embedding"
        scores["threshold"] = t
        rows.append(scores)
        print(f"  embedding @ threshold={t}: precision={scores['precision']:.3f}, recall={scores['recall']:.3f}")
    return rows


def plot_tradeoff(results_df, out_path="tradeoff_curve.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    for strategy, group in results_df.groupby("strategy"):
        group = group.sort_values("recall")
        ax.plot(group["recall"], group["precision"], marker="o", label=strategy, linewidth=2)
        for _, row in group.iterrows():
            ax.annotate(f"{row['threshold']}", (row["recall"], row["precision"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    # mark the exact-match baseline as a single reference point
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, label="exact match precision (100%)")

    ax.set_xlabel("Recall (% of true duplicates caught)")
    ax.set_ylabel("Precision (% of predicted matches that were correct)")
    ax.set_title("Precision vs Recall Tradeoff Across Matching Strategies")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved chart to {out_path}")


if __name__ == "__main__":
    df = pd.read_csv("people_raw.csv")

    fuzzy_thresholds = [70, 75, 80, 85, 90, 95]
    embedding_thresholds = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    print("Sweeping fuzzy matching thresholds...")
    fuzzy_rows = sweep_fuzzy(df, fuzzy_thresholds)

    print("\nLoading embedding model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Sweeping embedding matching thresholds...")
    embedding_rows = sweep_embedding(df, embedding_thresholds, model)

    all_rows = fuzzy_rows + embedding_rows
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv("sweep_results.csv", index=False)
    print("\nSaved full results to sweep_results.csv")

    plot_tradeoff(results_df)

    print("\n" + "=" * 60)
    print("BEST F1 SCORE PER STRATEGY")
    print("=" * 60)
    best_per_strategy = results_df.loc[results_df.groupby("strategy")["f1"].idxmax()]
    print(best_per_strategy[["strategy", "threshold", "precision", "recall", "f1"]].to_string(index=False))