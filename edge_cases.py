"""
edge_cases.py

Adversarial test suite: specific, hand-crafted tricky scenarios with a
KNOWN correct answer, tested against each strategy individually.

Aggregate precision/recall (from evaluate.py and multi_seed_eval.py) tells
you overall performance, but can hide specific systematic weaknesses. This
file tests deliberately tricky cases one at a time, so failures are
individually visible and explainable - not just averaged away.

Each test case defines what SHOULD happen (should_match: True/False) and
we check whether each strategy agrees.
"""

import pandas as pd
from strategy_exact import match_exact
from strategy_fuzzy import match_fuzzy


TEST_CASES = [
    {
        "name": "Identical strings",
        "description": "Exact same name and phone - trivial case, everything should catch this.",
        "records": [
            {"record_id": 1, "full_name": "Rohan Sharma", "phone": "9000000001", "city": "Delhi"},
            {"record_id": 2, "full_name": "Rohan Sharma", "phone": "9000000001", "city": "Delhi"},
        ],
        "should_match": True,
    },
    {
        "name": "Same name, DIFFERENT phone (different people)",
        "description": "Two genuinely different people who happen to share a common name. "
                        "A system that merges these is dangerous - this is the false-merge risk.",
        "records": [
            {"record_id": 3, "full_name": "Priya Sharma", "phone": "9000000002", "city": "Mumbai"},
            {"record_id": 4, "full_name": "Priya Sharma", "phone": "9111111111", "city": "Pune"},
        ],
        "should_match": False,
    },
    {
        "name": "Phone formatting difference only",
        "description": "Same person, same name, phone differs only in formatting "
                        "(+91 prefix). Should still match after normalization.",
        "records": [
            {"record_id": 5, "full_name": "Arjun Mehta", "phone": "9000000003", "city": "Noida"},
            {"record_id": 6, "full_name": "Arjun Mehta", "phone": "+919000000003", "city": "Noida"},
        ],
        "should_match": True,
    },
    {
        "name": "Single-character typo",
        "description": "Realistic typo - one letter swapped. Should match via fuzzy, "
                        "will NOT match via exact.",
        "records": [
            {"record_id": 7, "full_name": "Vikram Singh", "phone": "9000000004", "city": "Gurugram"},
            {"record_id": 8, "full_name": "Vikarm Singh", "phone": "9000000004", "city": "Gurugram"},
        ],
        "should_match": True,
    },
    {
        "name": "Missing/empty last name",
        "description": "One record has an empty last name field - common real-world "
                        "data quality issue. Should still link on first name + phone.",
        "records": [
            {"record_id": 9, "full_name": "Tanvi Gupta", "phone": "9000000005", "city": "Bengaluru"},
            {"record_id": 10, "full_name": "Tanvi", "phone": "9000000005", "city": "Bengaluru"},
        ],
        "should_match": True,
    },
    {
        "name": "Completely unrelated people",
        "description": "Two records with nothing in common. Should never match, "
                        "under any strategy.",
        "records": [
            {"record_id": 11, "full_name": "Karan Bhatia", "phone": "9222222222", "city": "Delhi"},
            {"record_id": 12, "full_name": "Sneha Reddy", "phone": "9333333333", "city": "Pune"},
        ],
        "should_match": False,
    },
    {
        "name": "Short name / initials",
        "description": "Name shortened to initial - common in casual data entry. "
                        "Genuinely hard case; not all strategies are expected to catch it.",
        "records": [
            {"record_id": 13, "full_name": "R. Verma", "phone": "9000000006", "city": "Noida"},
            {"record_id": 14, "full_name": "Rohit Verma", "phone": "9000000006", "city": "Noida"},
        ],
        "should_match": True,
    },
]


def run_strategy_on_case(match_fn, records):
    df = pd.DataFrame(records)
    result = match_fn(df)
    clusters = result["predicted_cluster"].tolist()
    return clusters[0] == clusters[1]  # True if both records ended in the same group


def run_edge_case_suite():
    strategies = {
        "exact": lambda df: match_exact(df),
        "fuzzy (t=85)": lambda df: match_fuzzy(df, threshold=85),
    }

    results = []
    for case in TEST_CASES:
        row = {"case": case["name"], "expected": case["should_match"]}
        for strat_name, strat_fn in strategies.items():
            predicted = run_strategy_on_case(strat_fn, case["records"])
            correct = predicted == case["should_match"]
            row[strat_name] = "✓" if correct else "✗ WRONG"
        results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 70)
    print("EDGE CASE TEST SUITE")
    print("=" * 70)

    for case in TEST_CASES:
        print(f"\n{case['name']}")
        print(f"  {case['description']}")
        print(f"  Expected: {'SHOULD match' if case['should_match'] else 'should NOT match'}")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    results_df = run_edge_case_suite()
    print(results_df.to_string(index=False))

    results_df.to_csv("edge_case_results.csv", index=False)
    print("\nSaved to edge_case_results.csv")