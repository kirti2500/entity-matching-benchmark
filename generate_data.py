"""
generate_data.py

Builds a synthetic dataset of "people" records with KNOWN duplicates,
so we can measure precision/recall of different matching strategies
against ground truth we control.

How it works:
1. Create N "true" people (each gets a unique person_id).
2. For each true person, generate 1-3 "noisy variants" — records that
   represent the SAME person but with realistic messiness: typos,
   nickname swaps, phone format changes, missing fields.
3. Shuffle everything together into one flat table, WITHOUT the
   person_id visible to any matching algorithm (that's the whole
   point — the algorithm has to figure it out; person_id is only used
   afterward to score how well it did).
"""

import random
import pandas as pd

random.seed(42)  # reproducible — anyone re-running this gets the same data

FIRST_NAMES = ["Aarav", "Isha", "Rohan", "Priya", "Karan", "Neha", "Arjun",
               "Divya", "Vikram", "Tanvi", "Sahil", "Meera", "Rahul", "Anjali"]
LAST_NAMES = ["Sharma", "Gupta", "Mehta", "Verma", "Kapoor", "Chopra",
              "Malhotra", "Nair", "Reddy", "Bhatia", "Singh", "Agarwal"]
CITIES = ["Delhi", "Mumbai", "Bengaluru", "Pune", "Gurugram", "Noida"]

NICKNAME_MAP = {
    "Rohan": "Ro", "Priya": "Pri", "Vikram": "Vicky", "Arjun": "Aj",
    "Anjali": "Anji", "Rahul": "Rah",
}


def random_phone():
    return "9" + "".join(str(random.randint(0, 9)) for _ in range(9))


def messy_phone_format(phone):
    """Simulates the same real-world formatting inconsistencies as last project."""
    fmt = random.choice(["plain", "plus91", "zero_prefix", "dashed"])
    if fmt == "plain":
        return phone
    if fmt == "plus91":
        return "+91" + phone
    if fmt == "zero_prefix":
        return "0" + phone
    return "+91-" + phone


def typo(name):
    """Introduce a small, realistic typo into a name."""
    if len(name) < 3:
        return name
    i = random.randint(1, len(name) - 2)
    return name[:i] + name[i + 1] + name[i] + name[i + 2:]  # swap two adjacent letters


def make_variant(first, last, phone, city, is_first_copy):
    """Generate one noisy but same-person record."""
    f, l, p, c = first, last, phone, city

    if not is_first_copy:
        # apply 1-2 random distortions so variants aren't all identical
        distortions = random.sample(
            ["nickname", "typo_first", "typo_last", "case", "city_case", "drop_last"],
            k=random.randint(1, 2)
        )
        if "nickname" in distortions and first in NICKNAME_MAP:
            f = NICKNAME_MAP[first]
        if "typo_first" in distortions:
            f = typo(f)
        if "typo_last" in distortions:
            l = typo(l)
        if "case" in distortions:
            f, l = f.upper(), l.lower()
        if "city_case" in distortions:
            c = c.upper()
        if "drop_last" in distortions and random.random() < 0.3:
            l = ""  # simulate a record with a missing last name

    p = messy_phone_format(p)
    return f, l, p, c


def generate_dataset(n_people=200, max_variants=3, out_path="people_raw.csv"):
    rows = []
    record_id = 0

    for person_id in range(n_people):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        true_phone = random_phone()
        city = random.choice(CITIES)

        n_variants = random.randint(1, max_variants)
        for v in range(n_variants):
            f, l, p, c = make_variant(first, last, true_phone, city, is_first_copy=(v == 0))
            rows.append({
                "record_id": record_id,
                "person_id": person_id,   # GROUND TRUTH — hidden from matchers, used only for scoring
                "full_name": f"{f} {l}".strip(),
                "phone": p,
                "city": c,
            })
            record_id += 1

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle order
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(f"Generated {len(df)} records for {df['person_id'].nunique()} unique people")
    print(f"Records per person — min: {df.groupby('person_id').size().min()}, "
          f"max: {df.groupby('person_id').size().max()}, "
          f"avg: {df.groupby('person_id').size().mean():.2f}")
    print("\nSample rows:")
    print(df.head(10).to_string(index=False))
