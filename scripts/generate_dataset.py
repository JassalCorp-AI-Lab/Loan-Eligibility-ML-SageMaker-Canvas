"""
Generate a synthetic, deliberately messy loan application dataset.

Each row is one past loan application with a known outcome (loan_status).
Values are written as raw strings with inconsistent formats, missing values,
invalid entries and duplicate rows, so the data has to be cleaned before
training -- just like a real export from an operational system.

The data is SYNTHETIC. Approval outcomes come from a hidden scoring rule plus
random noise (see `approval_probability`). It is meant for learning, not for
making real lending decisions.

Usage:
    python scripts/generate_dataset.py            # writes data/loan_applications_raw.csv
    python scripts/generate_dataset.py --rows 1000 --seed 7 --out /tmp/sample.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

STATUSES = ["Salaried", "Self-Employed", "Contract", "Unemployed", "Retired"]
STATUS_P = [0.60, 0.17, 0.08, 0.07, 0.08]

PURPOSES = ["Debt Consolidation", "Home Improvement", "Car", "Education",
            "Medical", "Business", "Other"]
PURPOSE_P = [0.30, 0.18, 0.18, 0.10, 0.08, 0.08, 0.08]


# --------------------------------------------------------------------------
# 1. Generate CLEAN values
# --------------------------------------------------------------------------
def generate_clean(n, rng):
    status = rng.choice(STATUSES, size=n, p=STATUS_P)

    age = np.clip(rng.normal(39, 11, n), 21, 75).round().astype(int)
    age[status == "Retired"] = rng.integers(58, 80, (status == "Retired").sum())

    # Income depends on employment status (log-normal, rounded to $500)
    median_income = {"Salaried": 68000, "Self-Employed": 62000, "Contract": 55000,
                     "Unemployed": 14000, "Retired": 38000}
    spread = {"Salaried": 0.45, "Self-Employed": 0.70, "Contract": 0.50,
              "Unemployed": 0.50, "Retired": 0.40}
    income = np.array([rng.lognormal(np.log(median_income[s]), spread[s]) for s in status])
    income = (np.clip(income, 6000, 600000) / 500).round() * 500

    # Years at current job (0 for unemployed / retired)
    max_years = np.maximum(age - 20, 0)
    emp_years = np.minimum(rng.exponential(5.5, n), max_years).round(1)
    emp_years = np.where(emp_years < 1, np.round(emp_years * 12) / 12, np.round(emp_years))
    emp_years[np.isin(status, ["Unemployed", "Retired"])] = 0

    credit = np.clip(rng.normal(680, 75, n), 300, 850).round().astype(int)

    # Existing debt as a share of income; about 12% have no debt at all
    dti = rng.beta(2.0, 5.0, n) * 1.1
    debt = (income * dti / 100).round() * 100
    debt[rng.random(n) < 0.12] = 0

    purpose = rng.choice(PURPOSES, size=n, p=PURPOSE_P)
    base_loan = {"Debt Consolidation": 18000, "Home Improvement": 25000, "Car": 22000,
                 "Education": 15000, "Medical": 8000, "Business": 40000, "Other": 10000}
    loan = np.array([rng.lognormal(np.log(base_loan[p]), 0.55) for p in purpose])
    loan = (np.clip(loan, 1000, 150000) / 500).round() * 500

    df = pd.DataFrame({
        "age": age, "employment_status": status, "annual_income": income,
        "credit_score": credit, "existing_debt": debt, "loan_amount": loan,
        "employment_length": emp_years, "loan_purpose": purpose,
    })
    return df


def approval_probability(df, rng):
    """Hidden 'bank policy' the model should rediscover."""
    dti = df["existing_debt"] / df["annual_income"]
    lti = df["loan_amount"] / df["annual_income"]
    status_effect = df["employment_status"].map({
        "Salaried": 0.4, "Self-Employed": -0.2, "Contract": 0.0,
        "Unemployed": -1.8, "Retired": 0.1})
    purpose_effect = df["loan_purpose"].map({
        "Debt Consolidation": 0.0, "Home Improvement": 0.2, "Car": 0.2,
        "Education": 0.1, "Medical": -0.1, "Business": -0.4, "Other": -0.2})
    z = (0.7
         + 0.022 * (df["credit_score"] - 680)
         - 5.5 * (dti - 0.30)
         - 2.2 * (lti - 0.35)
         + 0.07 * np.minimum(df["employment_length"], 10)
         + status_effect + purpose_effect
         + rng.normal(0, 0.6, len(df)))          # unexplained noise
    return 1 / (1 + np.exp(-z))


# --------------------------------------------------------------------------
# 2. Render values as MESSY strings
# --------------------------------------------------------------------------
def pick(rng, options, probs):
    return options[rng.choice(len(options), p=probs)]


def fmt_money(v, rng, allow_k=True):
    styles = ["dollar", "plain", "k", "usd", "comma", "blank"]
    p = [0.40, 0.25, 0.10, 0.10, 0.10, 0.05] if allow_k else [0.45, 0.30, 0.0, 0.10, 0.12, 0.03]
    s = pick(rng, styles, p)
    if s == "dollar":
        return f"${v:,.0f}"
    if s == "plain":
        return f"{v:.0f}"
    if s == "k":
        k = v / 1000
        return f"{k:g}{rng.choice(['k', 'K'])}"
    if s == "usd":
        return f"USD {v:,.2f}"
    if s == "comma":
        return f"{v:,.0f}"
    return ""


def fmt_debt(v, rng):
    if v == 0:
        return pick(rng, ["0", "$0", "none", "None", ""], [0.4, 0.2, 0.2, 0.1, 0.1])
    return fmt_money(v, rng)


def fmt_loan(v, rng):
    s = pick(rng, ["dollar", "plain", "k", "decimal", "blank"], [0.45, 0.30, 0.12, 0.12, 0.01])
    if s == "dollar":
        return f"${v:,.0f}"
    if s == "plain":
        return f"{v:.0f}"
    if s == "k":
        return f"{v / 1000:g}{rng.choice(['k', 'K'])}"
    if s == "decimal":
        return f"{v:,.2f}"
    return ""


def fmt_age(v, rng):
    s = pick(rng, ["plain", "years", "padded", "blank", "invalid"], [0.80, 0.08, 0.05, 0.065, 0.005])
    if s == "plain":
        return str(v)
    if s == "years":
        return f"{v} years"
    if s == "padded":
        return f" {v} "
    if s == "blank":
        return ""
    return str(rng.choice(["-1", "230", "999", "0"]))


def fmt_credit(v, rng):
    s = pick(rng, ["plain", "decimal", "unknown", "blank", "invalid"], [0.85, 0.07, 0.03, 0.045, 0.005])
    if s == "plain":
        return str(v)
    if s == "decimal":
        return f"{v}.0"
    if s == "unknown":
        return str(rng.choice(["Unknown", "N/A", "unknown"]))
    if s == "blank":
        return ""
    return str(rng.choice(["9999", "0", "1200"]))


def fmt_emp_status(v, rng):
    variants = {
        "Salaried": ["Salaried", "salaried", "SALARIED", "Salaried "],
        "Self-Employed": ["Self-Employed", "self employed", "Self employed", "SELF-EMPLOYED", "self_employed"],
        "Contract": ["Contract", "contract", "Contractor", "CONTRACT"],
        "Unemployed": ["Unemployed", "unemployed", "UNEMPLOYED"],
        "Retired": ["Retired", "retired", " Retired"],
    }
    if rng.random() < 0.03:
        return ""
    opts = variants[v]
    p = [0.6] + [0.4 / (len(opts) - 1)] * (len(opts) - 1)
    return pick(rng, opts, p)


def fmt_emp_length(v, status, rng):
    if rng.random() < 0.04:
        return ""
    if status in ("Unemployed", "Retired"):
        return pick(rng, ["0", "N/A", "0 years", ""], [0.4, 0.3, 0.2, 0.1])
    if v < 1:
        months = int(round(v * 12))
        return pick(rng, ["< 1 year", f"{months} months", "0"], [0.5, 0.4, 0.1]) if months else "< 1 year"
    years = int(v)
    if years >= 10 and rng.random() < 0.5:
        return "10+ years"
    return pick(rng, [f"{years} years", f"{years} yrs", f"{years}", f"{years} Years"],
                [0.45, 0.2, 0.25, 0.10])


def fmt_purpose(v, rng):
    variants = {
        "Debt Consolidation": ["Debt Consolidation", "debt consolidation", "debt_consolidation", "DEBT CONSOLIDATION"],
        "Home Improvement": ["Home Improvement", "home improvement", "home_improvement", "Home improvement "],
        "Car": ["Car", "car", "Auto", "auto loan"],
        "Education": ["Education", "education", "EDUCATION"],
        "Medical": ["Medical", "medical", "MEDICAL"],
        "Business": ["Business", "business", "Small Business"],
        "Other": ["Other", "other", "OTHER"],
    }
    if rng.random() < 0.02:
        return ""
    opts = variants[v]
    p = [0.6] + [0.4 / (len(opts) - 1)] * (len(opts) - 1)
    return pick(rng, opts, p)


def fmt_status(approved, rng):
    if rng.random() < 0.005:
        return ""
    if approved:
        return pick(rng, ["Approved", "approved", "APPROVED", "Y"], [0.70, 0.10, 0.10, 0.10])
    return pick(rng, ["Denied", "denied", "Rejected", "N"], [0.70, 0.10, 0.10, 0.10])


def make_messy(clean, approved, rng):
    rows = []
    for i, r in enumerate(clean.itertuples(index=False)):
        rows.append({
            "applicant_id": f"APP-{i + 1:07d}",
            "age": fmt_age(r.age, rng),
            "employment_status": fmt_emp_status(r.employment_status, rng),
            "annual_income": fmt_money(r.annual_income, rng),
            "credit_score": fmt_credit(r.credit_score, rng),
            "existing_debt": fmt_debt(r.existing_debt, rng),
            "loan_amount": fmt_loan(r.loan_amount, rng),
            "employment_length": fmt_emp_length(r.employment_length, r.employment_status, rng),
            "loan_purpose": fmt_purpose(r.loan_purpose, rng),
            "loan_status": fmt_status(approved[i], rng),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50000, help="total rows incl. duplicates")
    ap.add_argument("--dup-rate", type=float, default=0.01, help="share of rows that are exact duplicates")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data",
                                                  "loan_applications_raw.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_dupes = int(args.rows * args.dup_rate)
    n_unique = args.rows - n_dupes

    clean = generate_clean(n_unique, rng)
    approved = rng.random(n_unique) < approval_probability(clean, rng)
    messy = make_messy(clean, approved, rng)

    # Exact duplicate rows (same applicant exported twice), shuffled in
    dupes = messy.sample(n=n_dupes, random_state=args.seed)
    messy = pd.concat([messy, dupes]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    messy.to_csv(args.out, index=False)
    print(f"Wrote {len(messy):,} rows ({n_dupes:,} duplicates) to {os.path.abspath(args.out)}")
    print(f"Approval rate (clean labels): {approved.mean():.1%}")


if __name__ == "__main__":
    main()
