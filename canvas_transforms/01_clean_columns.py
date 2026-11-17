# =============================================================================
# SageMaker Canvas / Data Wrangler custom transform -- Python (Pandas)
# Step: "Clean and standardize raw columns"
#
# Paste this whole file into: Add transform > Custom transform > Python (Pandas)
# Canvas gives you the dataframe as `df`; whatever is left in `df` at the end
# becomes the output of this step.
#
# This SAME file is executed by app/preprocess.py at prediction time, so the
# model sees identically cleaned data in training and in production
# (no training-serving skew). Edit it here, and both paths pick up the change.
#
# Written for pandas 1.0.x (the version Canvas custom transforms run on), so it
# avoids newer pandas APIs.
#
# Missing values: text columns use None (not NaN), so Canvas stores them as
# real nulls instead of a literal "NaN" category. Rows with no label are
# dropped inside this script, so the target only ever contains
# "Approved" / "Denied".
# =============================================================================
import re

import numpy as np
import pandas as pd

MISSING_TOKENS = {"", "nan", "none", "null", "n/a", "na", "unknown", "-", "?"}


def _norm(value):
    """Lower-case, trim, and turn '-' / '_' into spaces. Returns None if missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = re.sub(r"[-_]+", " ", str(value).strip().lower())
    s = re.sub(r"\s+", " ", s)
    return None if s in MISSING_TOKENS else s


def _first_number(s):
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


def parse_money(value, none_means_zero=False):
    """'$85,000' | '85000' | '85k' | 'USD 85,000.00' -> 85000.0"""
    if none_means_zero and str(value).strip().lower() == "none":
        return 0.0
    s = _norm(value)
    if s is None:
        return np.nan
    s = s.replace("usd", "").replace("$", "").replace(",", "").replace(" ", "")
    multiplier = 1000.0 if s.endswith("k") else 1.0
    s = s.rstrip("k")
    try:
        return float(s) * multiplier
    except ValueError:
        return np.nan


def parse_age(value):
    s = _norm(value)
    n = np.nan if s is None else _first_number(s)
    return n if 18 <= n <= 100 else np.nan


def parse_credit_score(value):
    s = _norm(value)
    n = np.nan if s is None else _first_number(s)
    return n if 300 <= n <= 850 else np.nan


def parse_employment_length(value):
    """'5 years' | '5 yrs' | '10+ years' | '< 1 year' | '6 months' -> years (float)"""
    s = _norm(value)
    if s is None:
        return np.nan
    if s.startswith("<"):
        return 0.5
    n = _first_number(s)
    if "month" in s:
        return round(n / 12.0, 2)
    return n


EMPLOYMENT_STATUS_MAP = {
    "salaried": "Salaried",
    "self employed": "Self-Employed",
    "contract": "Contract",
    "contractor": "Contract",
    "unemployed": "Unemployed",
    "retired": "Retired",
}

LOAN_PURPOSE_MAP = {
    "debt consolidation": "Debt Consolidation",
    "home improvement": "Home Improvement",
    "car": "Car",
    "auto": "Car",
    "auto loan": "Car",
    "education": "Education",
    "medical": "Medical",
    "business": "Business",
    "small business": "Business",
    "other": "Other",
}

LOAN_STATUS_MAP = {
    "approved": "Approved", "y": "Approved", "yes": "Approved",
    "denied": "Denied", "rejected": "Denied", "n": "Denied", "no": "Denied",
}


def _map_category(value, mapping, unknown=None):
    s = _norm(value)
    if s is None:
        return None
    return mapping.get(s, unknown)


def _as_text_column(series):
    """Keep text columns as object dtype with None for missing values."""
    return [None if v is None or (isinstance(v, float) and np.isnan(v)) else v
            for v in series]


# ---- apply to the dataframe -------------------------------------------------
if "age" in df.columns:
    df["age"] = df["age"].map(parse_age)
if "employment_status" in df.columns:
    df["employment_status"] = df["employment_status"].map(
        lambda v: _map_category(v, EMPLOYMENT_STATUS_MAP))
if "annual_income" in df.columns:
    df["annual_income"] = df["annual_income"].map(parse_money)
if "credit_score" in df.columns:
    df["credit_score"] = df["credit_score"].map(parse_credit_score)
if "existing_debt" in df.columns:
    df["existing_debt"] = df["existing_debt"].map(lambda v: parse_money(v, none_means_zero=True))
if "loan_amount" in df.columns:
    df["loan_amount"] = df["loan_amount"].map(parse_money)
if "employment_length" in df.columns:
    df["employment_length"] = df["employment_length"].map(parse_employment_length)
if "loan_purpose" in df.columns:
    df["loan_purpose"] = df["loan_purpose"].map(
        lambda v: _map_category(v, LOAN_PURPOSE_MAP, unknown="Other"))
# The label only exists in training data; at prediction time this is skipped.
if "loan_status" in df.columns:
    df["loan_status"] = df["loan_status"].map(lambda v: _map_category(v, LOAN_STATUS_MAP))
    # A row without a known outcome can't be used for training: drop it here.
    df = df[df["loan_status"].isin(["Approved", "Denied"])].copy()

for _col in ("employment_status", "loan_purpose", "loan_status"):
    if _col in df.columns:
        df[_col] = _as_text_column(df[_col])