"""
Prediction-time preprocessing for the loan eligibility endpoint.

The model behind the endpoint was trained on data that went through the
Canvas data flow. The endpoint does NOT run that data flow for you, so a raw
application like {"annual_income": "$85,000"} must be cleaned the same way
before it is sent.

To guarantee identical logic, this module executes the very same custom
transform files that were pasted into Canvas (canvas_transforms/*.py)
instead of re-implementing them.
"""

import csv
import io
import os

import numpy as np
import pandas as pd

TRANSFORM_DIR = os.environ.get(
    "TRANSFORM_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "canvas_transforms"),
)
TRANSFORM_FILES = ["01_clean_columns.py", "02_add_debt_to_income.py"]

# Column order of the training dataset WITHOUT the target column (loan_status)
# and without applicant_id (dropped in Canvas). The endpoint expects CSV values
# in exactly this order. Confirm it on the deployment's "Test deployment" tab
# in Canvas, and update this list if your data flow produced a different order.
FEATURE_ORDER = [
    "age",
    "employment_status",
    "annual_income",
    "credit_score",
    "existing_debt",
    "loan_amount",
    "employment_length",
    "loan_purpose",
    "debt_to_income",
]

RAW_INPUT_FIELDS = [c for c in FEATURE_ORDER if c != "debt_to_income"]


def _run_transform(df, filename):
    path = os.path.join(TRANSFORM_DIR, filename)
    with open(path, encoding="utf-8") as fh:
        code = fh.read()
    scope = {"df": df}
    exec(compile(code, path, "exec"), scope)  # trusted, version-controlled code
    return scope["df"]


def transform(df):
    """Apply every Canvas custom transform, in order."""
    df = df.copy()
    for name in TRANSFORM_FILES:
        df = _run_transform(df, name)
    return df


def applications_to_csv(applications):
    """
    applications: list of dicts with raw (possibly messy) field values, e.g.
        {"age": "34", "annual_income": "$85,000", ...}
    Returns a header-less CSV string ready for invoke_endpoint.
    """
    raw = pd.DataFrame(applications)
    for col in RAW_INPUT_FIELDS:
        if col not in raw.columns:
            raw[col] = np.nan          # missing field -> empty value
    raw = raw[RAW_INPUT_FIELDS].astype(object)

    features = transform(raw)[FEATURE_ORDER]

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for row in features.itertuples(index=False):
        writer.writerow(["" if pd.isna(v) else v for v in row])
    return buf.getvalue()
