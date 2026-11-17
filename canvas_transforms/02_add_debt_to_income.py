# =============================================================================
# SageMaker Canvas / Data Wrangler custom transform -- Python (Pandas)
# Step: "Feature engineering: debt_to_income"
#
# Paste into: Add transform > Custom transform > Python (Pandas)
# Run it AFTER 01_clean_columns.py, because it needs numeric columns.
#
# debt_to_income = existing_debt / annual_income
# Left empty (NaN) when either value is missing or income is 0, so the model
# treats it as "unknown" instead of seeing a fake number.
#
# Also executed by app/preprocess.py at prediction time.
# =============================================================================
import numpy as np

income = df["annual_income"].astype(float)
debt = df["existing_debt"].astype(float)
df["debt_to_income"] = np.where(income > 0, debt / income, np.nan)
df["debt_to_income"] = df["debt_to_income"].round(4)
