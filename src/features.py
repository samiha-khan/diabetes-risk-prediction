"""Feature engineering for diabetes risk prediction.

Clinical thresholds only, no dataset-derived stats — safe to apply
identically across train/val/test with zero leakage risk.
"""

import pandas as pd

HBA1C_DIABETIC_THRESHOLD = 6.5
GLUCOSE_DIABETIC_THRESHOLD = 140  # mg/dL, random plasma glucose (ADA)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows labeled gender='Other' — too few (~18/100k) to trust
    for training or a subgroup check, so dropped and logged instead of
    silently folded into another category."""
    out = df.copy()
    n_before = len(out)
    out = out[out["gender"] != "Other"].reset_index(drop=True)
    n_dropped = n_before - len(out)
    if n_dropped:
        print(f"[clean] dropped {n_dropped} rows with gender='Other' "
              f"({n_dropped / n_before:.3%} of data)")
    return out


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["high_glucose"] = (out["blood_glucose_level"] > GLUCOSE_DIABETIC_THRESHOLD).astype(int)
    out["high_hba1c"] = (out["HbA1c_level"] > HBA1C_DIABETIC_THRESHOLD).astype(int)
    out["both_high"] = out["high_glucose"] * out["high_hba1c"]
    out["comorbidity_score"] = out["hypertension"] + out["heart_disease"]
    out["age_bmi_interaction"] = out["age"] * out["bmi"]
    return out


def encode_categoricals(df: pd.DataFrame, gender_map: dict, smoking_map: dict) -> pd.DataFrame:
    """Applies already-fitted maps. Fit these on train only
    (see pipeline.fit_encoders) — this function never fits anything."""
    out = df.copy()
    out["gender"] = out["gender"].map(gender_map)
    out["smoking_history"] = out["smoking_history"].map(smoking_map)
    return out


FEATURE_COLUMNS = [
    "gender", "age", "hypertension", "heart_disease", "smoking_history",
    "bmi", "HbA1c_level", "blood_glucose_level",
    "high_glucose", "high_hba1c", "both_high", "comorbidity_score",
    "age_bmi_interaction",
]

TARGET_COLUMN = "diabetes"
