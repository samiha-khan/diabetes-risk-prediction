"""Train/val/test split with an explicit boundary — this is where the
original leakage bug got fixed.

Old version: ensemble weights and threshold were tuned by scanning for
whatever maximized accuracy on the test set. So "test" metrics weren't a
real generalization estimate.

Fix: three-way split. train = fit models + encoders. val = tune weights
and threshold. test = touched once, for final numbers only.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_clinical_features,
    clean,
    encode_categoricals,
)


@dataclass
class Split:
    X: pd.DataFrame
    y: pd.Series


@dataclass
class Encoders:
    gender_map: dict
    smoking_map: dict


def fit_encoders(train_df: pd.DataFrame) -> Encoders:
    """Fit on TRAIN only. A category seen in val/test but not train maps
    to NaN instead of silently getting guessed."""
    gender_map = {v: i for i, v in enumerate(sorted(train_df["gender"].unique()))}
    smoking_map = {v: i for i, v in enumerate(sorted(train_df["smoking_history"].unique()))}
    return Encoders(gender_map=gender_map, smoking_map=smoking_map)


def _prepare(df: pd.DataFrame, encoders: Encoders) -> pd.DataFrame:
    df = add_clinical_features(df)
    df = encode_categoricals(df, encoders.gender_map, encoders.smoking_map)
    return df


def load_and_split(
    csv_path: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[Split, Split, Split, Encoders]:
    """val_size/test_size are fractions of the full dataset, not of the
    train remainder — 0.15/0.15 leaves 70% for train. Stratified on the
    target since 8.5% positive is small enough that an unstratified split
    can shift class balance by chance."""
    raw = pd.read_csv(csv_path)
    raw = clean(raw)

    train_val_df, test_df = train_test_split(
        raw, test_size=test_size, random_state=random_state, stratify=raw[TARGET_COLUMN]
    )
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_df[TARGET_COLUMN],
    )

    encoders = fit_encoders(train_df)

    def to_split(df: pd.DataFrame) -> Split:
        prepared = _prepare(df, encoders)
        return Split(X=prepared[FEATURE_COLUMNS], y=prepared[TARGET_COLUMN])

    train_split = to_split(train_df)
    val_split = to_split(val_df)
    test_split = to_split(test_df)

    print(
        f"[split] train={len(train_split.X)} "
        f"val={len(val_split.X)} test={len(test_split.X)} "
        f"(positive rate train/val/test = "
        f"{train_split.y.mean():.4f}/{val_split.y.mean():.4f}/{test_split.y.mean():.4f})"
    )

    return train_split, val_split, test_split, encoders
