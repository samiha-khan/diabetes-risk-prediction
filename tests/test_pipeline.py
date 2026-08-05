import pandas as pd
import pytest

from src.pipeline import fit_encoders, load_and_split


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    # Small synthetic dataset so tests run fast and don't depend on the
    # real 100k-row CSV being present.
    import numpy as np
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], n),
        "age": rng.uniform(1, 90, n),
        "hypertension": rng.integers(0, 2, n),
        "heart_disease": rng.integers(0, 2, n),
        "smoking_history": rng.choice(["never", "current", "former"], n),
        "bmi": rng.uniform(15, 45, n),
        "HbA1c_level": rng.uniform(3.5, 9.0, n),
        "blood_glucose_level": rng.integers(70, 300, n),
        "diabetes": rng.choice([0, 1], n, p=[0.85, 0.15]),
    })
    path = tmp_path_factory.mktemp("data") / "synthetic.csv"
    df.to_csv(path, index=False)
    return load_and_split(str(path), test_size=0.15, val_size=0.15, random_state=0)


def test_splits_are_disjoint(splits):
    train, val, test, _ = splits
    train_idx = set(train.X.index)
    val_idx = set(val.X.index)
    test_idx = set(test.X.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_split_sizes_roughly_match_requested_fractions(splits):
    train, val, test, _ = splits
    total = len(train.X) + len(val.X) + len(test.X)
    assert abs(len(test.X) / total - 0.15) < 0.02
    assert abs(len(val.X) / total - 0.15) < 0.02


def test_stratification_preserves_positive_rate(splits):
    train, val, test, _ = splits
    overall_rate = (train.y.sum() + val.y.sum() + test.y.sum()) / (
        len(train.y) + len(val.y) + len(test.y)
    )
    for split in (train, val, test):
        assert abs(split.y.mean() - overall_rate) < 0.05


def test_encoders_are_fit_on_train_only_not_full_data():
    """Regression test for the leakage class of bug: encoders must be
    derived from the train split's categories, not from val/test, so a
    category that only appears in val/test is detectable as unseen
    rather than silently having influenced the encoding.
    """
    train_df = pd.DataFrame({
        "gender": ["Male", "Female"],
        "smoking_history": ["never", "current"],
    })
    encoders = fit_encoders(train_df)
    assert set(encoders.gender_map.keys()) == {"Male", "Female"}
    assert set(encoders.smoking_map.keys()) == {"never", "current"}
    # "former" appears nowhere in train -> must not be in the fitted map
    assert "former" not in encoders.smoking_map
