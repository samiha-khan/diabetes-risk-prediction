import pandas as pd

from src.features import add_clinical_features, clean, encode_categoricals


def test_clean_drops_other_gender():
    df = pd.DataFrame({
        "gender": ["Male", "Female", "Other"],
        "diabetes": [0, 1, 0],
    })
    out = clean(df)
    assert "Other" not in out["gender"].values
    assert len(out) == 2


def test_high_glucose_flag_matches_clinical_threshold():
    df = pd.DataFrame({
        "blood_glucose_level": [139, 140, 141],
        "HbA1c_level": [6.5, 6.5, 6.5],
        "hypertension": [0, 0, 0],
        "heart_disease": [0, 0, 0],
        "age": [50, 50, 50],
        "bmi": [25, 25, 25],
    })
    out = add_clinical_features(df)
    assert list(out["high_glucose"]) == [0, 0, 1]


def test_high_hba1c_flag_matches_clinical_threshold():
    df = pd.DataFrame({
        "blood_glucose_level": [100, 100, 100],
        "HbA1c_level": [6.4, 6.5, 6.6],
        "hypertension": [0, 0, 0],
        "heart_disease": [0, 0, 0],
        "age": [50, 50, 50],
        "bmi": [25, 25, 25],
    })
    out = add_clinical_features(df)
    assert list(out["high_hba1c"]) == [0, 0, 1]


def test_encode_categoricals_unseen_category_maps_to_nan():
    df = pd.DataFrame({"gender": ["Male", "Female"], "smoking_history": ["never", "current"]})
    gender_map = {"Female": 0, "Male": 1}
    smoking_map = {"never": 0}  # "current" deliberately not in the fitted map
    out = encode_categoricals(df, gender_map, smoking_map)
    assert out["gender"].tolist() == [1, 0]
    assert out["smoking_history"].isna().sum() == 1
