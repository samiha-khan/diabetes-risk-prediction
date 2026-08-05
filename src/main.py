"""
End-to-end run: load -> split -> train -> tune on val -> evaluate on test
(once) -> subgroup fairness check -> save report + models.
"""

import json
import pickle
from pathlib import Path

from src.evaluate import evaluate_on_test, subgroup_report
from src.pipeline import load_and_split
from src.train import fit_ensemble

ROOT = Path(__file__).resolve().parent.parent


def main():
    train, val, test, encoders = load_and_split(str(ROOT / "data" / "diabetes_prediction_dataset.csv"))

    ensemble = fit_ensemble(train, val)

    report = evaluate_on_test(ensemble, test)
    print("\n=== TEST SET RESULTS (touched once) ===")
    print(report.summary())

    # Subgroup check needs the raw (undropped-index) gender column aligned to test.X
    import pandas as pd

    raw = pd.read_csv(ROOT / "data" / "diabetes_prediction_dataset.csv")
    raw = raw[raw["gender"] != "Other"].reset_index(drop=True)
    test_gender = raw.loc[test.X.index, ["gender"]].reset_index(drop=True)
    test_reset = pd.concat(
        [test.X.reset_index(drop=True), test.y.reset_index(drop=True).rename("diabetes")], axis=1
    )
    from src.pipeline import Split as _Split
    test_for_subgroup = _Split(X=test_reset[test.X.columns], y=test_reset["diabetes"])
    subgroups = subgroup_report(ensemble, test_for_subgroup, test_gender)
    print("\n=== SUBGROUP (GENDER) FAIRNESS CHECK ===")
    print(json.dumps(subgroups, indent=2))

    (ROOT / "reports").mkdir(exist_ok=True)
    with open(ROOT / "reports" / "test_metrics.json", "w") as f:
        json.dump({
            "roc_auc": report.roc_auc, "pr_auc": report.pr_auc,
            "precision": report.precision, "recall": report.recall,
            "f1": report.f1, "brier": report.brier,
            "confusion_matrix": {"tn": report.tn, "fp": report.fp, "fn": report.fn, "tp": report.tp},
            "n_test": report.n, "positive_rate": report.positive_rate,
            "missed_case_rate": report.missed_case_rate(),
            "ensemble_weights": {
                "lgb": ensemble.weights.lgb, "xgb": ensemble.weights.xgb,
                "cat": ensemble.weights.cat, "threshold": ensemble.weights.threshold,
            },
            "subgroup_fairness": subgroups,
        }, f, indent=2)

    (ROOT / "models").mkdir(exist_ok=True)
    with open(ROOT / "models" / "ensemble.pkl", "wb") as f:
        pickle.dump({"ensemble": ensemble, "encoders": encoders}, f)

    print(f"\nSaved metrics -> reports/test_metrics.json")
    print(f"Saved model -> models/ensemble.pkl")


if __name__ == "__main__":
    main()
