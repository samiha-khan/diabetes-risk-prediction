"""
Demo: pick a few test-set patients and generate grounded explanations.

Run with ANTHROPIC_API_KEY set to get real LLM phrasing; without it, the
same code path produces a deterministic template so the demo never
breaks in front of someone.
"""

import pickle
from pathlib import Path

from src.evaluate import evaluate_on_test
from src.explain import explain_instance, global_importance, llm_explanation
from src.pipeline import load_and_split

ROOT = Path(__file__).resolve().parent.parent


def main():
    with open(ROOT / "models" / "ensemble.pkl", "rb") as f:
        d = pickle.load(f)
    ensemble = d["ensemble"]

    train, val, test, encoders = load_and_split(str(ROOT / "data" / "diabetes_prediction_dataset.csv"))

    print("=== Global feature importance (mean |SHAP|, sampled) ===")
    gi = global_importance(ensemble, test.X)
    for feat, val_ in list(gi.items())[:8]:
        print(f"  {feat:<24} {val_:.4f}")

    print("\n=== Per-patient explanations ===")
    prob = ensemble.predict_proba(test.X)
    # pick: one high-risk true-positive, one low-risk true-negative, one false-negative (missed case)
    pred = (prob >= ensemble.weights.threshold).astype(int)
    import numpy as np
    y = test.y.values
    tp_idx = np.where((pred == 1) & (y == 1))[0][0]
    tn_idx = np.where((pred == 0) & (y == 0))[0][0]
    fn_idx = np.where((pred == 0) & (y == 1))[0][0]

    for label, idx in [("TRUE POSITIVE (correctly flagged)", tp_idx),
                        ("TRUE NEGATIVE (correctly cleared)", tn_idx),
                        ("FALSE NEGATIVE (missed case)", fn_idx)]:
        row = test.X.iloc[[idx]]
        contributions = explain_instance(ensemble, row)
        risk = float(prob[idx])
        explanation = llm_explanation(contributions, risk)
        print(f"\n--- {label} ---")
        print(f"Risk score: {risk:.1%}  (actual label: {'diabetic' if y[idx] == 1 else 'not diabetic'})")
        print(f"Explanation: {explanation}")


if __name__ == "__main__":
    main()
