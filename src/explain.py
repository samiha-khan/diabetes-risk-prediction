"""SHAP for the numbers, LLM for the plain-language phrasing. The LLM
only ever sees already-computed SHAP values — feature, contribution,
direction — and is told to summarize those, nothing else. It can't
invent a clinical judgment that isn't backed by an actual attribution.
No API key -> deterministic template instead of failing."""

import os
from dataclasses import dataclass

import numpy as np
import shap

from src.train import TrainedEnsemble

FEATURE_LABELS = {
    "gender": "gender",
    "age": "age",
    "hypertension": "hypertension",
    "heart_disease": "heart disease history",
    "smoking_history": "smoking history",
    "bmi": "BMI",
    "HbA1c_level": "HbA1c level",
    "blood_glucose_level": "blood glucose level",
    "high_glucose": "elevated glucose flag",
    "high_hba1c": "elevated HbA1c flag",
    "both_high": "combined glucose+HbA1c flag",
    "comorbidity_score": "comorbidity score",
    "age_bmi_interaction": "age×BMI interaction",
}


@dataclass
class FeatureContribution:
    feature: str
    label: str
    value: float
    shap_value: float

    def direction(self) -> str:
        return "increased" if self.shap_value > 0 else "decreased"


def global_importance(ensemble: TrainedEnsemble, X_sample) -> dict:
    """Mean |SHAP value| per feature, LightGBM base model, sampled to 500
    rows for speed."""
    sample = X_sample.sample(min(500, len(X_sample)), random_state=42)
    explainer = shap.TreeExplainer(ensemble.lgb_model)
    shap_values = explainer.shap_values(sample)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs = np.abs(sv).mean(axis=0)
    return dict(sorted(zip(sample.columns, mean_abs), key=lambda kv: -kv[1]))


def explain_instance(ensemble: TrainedEnsemble, X_row) -> list[FeatureContribution]:
    """SHAP explanation for one patient row, sorted by impact."""
    explainer = shap.TreeExplainer(ensemble.lgb_model)
    shap_values = explainer.shap_values(X_row)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    sv = np.asarray(sv).reshape(-1)

    contributions = [
        FeatureContribution(
            feature=col,
            label=FEATURE_LABELS.get(col, col),
            value=float(X_row.iloc[0][col]),
            shap_value=float(sv[i]),
        )
        for i, col in enumerate(X_row.columns)
    ]
    contributions.sort(key=lambda c: -abs(c.shap_value))
    return contributions


def _template_explanation(contributions: list[FeatureContribution], risk_score: float, top_k: int = 3) -> str:
    top = contributions[:top_k]
    parts = [f"{c.label} ({c.direction()} risk)" for c in top]
    return (
        f"Predicted risk score: {risk_score:.1%}. "
        f"The largest contributing factors were: {', '.join(parts)}."
    )


def llm_explanation(
    contributions: list[FeatureContribution],
    risk_score: float,
    top_k: int = 5,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Plain-language explanation, grounded in the SHAP contributions
    passed in. Falls back to a template if no API key is set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _template_explanation(contributions, risk_score, top_k=3)

    top = contributions[:top_k]
    facts = "\n".join(
        f"- {c.label}: patient value={c.value:.2f}, SHAP contribution={c.shap_value:+.4f} "
        f"({c.direction()} predicted risk)"
        for c in top
    )

    prompt = (
        "You are summarizing a machine learning model's risk prediction for a "
        "clinician. You are given the model's predicted risk score and its top "
        "SHAP feature attributions for this specific patient. Write 2-3 plain-"
        "language sentences explaining the prediction using ONLY the numbers "
        "given below. Do not introduce any clinical judgment, diagnosis, or "
        "information not present in the data below. Do not recommend treatment.\n\n"
        f"Predicted risk score: {risk_score:.1%}\n"
        f"Top SHAP contributions (higher = pushed risk up):\n{facts}\n"
    )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()
    except Exception as e:  # don't let an API hiccup kill a demo
        print(f"[explain] LLM explanation failed ({e}), falling back to template")
        return _template_explanation(contributions, risk_score, top_k=3)
