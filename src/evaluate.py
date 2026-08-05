"""Evaluation on the held-out TEST split, touched exactly once.
ROC-AUC + PR-AUC + precision/recall/F1 on the positive class + Brier
score for calibration + raw confusion matrix. Accuracy alone isn't
reported as a headline number here — see README for why."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.pipeline import Split
from src.train import TrainedEnsemble


@dataclass
class TestReport:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    brier: float
    tn: int
    fp: int
    fn: int
    tp: int
    n: int
    positive_rate: float

    def missed_case_rate(self) -> float:
        """Of all actual diabetic patients, what fraction did we miss?"""
        return self.fn / (self.fn + self.tp) if (self.fn + self.tp) else float("nan")

    def summary(self) -> str:
        return (
            f"ROC-AUC={self.roc_auc:.4f}  PR-AUC={self.pr_auc:.4f}  "
            f"Precision={self.precision:.4f}  Recall={self.recall:.4f}  "
            f"F1={self.f1:.4f}  Brier={self.brier:.4f}\n"
            f"Confusion matrix -> TN={self.tn} FP={self.fp} FN={self.fn} TP={self.tp}\n"
            f"Missed {self.fn} of {self.fn + self.tp} true diabetic cases "
            f"({self.missed_case_rate():.1%})"
        )


def evaluate_on_test(ensemble: TrainedEnsemble, test: Split) -> TestReport:
    prob = ensemble.predict_proba(test.X)
    pred = (prob >= ensemble.weights.threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(test.y, pred).ravel()

    return TestReport(
        roc_auc=roc_auc_score(test.y, prob),
        pr_auc=average_precision_score(test.y, prob),
        precision=precision_score(test.y, pred, zero_division=0),
        recall=recall_score(test.y, pred, zero_division=0),
        f1=f1_score(test.y, pred, zero_division=0),
        brier=brier_score_loss(test.y, prob),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        n=len(test.y),
        positive_rate=float(np.mean(test.y)),
    )


def subgroup_report(ensemble: TrainedEnsemble, test: Split, raw_test_df) -> dict:
    """Recall by gender. Reports the number whether or not it's flattering."""
    prob = ensemble.predict_proba(test.X)
    pred = (prob >= ensemble.weights.threshold).astype(int)

    out = {}
    genders = raw_test_df["gender"].unique()
    for g in genders:
        mask = (raw_test_df["gender"] == g).values
        if mask.sum() == 0 or test.y[mask].sum() == 0:
            continue
        out[g] = {
            "n": int(mask.sum()),
            "positive_rate": float(test.y[mask].mean()),
            "recall": float(recall_score(test.y[mask], pred[mask], zero_division=0)),
            "precision": float(precision_score(test.y[mask], pred[mask], zero_division=0)),
        }
    return out
