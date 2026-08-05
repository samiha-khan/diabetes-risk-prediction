"""Stacking ensemble. Base models fit on TRAIN (SMOTEENN-resampled for
the ~11:1 imbalance). Weights + threshold tuned on VAL only. TEST stays
untouched until evaluate.py."""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN
from sklearn.metrics import average_precision_score

from src.pipeline import Split


@dataclass
class EnsembleWeights:
    lgb: float
    xgb: float
    cat: float
    threshold: float


@dataclass
class TrainedEnsemble:
    lgb_model: lgb.LGBMClassifier
    xgb_model: xgb.XGBClassifier
    cat_model: CatBoostClassifier
    weights: EnsembleWeights

    def predict_proba(self, X) -> np.ndarray:
        p = (
            self.weights.lgb * self.lgb_model.predict_proba(X)[:, 1]
            + self.weights.xgb * self.xgb_model.predict_proba(X)[:, 1]
            + self.weights.cat * self.cat_model.predict_proba(X)[:, 1]
        )
        return p

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= self.weights.threshold).astype(int)


def _resample_train(train: Split, random_state: int = 42):
    sm = SMOTEENN(random_state=random_state)
    X_res, y_res = sm.fit_resample(train.X, train.y)
    print(
        f"[resample] train before={len(train.X)} (pos={train.y.sum()}) "
        f"-> after SMOTEENN={len(X_res)} (pos={y_res.sum()})"
    )
    return X_res, y_res


def train_base_models(train: Split, random_state: int = 42):
    X_res, y_res = _resample_train(train, random_state=random_state)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=63,
        random_state=random_state, verbose=-1, n_jobs=1,
    )
    lgb_model.fit(X_res, y_res)

    xgb_model = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        eval_metric="logloss", random_state=random_state, n_jobs=1,
    )
    xgb_model.fit(X_res, y_res)

    cat_model = CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=6,
        random_state=random_state, verbose=False,
    )
    cat_model.fit(X_res, y_res)

    return lgb_model, xgb_model, cat_model


def tune_weights_and_threshold(
    lgb_model, xgb_model, cat_model, val: Split
) -> EnsembleWeights:
    """Grid search weights + threshold on VAL, scored by F1 on the
    positive class — accuracy would just reward predicting "no diabetes"
    for everyone on data this imbalanced."""
    from sklearn.metrics import f1_score

    lgb_prob = lgb_model.predict_proba(val.X)[:, 1]
    xgb_prob = xgb_model.predict_proba(val.X)[:, 1]
    cat_prob = cat_model.predict_proba(val.X)[:, 1]

    best_f1 = -1.0
    best = EnsembleWeights(lgb=0.4, xgb=0.3, cat=0.3, threshold=0.5)

    for w1 in np.arange(0.2, 0.7, 0.05):
        for w2 in np.arange(0.1, 0.6, 0.05):
            w3 = round(1 - w1 - w2, 2)
            if w3 < 0.05:
                continue
            prob = w1 * lgb_prob + w2 * xgb_prob + w3 * cat_prob
            for t in np.arange(0.20, 0.60, 0.01):
                pred = (prob >= t).astype(int)
                f1 = f1_score(val.y, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best = EnsembleWeights(lgb=round(w1, 2), xgb=round(w2, 2), cat=w3, threshold=round(t, 2))

    val_prob = best.lgb * lgb_prob + best.xgb * xgb_prob + best.cat * cat_prob
    val_ap = average_precision_score(val.y, val_prob)
    print(
        f"[tune] best weights lgb={best.lgb} xgb={best.xgb} cat={best.cat} "
        f"threshold={best.threshold} | val F1={best_f1:.4f} val PR-AUC={val_ap:.4f}"
    )
    return best


def fit_ensemble(train: Split, val: Split, random_state: int = 42) -> TrainedEnsemble:
    lgb_model, xgb_model, cat_model = train_base_models(train, random_state=random_state)
    weights = tune_weights_and_threshold(lgb_model, xgb_model, cat_model, val)
    return TrainedEnsemble(lgb_model=lgb_model, xgb_model=xgb_model, cat_model=cat_model, weights=weights)
