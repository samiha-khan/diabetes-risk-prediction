# Diabetes Risk Prediction

Stacking ensemble (LightGBM + XGBoost + CatBoost) predicting diabetes risk
from routine clinical measurements. SHAP for feature attribution, LLM layer
on top to phrase it in plain language.

**Test set: ROC-AUC 0.977 · PR-AUC 0.879 · Recall 0.78 · Precision 0.74**

## The actual problem

The dataset is ~91.5% non-diabetic. A model that predicts "no diabetes"
for everyone scores 91.5% accuracy and catches zero real cases. So
accuracy isn't the metric that matters here. The real questions are how
you evaluate this honestly under that imbalance, where you set the
threshold when missing a diabetic is worse than a false alarm, and how
you explain a stacking ensemble's output to someone who isn't reading a
SHAP plot.

## Fixed a leakage bug

Earlier version tuned ensemble weights and the decision threshold by
grid-searching for whatever maximized accuracy on the test set. That's
leakage. The "test" numbers weren't a real generalization estimate,
because model selection had already seen that exact data.

Fixed with a proper split: train models on 70%, tune weights/threshold on
a 15% validation set (by F1, not accuracy), evaluate on the last 15% once
and never touch it again. There's a regression test for this
(`test_encoders_are_fit_on_train_only_not_full_data`) so it can't quietly
come back.

For what it's worth, fixing it barely moved the number: 96.0% claimed
accuracy vs. 95.8% honest accuracy. Dataset's big enough that the leakage
didn't do much damage in practice. Still wrong to do it, and now it's
actually fixed instead of just less-bad.

## Metrics

```
ROC-AUC   0.977
PR-AUC    0.879
Precision 0.744
Recall    0.780
F1        0.761
Brier     0.034

Confusion matrix:  TN=13381  FP=342  FN=281  TP=994
Missed 281 of 1,275 true diabetic cases (22%)
```

![Confusion matrix](reports/figures/confusion_matrix.png)

The 22% miss rate is the number I'd lead with, not accuracy. Threshold is
currently set to the F1-optimal point on validation; `src/train.py` has
the full precision/recall grid if it needs to move toward higher recall
for a real screening use case.

## Subgroup check (gender)

| | n | Positive rate | Recall | Precision |
|---|---|---|---|---|
| Female | 8,713 | 7.6% | 0.793 | 0.705 |
| Male | 6,285 | 9.8% | 0.765 | 0.792 |

~3 point recall gap. Could be noise at this sample size, could be real.
Reporting it either way.

## SHAP + LLM explanations

SHAP gives feature attributions as numbers. The LLM layer (`src/explain.py`)
turns a given patient's SHAP output into a sentence. It only ever sees
the SHAP values, never raw patient data. It summarizes the attributions
it's given rather than making an independent prediction; that scopes what
it can say, though it's still a model call, so the wording itself isn't
guaranteed to be perfect even when the underlying numbers are.

![SHAP summary](reports/figures/shap_summary.png)

HbA1c and blood glucose dominate, which matches how these are actually
diagnosed clinically. No surprises there, which is itself a reasonable
sanity check on the model.

```
TRUE POSITIVE (risk 100%, actual: diabetic)
HbA1c level, age×BMI interaction, and age pushed risk up.

FALSE NEGATIVE (risk 20.2%, actual: diabetic)
HbA1c and glucose pushed risk down; age×BMI pushed it up. Model missed
this one on the two strongest lab features not flagging despite the
true outcome being diabetic.
```

No API key set → falls back to a templated version of the same sentence
instead of breaking.

## Structure

```
src/
  features.py     - clinical feature engineering, fixed ADA thresholds
  pipeline.py      - train/val/test split, encoders fit on train only
  train.py         - base models + ensemble tuning (val only)
  evaluate.py      - test metrics + subgroup check
  explain.py       - SHAP + LLM explanation layer
  main.py          - full run
  demo_explain.py  - per-patient explanation demo
tests/
  test_features.py
  test_pipeline.py  - split integrity, leakage regression guard
reports/
  test_metrics.json
```

## Running it

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests/ -v
PYTHONPATH=. python -m src.main
PYTHONPATH=. python -m src.demo_explain
```

Set `ANTHROPIC_API_KEY` for LLM-phrased explanations, otherwise it uses
the template.

## How this compares to published results

A true apples-to-apples external validation (same model, a different real
dataset) isn't possible here without retraining: this project's features
(gender, age, hypertension, heart disease, smoking history, BMI, HbA1c,
blood glucose) are specific to this one Kaggle dataset, and no other
public diabetes dataset shares that exact schema. What's possible instead,
and genuinely useful, is checking this project's 0.977 ROC-AUC against
published results on related diabetes-classification problems:

| Study | Dataset | Model | Accuracy | AUC | Precision | Recall |
|---|---|---|---|---|---|---|
| This project | Kaggle, ~100K rows, clinical features | LightGBM+XGBoost+CatBoost stacking | 95.8% | 0.977 | 0.744 | 0.780 |
| [Li, Peng & Peng, 2024, *PLOS ONE*](https://doi.org/10.1371/journal.pone.0311222) | CDC BRFSS 2021 survey, 25 of 303 features | GA-XGBoost + LightGBM stacking | 94.86% | 0.989 | 0.965 | 0.952 |
| [arXiv:2501.18071](https://arxiv.org/abs/2501.18071) | Diabetes Binary Health Indicators (BRFSS-derived) | Ensemble + SMOTE | 92.50% | 0.975 | - | - |
| [Rahman, Hossain, Tiang & Nahid, 2025, *Diagnostics* 15(20):2622](https://doi.org/10.3390/diagnostics15202622) | Pima Indians (768 rows, 8 features) | LightGBM + Boruta feature selection | 85.16% | 0.905 | 0.840 | 0.868 |

None of these three studies use this project's exact dataset or feature
set, so this isn't a controlled comparison, just the honest context
available: a 2024 BRFSS stacking-ensemble study (methodologically close to
this project, also a boosting stack) reports higher precision and recall
on its own data, this project's ROC-AUC sits right in the range of the two
other BRFSS-adjacent studies (0.975, 0.989), and this project's model
clears the classic, much smaller Pima Indians benchmark on every metric.
That last comparison matters least, since 768 rows is a different scale
problem entirely, but it's the most commonly cited diabetes-ML benchmark,
so it's included for that reason.

I also found a paper reporting results on what looks like the exact same
dataset (96,146 rows after cleaning, same feature list), but could not
retrieve its exact numbers: the primary source is paywalled on
ResearchGate and a related open-access PDF I found didn't extract to
readable text. Noting the gap rather than guessing at a number to fill it.

One honest asymmetry worth naming: the BRFSS-based studies above report
notably higher precision and recall (0.95+/0.95+) than this project
(0.744/0.780) despite similar or lower AUC. That's a real signal worth
investigating before claiming parity, not just a methodology footnote to
skim past. The most likely cause is threshold choice, not model quality:
this project's threshold is tuned to the F1-optimal point (`src/train.py`),
which trades recall for precision in a way a study optimizing for a
different operating point wouldn't. It could also reflect real differences
in how cleanly BRFSS's self-reported survey labels separate from this
project's lab-value-based labels. Both are plausible from the numbers
alone; distinguishing them would need the other papers' precision/recall
curves, not just their reported single operating point, which none of the
three papers published.

## Limitations

- Dataset's synthetic (Kaggle), not real clinical records. Real-world
  noise and missingness would likely hurt these numbers.
- Screening aid, not diagnosis. 22% miss rate matters in a real clinical
  setting; threshold should get set by whoever owns that tradeoff, not
  default to F1.
- Fairness check only covers gender, and only the two categories with
  enough samples to say anything reliable (dropped ~18 rows labeled
  "Other" for that reason). Doesn't cover other attributes.
- LLM explanation is constrained to the SHAP numbers it's given, but it's
  still a model call. The template fallback exists partly because a demo
  shouldn't depend on an API being up.
