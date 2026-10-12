# Telco Customer Churn: from cleaning to a costed A/B test

An end-to-end churn analysis of the IBM Telco dataset (7,043 customers, California). It is built for **insight and a business decision**, not leaderboard accuracy. The analysis covers: problem framing → documented cleaning → EDA → formal hypothesis tests → a model zoo with ensembles, interpretation and calibration → an A/B test design with power analysis, simulation and ROI.

## Headline results

| | |
|---|---|
| **Churn this quarter** | 26.5 % of customers, **30.5 % of monthly recurring revenue**, ≈ $1.37 M expected lifetime revenue |
| **Biggest driver** | Contract type: month-to-month customers churn **36 pts** more than annual (42.7 % vs 6.8 %; Cramér's V = 0.41) |
| **Other confirmed drivers** | E-check payment (+28 pts), fiber vs DSL (+23 pts, persists within contract: MH OR 2.55), no dependents, short tenure, no security/tech-support add-ons |
| **Price** | Churners' median bill is $15 higher (small-to-moderate, r = 0.24). `monthly_charges` is 99.9 % explained by the services bought, so price acts *through* the product mix |
| **Models** | 18 configurations (12 families + tuned + soft-voting + stacking) all reach CV AUC 0.83–0.87. Soft voting is best (0.865), but **tuned logistic regression (0.863)** is within 1 SE and is selected. Test AUC 0.853, Brier 0.134, already well calibrated |
| **Power** | 26 % → 21 % needs **1,126 per arm**. The real target segment (2,039 customers, 62 % churn) needs **1,513 per arm** for a 5-pt drop, so the test needs two waves |
| **ROI** | A 20 %/3-month discount breaks even at a **4.3-pt** churn reduction (range 2.4–7.5 pts depending on how long saved customers stay). A simulated test with a true 5-pt effect gives 4.3 pts [1.3, 7.3], so the decision is **inconclusive** and the pre-registered rule correctly withholds roll-out |

## Project layout

```
telco-churn-analysis/
├── notebooks/                             the analysis, read in order
│   ├── 01_problem_framing.ipynb           churn + cost-of-churn definitions, sizing
│   ├── 02_data_cleaning_and_eda.ipynb     cleaning decisions, 5 one-message charts
│   ├── 03_hypothesis_testing.ipynb        χ², Mann-Whitney, z-tests, CMH, Holm
│   ├── 04_churn_modelling.ipynb           model zoo, odds ratios, SHAP, calibration, lift
│   └── 05_retention_ab_test.ipynb         power, simulation, ROI sensitivity, recommendation
├── src/churn/                             reusable code the notebooks, scripts and tests import
│   ├── config.py                          paths, feature groups, constants
│   ├── data_cleaning.py                   loading + cleaning with a decision log
│   ├── preprocessing.py                   shared sklearn ColumnTransformer / Pipeline
│   ├── model_zoo.py                       12 model families, search spaces, voting + stacking
│   ├── training.py                        CV comparison, tuning, 1-SE selection, calibration
│   ├── metrics.py                         AUC/PR-AUC/Brier/ECE, reliability, lift, thresholds
│   ├── hypothesis_tests.py                tests that always return effect sizes + CIs
│   ├── ab_testing.py                      cost of churn, power, MDE, simulation, ROI, break-even
│   ├── interpretation.py                  logit odds ratios, LightGBM TreeSHAP
│   ├── charts.py                          every notebook chart
│   └── style.py                           chart style, styled tables, result cards
├── tests/                                 pytest suite (48 tests)
├── train_pipeline.py                      CLI: end-to-end training
├── score_customers.py                     CLI: score customers with the saved model
├── data/
│   ├── raw/telco_customer_churn.xlsx      IBM extended version, 33 columns
│   └── processed/                         clean table + out-of-fold scores (generated)
├── models/                                churn_model.joblib, evaluation_artifacts.joblib (generated)
├── reports/
│   ├── figures/                           every chart as PNG (generated)
│   ├── metrics/                           CV comparison, test results, tuning, calibration (generated)
│   ├── customer_risk_scores.csv           output of score_customers.py
│   └── training_log.txt                   console log of the last training run
├── requirements.txt
└── pytest.ini
```

## How to run

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe train_pipeline.py                                   # ~8 min
.venv\Scripts\python.exe score_customers.py data\raw\telco_customer_churn.xlsx   # → reports/customer_risk_scores.csv
.venv\Scripts\python.exe -m pytest                                           # 48 tests, ~10 s
```

The tests cover cleaning on a hand-built messy fixture plus the real data, the statistics helpers against known answers (e.g. a Mantel-Haenszel OR of exactly 4, Holm adjustments), the power and sample-size maths against two textbook formulas, A/B simulation false-positive rate ≈ α, break-even ⇒ zero net revenue, valid probabilities from every model in the zoo, the one-SE selection rule, and SHAP additivity.

Then open the notebooks in order with the **Python (churn-ds)** kernel. Notebooks 04 and 05 read the pipeline's artefacts, so run the pipeline first.

## Pipeline design choices worth discussing

- **No leakage:** `Churn Score` (IBM's own model), `Churn Reason` (only known after churn), `Churn Label` and `CLTV` are excluded from features. All preprocessing sits inside the sklearn `Pipeline`, so every CV fold is fit only on its training rows.
- **One-standard-error rule:** the fold-to-fold SD of AUC (~0.012) is several times the gap between top models, so the simplest statistically tied model wins.
- **Calibration is tested, not assumed:** raw vs. Platt vs. isotonic are compared on out-of-fold Brier score. Naive Bayes and AdaBoost show how a good-AUC model can still give unusable probabilities.
- **Out-of-fold scores for the A/B design:** every customer's risk comes from a model that never saw them.
- **SHAP without the `shap` package:** LightGBM's native `pred_contrib=True` computes exact TreeSHAP values. See the environment note below.

## Environment note

Windows Smart App Control on this machine blocks compiled extensions from very new wheels (pandas 3.x, scikit-learn 1.9, pyarrow, shap ≥ 0.46). `requirements.txt` pins older, widely used versions that load correctly. Processed data is stored as pickle rather than parquet because pyarrow is unavailable.
