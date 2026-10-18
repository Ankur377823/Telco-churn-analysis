"""
churn — reusable code for the Telco customer churn analysis
===========================================================

Every notebook, both command-line scripts and the test suite import from this
package, so each piece of logic exists in exactly one place.

    Module             What it does
    -----------------  ------------------------------------------------------------
    config             paths, column groups and shared constants
    data_cleaning      load the raw workbook, clean it, log every decision
    preprocessing      the shared sklearn transform (scale + one-hot) and Pipeline
    model_zoo          12 model families, search spaces, voting + stacking ensembles
    metrics            AUC, Brier, calibration error, lift and threshold tables
    training           CV comparison -> tuning -> 1-SE selection -> calibration -> save
    hypothesis_tests   statistical tests that always return effect sizes + CIs
    interpretation     logistic-regression odds ratios and LightGBM SHAP values
    ab_testing         cost of churn, power analysis, experiment simulation, ROI
    style              colours, chart settings and table styling
    charts             one function per chart in the notebooks

Typical use:
    from churn import config as C
    from churn.data_cleaning import load_clean
    df = load_clean()
"""
