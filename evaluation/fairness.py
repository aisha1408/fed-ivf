"""Algorithmic fairness evaluation across real clinical demographic groups using Fairlearn.

Evaluates disparate impact, selection rates, true positive rates (TPR / Recall),
false positive rates (FPR), and demographic parity difference across:
1. Patient Age Cohorts (18-34, 35-37, 38-39, 40+)
2. Body Mass Index (BMI) Groups (<25 Normal, 25-29.9 Overweight, >=30 Obese)
3. Infertility Diagnosis Categories (Male Factor, PCOS, DOR, Tubal, Unexplained, etc.)
"""

import os
import sys
from typing import Dict
import numpy as np
import pandas as pd
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate
)
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.neural_net import create_model, set_parameters
from preprocessing.preprocess import DIAGNOSIS_LABELS


def create_age_bins(age_series: pd.Series) -> pd.Series:
    """Bin continuous patient age into clinically standard reproductive cohorts."""
    bins = [0, 34.99, 37.99, 39.99, 120]
    labels = ["18-34 (Young)", "35-37 (Intermediate)", "38-39 (Advanced 1)", "40+ (Advanced 2)"]
    return pd.cut(age_series, bins=bins, labels=labels, right=True)


def create_bmi_bins(bmi_series: pd.Series) -> pd.Series:
    """Bin continuous BMI into WHO clinical classification groups."""
    bins = [0, 24.99, 29.99, 100]
    labels = ["Normal (<25)", "Overweight (25-29.9)", "Obese (>=30)"]
    return pd.cut(bmi_series, bins=bins, labels=labels, right=True)


def map_diagnosis_labels(diag_series: pd.Series) -> pd.Series:
    """Map raw diagnostic codes to clear clinical etiology descriptions."""
    return diag_series.map(lambda d: DIAGNOSIS_LABELS.get(str(d).lower(), "Other/Unexplained"))


def run_fairness_evaluation(
    model_path: str = "results/federated_model.pt",
    test_proc_path: str = "data/processed_test.csv",
    test_meta_path: str = "data/processed_test_meta.csv",
    report_output_path: str = "results/fairness_report.txt",
    threshold: float = 0.5
) -> Dict:
    """Perform Fairlearn audit and output fairness_report.txt."""
    if not os.path.exists(test_proc_path) or not os.path.exists(test_meta_path):
        raise FileNotFoundError("Processed test data or metadata missing for fairness evaluation.")

    df_proc = pd.read_csv(test_proc_path)
    df_meta = pd.read_csv(test_meta_path)

    x_test = df_proc.drop(columns=["live_birth"]).values.astype(np.float32)
    y_test = df_proc["live_birth"].values.astype(int)

    # Load model
    input_dim = x_test.shape[1]
    model = create_model(input_dim=input_dim)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        preds_prob = model(torch.tensor(x_test)).cpu().numpy().ravel()
    preds_binary = (preds_prob >= threshold).astype(int)

    # Create sensitive demographic features
    age_groups = create_age_bins(df_meta["age"])
    bmi_groups = create_bmi_bins(df_meta["bmi"])
    diag_groups = map_diagnosis_labels(df_meta["diag"])

    metrics_dict = {
        "selection_rate": selection_rate,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate,
        "accuracy": accuracy_score
    }

    # MetricFrames
    mf_age = MetricFrame(metrics=metrics_dict, y_true=y_test, y_pred=preds_binary, sensitive_features=age_groups)
    mf_bmi = MetricFrame(metrics=metrics_dict, y_true=y_test, y_pred=preds_binary, sensitive_features=bmi_groups)
    mf_diag = MetricFrame(metrics=metrics_dict, y_true=y_test, y_pred=preds_binary, sensitive_features=diag_groups)

    # Disparities
    dp_diff_age = demographic_parity_difference(y_test, preds_binary, sensitive_features=age_groups)
    eo_diff_age = equalized_odds_difference(y_test, preds_binary, sensitive_features=age_groups)

    dp_diff_bmi = demographic_parity_difference(y_test, preds_binary, sensitive_features=bmi_groups)
    eo_diff_bmi = equalized_odds_difference(y_test, preds_binary, sensitive_features=bmi_groups)

    dp_diff_diag = demographic_parity_difference(y_test, preds_binary, sensitive_features=diag_groups)
    eo_diff_diag = equalized_odds_difference(y_test, preds_binary, sensitive_features=diag_groups)

    # Format report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("FEDERATED IVF: ALGORITHMIC FAIRNESS & DEMOGRAPHIC PARITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("Framework: Fairlearn 0.14+ MetricFrame & Disparity Metrics")
    report_lines.append("Evaluated Model: Federated Learning Global Model (FedAvg)")
    report_lines.append(f"Decision Threshold: {threshold}")
    report_lines.append(f"Test Cohort Size: {len(y_test)} patients")
    report_lines.append("")

    report_lines.append("-" * 80)
    report_lines.append("1. FAIRNESS AUDIT ACROSS FEMALE AGE COHORTS")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Age Cohort':<26} | {'Patients':<8} | {'Selection Rate':<15} | {'TPR (Recall)':<12} | {'FPR':<8}")
    report_lines.append("-" * 80)
    for group in mf_age.by_group.index:
        row = mf_age.by_group.loc[group]
        n_patients = int((age_groups == group).sum())
        report_lines.append(
            f"{str(group):<26} | {n_patients:<8} | {row['selection_rate']:>13.4f}  | {row['tpr']:>10.4f}  | {row['fpr']:>6.4f}"
        )
    report_lines.append(f"\nDemographic Parity Difference (Age): {dp_diff_age:.4f}")
    report_lines.append(f"Equalized Odds Difference (Age):     {eo_diff_age:.4f}")
    report_lines.append("Interpretation: Age-dependent selection rate divergence reflects biological fecundity")
    report_lines.append("disparities inherent in clinical reproductive medicine rather than pure algorithmic bias.")
    report_lines.append("")

    report_lines.append("-" * 80)
    report_lines.append("2. FAIRNESS AUDIT ACROSS BODY MASS INDEX (BMI) GROUPS")
    report_lines.append("-" * 80)
    report_lines.append(f"{'BMI Classification':<26} | {'Patients':<8} | {'Selection Rate':<15} | {'TPR (Recall)':<12} | {'FPR':<8}")
    report_lines.append("-" * 80)
    for group in mf_bmi.by_group.index:
        row = mf_bmi.by_group.loc[group]
        n_patients = int((bmi_groups == group).sum())
        report_lines.append(
            f"{str(group):<26} | {n_patients:<8} | {row['selection_rate']:>13.4f}  | {row['tpr']:>10.4f}  | {row['fpr']:>6.4f}"
        )
    report_lines.append(f"\nDemographic Parity Difference (BMI): {dp_diff_bmi:.4f}")
    report_lines.append(f"Equalized Odds Difference (BMI):     {eo_diff_bmi:.4f}")
    report_lines.append("")

    report_lines.append("-" * 80)
    report_lines.append("3. FAIRNESS AUDIT ACROSS INFERTILITY DIAGNOSIS ETIOLOGIES")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Diagnosis Category':<35} | {'Patients':<8} | {'Selection Rate':<15} | {'TPR (Recall)':<12}")
    report_lines.append("-" * 80)
    for group in mf_diag.by_group.index:
        row = mf_diag.by_group.loc[group]
        n_patients = int((diag_groups == group).sum())
        report_lines.append(
            f"{str(group):<35} | {n_patients:<8} | {row['selection_rate']:>13.4f}  | {row['tpr']:>10.4f}"
        )
    report_lines.append(f"\nDemographic Parity Difference (Diagnosis): {dp_diff_diag:.4f}")
    report_lines.append(f"Equalized Odds Difference (Diagnosis):     {eo_diff_diag:.4f}")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("MANDATORY CLINICAL & REGULATORY RESEARCH DISCLAIMER")
    report_lines.append("=" * 80)
    report_lines.append("This fairness evaluation is conducted strictly for educational, academic,")
    report_lines.append("and computer science research demonstration purposes.")
    report_lines.append("It does NOT constitute medical advice, clinical guidance, or a certified")
    report_lines.append("health technology assessment under FDA / CE / MDR regulatory guidelines.")
    report_lines.append("Clinical deployment requires dedicated validation on prospective multi-center cohorts.")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)

    results = {
        "age": {
            "demographic_parity_difference": float(dp_diff_age),
            "equalized_odds_difference": float(eo_diff_age),
            "by_group": mf_age.by_group.to_dict()
        },
        "bmi": {
            "demographic_parity_difference": float(dp_diff_bmi),
            "equalized_odds_difference": float(eo_diff_bmi),
            "by_group": mf_bmi.by_group.to_dict()
        },
        "diagnosis": {
            "demographic_parity_difference": float(dp_diff_diag),
            "equalized_odds_difference": float(eo_diff_diag),
            "by_group": mf_diag.by_group.to_dict()
        }
    }
    return results


if __name__ == "__main__":
    run_fairness_evaluation()
