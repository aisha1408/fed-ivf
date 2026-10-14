"""Evaluation metrics suite for IVF live-birth classification.

Computes AUC-ROC, F1 Score, Brier Score, and Expected Calibration Error (ECE)
with comprehensive safeguards against single-class subsets, empty sets, or degenerate predictions.
"""

from typing import Dict, Union
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, accuracy_score


def calculate_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Area Under the ROC Curve with graceful handling of single-class edge cases."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        # Undefined AUC when only positive or negative class is present in the split
        return 0.5

    try:
        score = float(roc_auc_score(y_true, y_prob))
        if np.isnan(score):
            return 0.5
        return score
    except Exception:
        return 0.5


def calculate_f1(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Calculate F1 Score at the specified decision threshold."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    if len(y_true) == 0:
        return 0.0

    y_pred = (y_prob >= threshold).astype(int)
    try:
        return float(f1_score(y_true, y_pred, zero_division=0.0))
    except Exception:
        return 0.0


def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Brier Score (Mean Squared Error of probability predictions)."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    if len(y_true) == 0:
        return 1.0

    y_prob_clipped = np.clip(y_prob, 0.0, 1.0)
    try:
        return float(brier_score_loss(y_true, y_prob_clipped))
    except Exception:
        return float(np.mean((y_true - y_prob_clipped) ** 2))


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE) across n equal-width confidence intervals.
    
    ECE measures the alignment between predicted probabilities and actual positive frequencies.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    n = len(y_true)
    if n == 0:
        return 0.0

    y_prob = np.clip(y_prob, 0.0, 1.0)
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_limits[i]
        bin_upper = bin_limits[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / n) * np.abs(bin_conf - bin_acc)

    return float(ece)


def calculate_accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Calculate standard classification accuracy."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    if len(y_true) == 0:
        return 0.0

    y_pred = (y_prob >= threshold).astype(int)
    return float(accuracy_score(y_true, y_pred))


def evaluate_model(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    threshold: float = 0.5
) -> Dict[str, float]:
    """Comprehensive evaluation returning AUC, F1, Brier score, ECE, and Accuracy."""
    y_true_arr = np.asarray(y_true).ravel()
    y_prob_arr = np.asarray(y_prob).ravel()

    return {
        "auc": calculate_auc(y_true_arr, y_prob_arr),
        "f1": calculate_f1(y_true_arr, y_prob_arr, threshold=threshold),
        "brier": calculate_brier_score(y_true_arr, y_prob_arr),
        "ece": calculate_ece(y_true_arr, y_prob_arr),
        "accuracy": calculate_accuracy(y_true_arr, y_prob_arr, threshold=threshold)
    }


if __name__ == "__main__":
    y_test_mock = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    p_test_mock = np.array([0.9, 0.2, 0.8, 0.65, 0.3, 0.1, 0.75, 0.4])
    metrics = evaluate_model(y_test_mock, p_test_mock)
    print("Mock metrics evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
