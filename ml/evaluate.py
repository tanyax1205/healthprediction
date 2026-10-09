"""Evaluation metrics, curves and calibration reporting."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

__all__ = ["evaluate_binary", "evaluate_multiclass", "evaluate", "select_threshold"]


def _downsample(arr: np.ndarray, n: int = 100) -> list[float]:
    arr = np.asarray(arr)
    if arr.size <= n:
        return [float(x) for x in arr]
    idx = np.linspace(0, arr.size - 1, n).astype(int)
    return [float(x) for x in arr[idx]]


def _calibration(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> dict:
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, proba, n_bins=n_bins, strategy="quantile"
        )
        return {"mean_predicted": _downsample(mean_pred, 20), "fraction_positive": _downsample(frac_pos, 20)}
    except Exception:
        return {"mean_predicted": [], "fraction_positive": []}


def evaluate_binary(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }

    fpr, tpr, _ = roc_curve(y_true, proba)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    curves = {
        "roc": {"fpr": _downsample(fpr), "tpr": _downsample(tpr)},
        "calibration": _calibration(y_true, proba),
    }
    return {"metrics": metrics, "confusion_matrix": cm.tolist(), "curves": curves}


def evaluate_multiclass(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(str)
    proba = np.asarray(proba, dtype=float)
    class_index = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_index[v] for v in y_true])
    pred_idx = proba.argmax(axis=1)

    # Multiclass Brier = mean over samples of sum_c (p_c - onehot_c)^2
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_idx, pred_idx)),
        "f1_macro": float(f1_score(y_idx, pred_idx, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_idx, pred_idx, average="weighted", zero_division=0)),
        "brier": brier,
        "n_classes": len(classes),
    }
    try:
        metrics["roc_auc_ovr"] = float(
            roc_auc_score(onehot, proba, multi_class="ovr", average="macro")
        )
    except Exception:
        pass

    cm = confusion_matrix(y_idx, pred_idx)
    curves = {"calibration": _calibration((pred_idx == y_idx).astype(int), proba.max(axis=1))}
    return {"metrics": metrics, "confusion_matrix": cm.tolist(), "curves": curves}


def evaluate(
    kind: str,
    y_true,
    proba,
    classes: list[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if kind == "binary":
        return evaluate_binary(y_true, proba, threshold=threshold)
    return evaluate_multiclass(y_true, proba, classes or [str(c) for c in np.unique(y_true)])


def select_threshold(y_true: np.ndarray, proba: np.ndarray, min_recall: float = 0.60) -> float:
    """Pick the operating threshold on validation probabilities.

    Maximises F1 subject to ``recall >= min_recall`` for the positive class;
    falls back to the unconstrained best-F1 threshold when that is infeasible.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (proba >= t).astype(int)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        if r >= min_recall and f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    if best_f1 < 0:  # infeasible: unconstrained best-F1
        for t in np.linspace(0.01, 0.99, 99):
            pred = (proba >= t).astype(int)
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = float(t), float(f1)
    return best_t
