"""Calibration (Temperature Scaling, ECE) and per-step AUC-ROC / F1 metrics for DeBERTa PRM."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

STEP_NAMES: tuple[str, ...] = ("step1", "step2", "step3", "step4", "step5", "action")


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact ROC-AUC via Mann-Whitney U rank statistic (zero external dependencies)."""
    y_true = (y_true >= 0.5).astype(np.int64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_labels = y_true[order]

    # Average ranks for ties
    n = len(sorted_scores)
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[i : j + 1] = avg_rank
        i = j + 1

    sum_pos_ranks = float(np.sum(ranks[sorted_labels == 1]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_binary_f1(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    yt = (y_true >= 0.5).astype(np.int64)
    yp = (y_prob >= threshold).astype(np.int64)

    # Error-detection F1 (positive class = ERROR, i.e., label == 0) AND overall positive F1
    tp_pos = int(np.sum((yt == 1) & (yp == 1)))
    fp_pos = int(np.sum((yt == 0) & (yp == 1)))
    fn_pos = int(np.sum((yt == 1) & (yp == 0)))
    prec_pos = tp_pos / max(1, tp_pos + fp_pos)
    rec_pos = tp_pos / max(1, tp_pos + fn_pos)
    f1_pos = 2 * prec_pos * rec_pos / max(1e-12, prec_pos + rec_pos)

    tp_err = int(np.sum((yt == 0) & (yp == 0)))
    fp_err = int(np.sum((yt == 1) & (yp == 0)))
    fn_err = int(np.sum((yt == 0) & (yp == 1)))
    prec_err = tp_err / max(1, tp_err + fp_err)
    rec_err = tp_err / max(1, tp_err + fn_err)
    f1_err = 2 * prec_err * rec_err / max(1e-12, prec_err + rec_err)

    acc = float(np.mean(yt == yp))
    return {
        "accuracy": round(acc, 4),
        "f1_valid": round(f1_pos, 4),
        "f1_error": round(f1_err, 4),
        "precision_error": round(prec_err, 4),
        "recall_error": round(rec_err, 4),
    }


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) over probability bins in [0, 1]."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = (np.asarray(labels, dtype=np.float64).ravel() >= 0.5).astype(np.float64)
    if len(probs) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (probs >= lo) & (probs <= hi if b == n_bins - 1 else probs < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            avg_conf = float(np.mean(probs[mask]))
            avg_acc = float(np.mean(labels[mask]))
            ece += (cnt / n) * abs(avg_conf - avg_acc)
    return round(float(ece), 4)


def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find scalar temperature T in [0.25, 5.0] minimizing binary NLL on validation logits."""
    lg = np.asarray(logits, dtype=np.float64).ravel()
    yt = (np.asarray(labels, dtype=np.float64).ravel() >= 0.5).astype(np.float64)
    if len(lg) == 0:
        return 1.0

    best_t = 1.0
    best_nll = float("inf")
    # Coarse-to-fine grid search (fast, deterministic, never diverges)
    for grid in (np.linspace(0.3, 4.0, 75),):
        for t in grid:
            p = np.clip(sigmoid_np(lg / t), 1e-7, 1.0 - 1e-7)
            nll = -float(np.mean(yt * np.log(p) + (1.0 - yt) * np.log(1.0 - p)))
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)

    fine = np.linspace(max(0.15, best_t - 0.1), best_t + 0.1, 41)
    for t in fine:
        p = np.clip(sigmoid_np(lg / t), 1e-7, 1.0 - 1e-7)
        nll = -float(np.mean(yt * np.log(p) + (1.0 - yt) * np.log(1.0 - p)))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    return round(best_t, 4)


def compute_step_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Compute overall and per-step AUC-ROC, F1, and ECE for shape (N, 6) predictions."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    flat_p = probs.ravel()
    flat_y = labels.ravel()

    overall_auc = compute_roc_auc(flat_y, flat_p)
    overall_f1 = compute_binary_f1(flat_y, flat_p)
    overall_ece = compute_ece(flat_p, flat_y)

    per_step: dict[str, Any] = {}
    for i, name in enumerate(STEP_NAMES):
        sp = probs[:, i]
        sy = labels[:, i]
        auc = compute_roc_auc(sy, sp)
        f1_d = compute_binary_f1(sy, sp)
        per_step[name] = {
            "auc_roc": None if math.isnan(auc) else round(auc, 4),
            "error_rate": round(float(np.mean(sy < 0.5)), 4),
            **f1_d,
        }

    # Steps 1-3 combined factual accuracy
    s13_p = probs[:, :3].ravel()
    s13_y = labels[:, :3].ravel()
    s13_auc = compute_roc_auc(s13_y, s13_p)
    s13_f1 = compute_binary_f1(s13_y, s13_p)

    return {
        "overall": {
            "auc_roc": None if math.isnan(overall_auc) else round(overall_auc, 4),
            "ece": overall_ece,
            **overall_f1,
        },
        "steps_1_to_3": {
            "auc_roc": None if math.isnan(s13_auc) else round(s13_auc, 4),
            **s13_f1,
        },
        "per_step": per_step,
    }
