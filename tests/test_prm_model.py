"""Unit tests for PRM calibration metrics, Temperature Scaling, and [STEP] token encoding."""

from __future__ import annotations

import numpy as np
import pytest

from pocket_r1.prm.calibration import (
    compute_ece,
    compute_roc_auc,
    compute_step_metrics,
    fit_temperature_scaling,
    sigmoid_np,
)
from pocket_r1.prm.modeling_deberta import STEP_TOKEN, encode_prm_sample


class _DummyTokenizer:
    cls_token_id = 1
    sep_token_id = 2
    unk_token_id = 3
    step_id = 99

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.step_id if token == STEP_TOKEN else self.unk_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        for tok in text.split():
            out.append(self.step_id if tok == STEP_TOKEN else (10 + (len(tok) % 50)))
        return out


def test_encode_prm_sample_preserves_all_six_step_tokens_under_truncation() -> None:
    tok = _DummyTokenizer()
    huge_prompt = " ".join(f"word{i}" for i in range(1500))
    steps = [f"{i}. step content here" for i in range(1, 6)]
    enc = encode_prm_sample(tok, huge_prompt, steps, "CHECK_CALL", max_length=128)

    assert len(enc["input_ids"]) <= 128
    assert len(enc["step_positions"]) == 6
    for pos in enc["step_positions"]:
        assert enc["input_ids"][pos] == tok.step_id


def test_calibration_auc_ece_and_temperature_scaling() -> None:
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=(200, 6)).astype(np.float64)
    # Overconfident logits (too sharp -> optimal T > 1.0)
    logits = np.where(labels == 1, 4.5, -4.5) + rng.normal(0, 1.5, size=(200, 6))

    raw_probs = sigmoid_np(logits)
    t_opt = fit_temperature_scaling(logits, labels)
    cal_probs = sigmoid_np(logits / t_opt)

    assert compute_roc_auc(labels.ravel(), raw_probs.ravel()) > 0.95
    assert compute_ece(cal_probs, labels) <= compute_ece(raw_probs, labels) + 0.02

    metrics = compute_step_metrics(cal_probs, labels)
    assert metrics["overall"]["auc_roc"] > 0.95
    assert metrics["overall"]["f1_valid"] > 0.90
    assert metrics["overall"]["f1_error"] > 0.90
    assert len(metrics["per_step"]) == 6
    assert metrics["steps_1_to_3"]["auc_roc"] == pytest.approx(metrics["overall"]["auc_roc"], abs=0.05)
