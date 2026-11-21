"""Unit tests for SFT trainer helper utilities (assistant-only loss masking & YAML config)."""

from __future__ import annotations

from pocket_r1.training import SFTTrainConfig, build_assistant_only_features


def test_assistant_only_loss_masking() -> None:
    prompt_ids = [10, 20, 30, 40]
    full_ids = [10, 20, 30, 40, 50, 60, 70]
    feat = build_assistant_only_features(prompt_ids, full_ids, max_seq_length=16)

    assert feat["input_ids"] == full_ids
    assert feat["attention_mask"] == [1] * 7
    assert feat["labels"] == [-100, -100, -100, -100, 50, 60, 70]


def test_assistant_only_truncation() -> None:
    prompt_ids = [1, 2, 3, 4]
    full_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    feat = build_assistant_only_features(prompt_ids, full_ids, max_seq_length=6)
    assert feat["input_ids"] == [1, 2, 3, 4, 5, 6]
    assert feat["labels"] == [-100, -100, -100, -100, 5, 6]


def test_sft_config_loads_from_yaml() -> None:
    cfg = SFTTrainConfig.from_yaml("configs/training/sft_nlhe_fim.yaml")
    assert cfg.base_model == "Qwen/Qwen3-4B"
    assert cfg.enable_thinking is False
    assert cfg.fp16 is True
    assert cfg.lora_r == 32
