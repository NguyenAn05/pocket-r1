"""Unit tests for Phase 2 SFT data synthesis, PokerBench adapter, FIM, and zero-leakage builder."""

from __future__ import annotations

from pathlib import Path

from pocket_r1.data import (
    build_sft_splits,
    convert_pokerbench_records,
    create_fim_examples,
    synthesize_gym_dataset,
)


def test_synthesize_leduc_and_nlhe_verified_sft() -> None:
    leduc_ex = synthesize_gym_dataset(env_name="leduc", target_examples=30, base_seed=800000)
    assert len(leduc_ex) == 30
    assert all(len(ex.steps) == 5 for ex in leduc_ex)

    nlhe_ex = synthesize_gym_dataset(
        env_name="hu_nlhe", target_examples=60, base_seed=810000, equity_samples=40
    )
    assert len(nlhe_ex) == 60
    assert all(len(ex.messages) == 3 for ex in nlhe_ex)
    assert all("<action>" in ex.messages[2]["content"] for ex in nlhe_ex)


def test_fim_generator_masks_step2_or_step3() -> None:
    examples = synthesize_gym_dataset(
        env_name="hu_nlhe", target_examples=40, base_seed=820000, equity_samples=40
    )
    fim = create_fim_examples(examples, ratio=0.5, seed=42)
    assert len(fim) == 20
    for item in fim:
        assert item.task_type in ("fim_step2", "fim_step3")
        assert "<MASK_STEP>" in item.messages[1]["content"]
        assert item.messages[2]["content"].startswith("<step>")
        assert item.messages[2]["content"].endswith("</step>")


def test_pokerbench_adapter() -> None:
    spots = [
        {
            "hero_cards": ["As", "Ks"],
            "board": ["Qs", "7s", "2h"],
            "pot_bb": 18.0,
            "to_call_bb": 6.0,
            "hero_stack_bb": 89.0,
            "villain_stack_bb": 83.0,
            "position": "BTN",
            "action": "call",
        },
        {
            "hero_cards": ["Ah", "Ad"],
            "board": [],
            "pot_bb": 1.5,
            "to_call_bb": 0.5,
            "hero_stack_bb": 99.5,
            "villain_stack_bb": 99.0,
            "position": "BTN",
            "action": "raise 2.5",
        },
    ]
    out = convert_pokerbench_records(spots, seed=900000)
    assert len(out) == 2
    assert out[0].action == "CHECK_CALL"
    assert out[1].action in ("BET_75", "BET_150", "BET_33")


def test_build_sft_splits_hand_grouped_zero_leakage(tmp_path: Path) -> None:
    examples = synthesize_gym_dataset(
        env_name="hu_nlhe", target_examples=80, base_seed=830000, equity_samples=32
    )
    stats = build_sft_splits(
        examples=examples,
        output_dir=tmp_path,
        eval_suite_path="data/eval_suites/hu_nlhe_heldout_v1.jsonl",
        val_ratio=0.15,
        fim_ratio=0.25,
        seed=42,
    )
    assert stats["heldout_overlap"] == 0
    assert stats["train_standard_count"] > 0
    assert stats["train_fim_count"] > 0
    assert stats["val_count"] > 0
    assert (tmp_path / "train_mixed_fim.jsonl").exists()
    assert (tmp_path / "val_sft.jsonl").exists()
    assert (tmp_path / "stats_report.json").exists()
