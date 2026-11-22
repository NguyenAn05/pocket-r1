"""Unit tests for Phase 3 PRM Synthetic Corruption and On-policy Step Labeler."""

from __future__ import annotations

import json
from pathlib import Path

from pocket_r1.data import synthesize_gym_dataset
from pocket_r1.prm import (
    CORRUPTION_TYPES,
    corrupt_sft_example,
    generate_synthetic_prm_samples,
    label_onpolicy_trajectory_files,
)
from pocket_r1.utils.seeding import make_rng


def test_all_six_synthetic_corruption_types() -> None:
    examples = synthesize_gym_dataset(
        env_name="hu_nlhe", target_examples=30, base_seed=910000, equity_samples=32
    )
    rng = make_rng(42)

    for ctype in CORRUPTION_TYPES:
        produced = None
        for ex in examples:
            produced = corrupt_sft_example(ex, ctype, rng)
            if produced is not None:
                break
        assert produced is not None, f"Failed to produce corruption for {ctype}"
        assert len(produced.steps) == 5
        assert len(produced.labels) == 6
        assert any(y == 0.0 for y in produced.labels), f"Expected at least one 0 label for {ctype}"


def test_generate_synthetic_prm_samples_balance() -> None:
    examples = synthesize_gym_dataset(
        env_name="hu_nlhe", target_examples=25, base_seed=920000, equity_samples=32
    )
    clean, corrupted = generate_synthetic_prm_samples(examples, n_corruptions=30, seed=7)
    assert len(clean) == 25
    assert len(corrupted) == 30
    assert all(s.labels == [1.0] * 6 for s in clean)
    assert all(any(y == 0.0 for y in s.labels) for s in corrupted)


def test_label_onpolicy_trajectories(tmp_path: Path) -> None:
    traj_file = tmp_path / "exp0_sample" / "trajectories.jsonl"
    traj_file.parent.mkdir(parents=True, exist_ok=True)
    recs = [
        {
            "hand_id": "h1",
            "decision_idx": 0,
            "street": "FLOP",
            "opponent_type": "tag",
            "prompt_text": "Game: HU_NLHE ...",
            "state_json": {"to_call_bb": 6.0},
            "parsed": {
                "steps": ["1. ok", "2. bad hand", "3. ok", "4. opponent read ok here", "5. call"],
                "action": "CHECK_CALL",
                "error_codes": [],
            },
            "auto_checks": {
                "step1_card_exact": True,
                "step2_hand_eval_ok": False,
                "step3_pot_odds_ok": True,
                "step3_expected_pot_odds": 0.25,
                "step3_equity_est": 0.45,
                "logic_action_consistent": True,
            },
            "hand_outcome_bb": -5.0,
        },
        {
            "hand_id": "h2",
            "decision_idx": 0,
            "street": "TURN",
            "opponent_type": "nit",
            "prompt_text": "Game: HU_NLHE ...",
            "state_json": {"to_call_bb": 15.0},
            "parsed": {
                "steps": ["1. ok", "2. ok", "3. ok", "4. opponent read ok here", "5. call"],
                "action": "CHECK_CALL",
                "error_codes": [],
            },
            "auto_checks": {
                "step1_card_exact": True,
                "step2_hand_eval_ok": True,
                "step3_pot_odds_ok": True,
                "step3_expected_pot_odds": 0.35,
                "step3_equity_est": 0.12,
                "logic_action_consistent": True,
            },
            "hand_outcome_bb": -35.0,
        },
    ]
    with traj_file.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    samples = label_onpolicy_trajectory_files([traj_file])
    assert len(samples) == 2
    assert samples[0].labels == [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    # Second sample had severe negative-EV call with -35BB loss -> Step 5 & Action flagged 0
    assert samples[1].labels == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
