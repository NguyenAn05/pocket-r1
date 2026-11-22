"""Adversarial Co-evolution utilities: False Positive mining & Rejection Sampling (RFT)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pocket_r1.prm.data_corruptor import PRMSample


def mine_false_positives(
    rollout_logs: list[dict[str, Any]],
    prm_threshold: float = 0.75,
    regret_threshold_bb: float = -3.0,
) -> list[PRMSample]:
    """Extract high-PRM / bad-EV trajectories to upgrade DeBERTa-v3 -> v(t+1)."""
    mined: list[PRMSample] = []
    for idx, r in enumerate(rollout_logs):
        r_prm = float(r.get("r_prm", 0.0))
        regret = float(r.get("regret_bb", 0.0))
        steps = r.get("steps") or []
        action = r.get("action")
        if len(steps) != 5 or not action:
            continue
        if r_prm >= prm_threshold and regret <= regret_threshold_bb:
            mined.append(
                PRMSample(
                    sample_id=f"fp_mined_{idx:06d}",
                    hand_id=str(r.get("hand_id", f"fp_{idx}")),
                    source="coevolution:false_positive",
                    prompt_text=str(r["prompt_text"]),
                    steps=list(steps),
                    action=str(action),
                    # Mark Step 5 & Action as erroneous (0.0) so DeBERTa v(t+1) catches this leak
                    labels=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    meta={"orig_r_prm": r_prm, "regret_bb": regret},
                )
            )
    return mined


def rejection_sample_top_trajectories(
    rollout_logs: list[dict[str, Any]],
    min_total_reward: float = 0.45,
    max_regret_bb: float = -0.5,
) -> list[dict[str, Any]]:
    """Filter top-performing verified rollouts for Rejection Sampling Fine-Tuning (RFT)."""
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for r in rollout_logs:
        if r.get("hard_gate_failed", False):
            continue
        if float(r.get("total_reward", -1.0)) < min_total_reward:
            continue
        if float(r.get("regret_bb", -10.0)) < max_regret_bb:
            continue
        key = (str(r.get("prompt_text", "")), str(r.get("action", "")))
        if key in seen:
            continue
        seen.add(key)
        selected.append(r)
    return selected


def save_jsonl_dicts(items: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
