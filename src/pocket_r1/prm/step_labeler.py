"""Automatic 6-checkpoint step labeler for real on-policy SLM trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pocket_r1.prm.data_corruptor import PRMSample


def _label_single_trajectory(rec: dict[str, Any], source_tag: str) -> PRMSample | None:
    """Convert a logged TrajectoryRecord dict into a 6-label PRMSample."""
    parsed = rec.get("parsed") or {}
    steps = parsed.get("steps") or []
    action = parsed.get("action")
    if len(steps) != 5 or not action:
        return None

    checks = rec.get("auto_checks") or {}
    errors = parsed.get("error_codes") or []

    y1 = 1.0 if checks.get("step1_card_exact", True) else 0.0
    y2 = 1.0 if checks.get("step2_hand_eval_ok", True) else 0.0
    y3 = 1.0 if checks.get("step3_pot_odds_ok", True) else 0.0

    # Step 4: valid unless empty/degenerate
    y4 = 1.0 if len(steps[3].strip()) >= 12 else 0.0

    # Step 5 & Action (checkpoint 6): check consistency, legality, and severe strategic leaks
    consistent = bool(checks.get("logic_action_consistent", True))
    legal = "ILLEGAL_ACTION" not in errors and "UNKNOWN_ACTION" not in errors
    y5 = 1.0 if consistent else 0.0
    y6 = 1.0 if (consistent and legal) else 0.0

    # Strategic EV / over-calling leak signal:
    # If hero called/bet with equity well below pot odds and suffered heavy loss (>= 25 BB),
    # flag the decision (Step 5 + Action) as erroneous!
    state = rec.get("state_json") or {}
    to_call = float(state.get("to_call_bb", 0.0))
    expected_po = float(checks.get("step3_expected_pot_odds", 0.0))
    equity_est = float(checks.get("step3_equity_est", 0.5))
    outcome = rec.get("hand_outcome_bb")

    if (
        action != "FOLD"
        and to_call >= 4.0
        and expected_po > 0.0
        and equity_est < expected_po - 0.12
        and outcome is not None
        and float(outcome) <= -20.0
    ):
        y5 = 0.0
        y6 = 0.0

    # If Step 1 or Step 2 had a factual hallucination AND hero lost big, also mark conclusion suspect
    if (y1 == 0.0 or y2 == 0.0) and outcome is not None and float(outcome) <= -25.0:
        y5 = 0.0
        y6 = 0.0

    return PRMSample(
        sample_id=f"{source_tag}_{rec['hand_id']}_d{rec['decision_idx']}",
        hand_id=str(rec["hand_id"]),
        source=f"onpolicy:{source_tag}",
        prompt_text=str(rec["prompt_text"]),
        steps=list(steps),
        action=str(action),
        labels=[y1, y2, y3, y4, y5, y6],
        meta={
            "street": rec.get("street"),
            "opponent_type": rec.get("opponent_type"),
            "hand_outcome_bb": outcome,
        },
    )


def label_onpolicy_trajectory_files(
    file_paths: list[str | Path],
    max_per_file: int | None = None,
) -> list[PRMSample]:
    """Read one or more trajectories.jsonl files and auto-label each decision into 6 steps."""
    samples: list[PRMSample] = []
    for path_like in file_paths:
        p = Path(path_like)
        if not p.exists():
            continue
        source_tag = p.parent.name or p.stem
        count = 0
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sample = _label_single_trajectory(rec, source_tag=source_tag)
                if sample is not None:
                    samples.append(sample)
                    count += 1
                    if max_per_file is not None and count >= max_per_file:
                        break
    return samples
