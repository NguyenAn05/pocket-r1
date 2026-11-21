"""Dataset hygiene: zero-leakage check, action balancing, hand-grouped Train/Val split & FIM mixing."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pocket_r1.data.cot_synthesizer import SFTExample
from pocket_r1.data.fim_generator import create_fim_examples
from pocket_r1.eval.duplicate_arena import load_eval_suite
from pocket_r1.utils.seeding import make_rng


def _load_heldout_seeds(eval_suite_path: str | Path | None) -> set[int]:
    if eval_suite_path is None:
        return set()
    p = Path(eval_suite_path)
    if not p.exists():
        return set()
    return {spec.seed for spec in load_eval_suite(p)}


def _balance_by_action(
    examples: list[SFTExample],
    max_action_share: float = 0.45,
    seed: int = 42,
) -> list[SFTExample]:
    """Prevent a single action (e.g., FOLD or CHECK_CALL) from dominating the SFT set."""
    if not examples:
        return []
    rng = make_rng(seed)
    by_action: dict[str, list[SFTExample]] = {}
    for ex in examples:
        by_action.setdefault(ex.action, []).append(ex)

    # Compute cap relative to total non-dominant actions
    total = len(examples)
    cap = max(50, int(round(total * max_action_share)))
    balanced: list[SFTExample] = []
    for act, group in by_action.items():
        if len(group) > cap:
            balanced.extend(rng.sample(group, cap))
        else:
            balanced.extend(group)
    rng.shuffle(balanced)
    return balanced


def _save_jsonl(examples: list[SFTExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")


def build_sft_splits(
    examples: list[SFTExample],
    output_dir: str | Path,
    eval_suite_path: str | Path | None = "data/eval_suites/hu_nlhe_heldout_v1.jsonl",
    val_ratio: float = 0.05,
    fim_ratio: float = 0.25,
    max_action_share: float = 0.45,
    seed: int = 42,
) -> dict[str, Any]:
    """Deduplicate, verify zero held-out leakage, split by hand_id, add FIM, and write JSONLs."""
    rng = make_rng(seed)
    heldout_seeds = _load_heldout_seeds(eval_suite_path)

    # 1. Zero-leakage filter + deduplication by (user_prompt, action)
    clean: list[SFTExample] = []
    seen_keys: set[tuple[str, str]] = set()
    leaked_count = 0
    for ex in examples:
        if ex.seed in heldout_seeds:
            leaked_count += 1
            continue
        key = (ex.messages[1]["content"], ex.action)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        clean.append(ex)

    assert leaked_count == 0, f"Detected {leaked_count} leaked seeds from held-out eval suite!"

    # 2. Action balancing
    balanced = _balance_by_action(clean, max_action_share=max_action_share, seed=seed)

    # 3. Hand-grouped Train / Val split (no hand_id crosses train/val boundary)
    hand_ids = sorted({ex.hand_id for ex in balanced})
    rng.shuffle(hand_ids)
    n_val_hands = max(1, int(round(len(hand_ids) * val_ratio)))
    val_hand_set = set(hand_ids[:n_val_hands])

    train_standard = [ex for ex in balanced if ex.hand_id not in val_hand_set]
    val_standard = [ex for ex in balanced if ex.hand_id in val_hand_set]

    # 4. Generate FIM examples strictly from train_standard (zero val leakage!)
    train_fim = create_fim_examples(train_standard, ratio=fim_ratio, seed=seed + 1)
    train_mixed = list(train_standard) + list(train_fim)
    rng.shuffle(train_mixed)

    out_path = Path(output_dir)
    _save_jsonl(train_standard, out_path / "train_standard.jsonl")
    _save_jsonl(train_fim, out_path / "train_fim.jsonl")
    _save_jsonl(train_mixed, out_path / "train_mixed_fim.jsonl")
    _save_jsonl(val_standard, out_path / "val_sft.jsonl")

    stats = {
        "input_examples": len(examples),
        "after_dedup_clean": len(clean),
        "after_action_balance": len(balanced),
        "heldout_overlap": leaked_count,
        "train_standard_count": len(train_standard),
        "train_fim_count": len(train_fim),
        "train_mixed_fim_count": len(train_mixed),
        "val_count": len(val_standard),
        "train_action_dist": dict(Counter(ex.action for ex in train_standard)),
        "train_street_dist": dict(Counter(ex.street for ex in train_standard)),
        "train_task_dist": dict(Counter(ex.task_type for ex in train_mixed)),
    }
    (out_path / "stats_report.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats
