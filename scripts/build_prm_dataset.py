"""Build balanced 6-checkpoint PRM v1 dataset combining clean, synthetic RTD, and real on-policy traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pocket_r1.data import synthesize_gym_dataset
from pocket_r1.prm import (
    PRMSample,
    generate_synthetic_prm_samples,
    label_onpolicy_trajectory_files,
)
from pocket_r1.utils.seeding import make_rng


def _save_prm_jsonl(samples: list[PRMSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")


def _split_by_hand_id(
    samples: list[PRMSample], val_ratio: float, seed: int
) -> tuple[list[PRMSample], list[PRMSample]]:
    rng = make_rng(seed)
    hand_ids = sorted({s.hand_id for s in samples})
    rng.shuffle(hand_ids)
    n_val = max(1, int(round(len(hand_ids) * val_ratio))) if hand_ids else 0
    val_hands = set(hand_ids[:n_val])
    train = [s for s in samples if s.hand_id not in val_hands]
    val = [s for s in samples if s.hand_id in val_hands]
    return train, val


def _step_error_rates(samples: list[PRMSample]) -> dict[str, float]:
    if not samples:
        return {}
    n = len(samples)
    names = ["step1", "step2", "step3", "step4", "step5", "action"]
    return {
        name: round(sum(1 for s in samples if s.labels[i] < 0.5) / n, 4)
        for i, name in enumerate(names)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PRM v1 step-labeled dataset.")
    parser.add_argument("--clean-examples", type=int, default=8000)
    parser.add_argument("--synthetic-corruptions", type=int, default=10000)
    parser.add_argument("--base-seed", type=int, default=950000)
    parser.add_argument("--equity-samples", type=int, default=48)
    parser.add_argument("--val-ratio", type=float, default=0.12)
    parser.add_argument("--onpolicy-files", nargs="*", default=[])
    parser.add_argument("--output-dir", type=str, default="data/prm_v1")
    args = parser.parse_args()

    print(f"Synthesizing {args.clean_examples} verified CoT traces (seed={args.base_seed})...")
    sft_examples = synthesize_gym_dataset(
        env_name="hu_nlhe",
        target_examples=args.clean_examples,
        base_seed=args.base_seed,
        equity_samples=args.equity_samples,
    )
    clean_samples, corrupted_samples = generate_synthetic_prm_samples(
        sft_examples,
        n_corruptions=args.synthetic_corruptions,
        seed=args.base_seed + 1,
    )
    syn_all = clean_samples + corrupted_samples
    syn_train, syn_val = _split_by_hand_id(syn_all, val_ratio=args.val_ratio, seed=args.base_seed + 2)

    onpolicy_samples = label_onpolicy_trajectory_files(args.onpolicy_files)
    if onpolicy_samples:
        op_train, op_val = _split_by_hand_id(
            onpolicy_samples, val_ratio=0.20, seed=args.base_seed + 3
        )
    else:
        op_train, op_val = [], []

    train_all = syn_train + op_train
    rng = make_rng(args.base_seed + 4)
    rng.shuffle(train_all)

    out_dir = Path(args.output_dir)
    _save_prm_jsonl(train_all, out_dir / "train.jsonl")
    _save_prm_jsonl(syn_val, out_dir / "val_synthetic.jsonl")
    _save_prm_jsonl(op_val, out_dir / "val_onpolicy_real.jsonl")

    stats = {
        "train_total": len(train_all),
        "train_synthetic_clean_and_corrupted": len(syn_train),
        "train_onpolicy": len(op_train),
        "val_synthetic_total": len(syn_val),
        "val_onpolicy_real_total": len(op_val),
        "train_step_error_rates": _step_error_rates(train_all),
        "val_synthetic_error_rates": _step_error_rates(syn_val),
        "val_onpolicy_real_error_rates": _step_error_rates(op_val),
    }
    (out_dir / "stats_report.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
