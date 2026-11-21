"""End-to-end CLI to build verified Cold-start SFT + Masked Step Infilling datasets."""

from __future__ import annotations

import argparse
import json

from pocket_r1.data import build_sft_splits, synthesize_gym_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verified Cold-start SFT + FIM datasets.")
    parser.add_argument("--env", type=str, default="hu_nlhe", choices=["leduc", "hu_nlhe"])
    parser.add_argument("--target-examples", type=int, default=12000)
    parser.add_argument("--base-seed", type=int, default=700000)
    parser.add_argument("--fim-ratio", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--equity-samples", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="data/sft/processed")
    parser.add_argument(
        "--eval-suite",
        type=str,
        default="data/eval_suites/hu_nlhe_heldout_v1.jsonl",
    )
    args = parser.parse_args()

    print(f"Synthesizing {args.target_examples} verified {args.env} examples (seed={args.base_seed})...")
    raw_examples = synthesize_gym_dataset(
        env_name=args.env,
        target_examples=args.target_examples,
        base_seed=args.base_seed,
        equity_samples=args.equity_samples,
    )
    stats = build_sft_splits(
        examples=raw_examples,
        output_dir=args.output_dir,
        eval_suite_path=args.eval_suite if args.env == "hu_nlhe" else None,
        val_ratio=args.val_ratio,
        fim_ratio=args.fim_ratio,
        seed=args.base_seed,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
