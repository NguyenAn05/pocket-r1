"""Generate the fixed Held-out Evaluation Suite (Duplicate Deals) for HU-NLHE."""

from __future__ import annotations

import argparse
from pathlib import Path

from pocket_r1.eval import generate_duplicate_suite, save_eval_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed HU-NLHE Held-out Eval Suite.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval_suites/hu_nlhe_heldout_v1.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--pairs-per-opponent",
        type=int,
        default=250,
        help="Number of duplicate pairs per opponent (250 pairs = 500 hands/opponent)",
    )
    parser.add_argument("--seed", type=int, default=202601, help="Base seed")
    args = parser.parse_args()

    opponents = ["calling_station", "nit", "maniac", "tag"]
    specs = generate_duplicate_suite(
        opponents=opponents,
        pairs_per_opponent=args.pairs_per_opponent,
        base_seed=args.seed,
    )
    save_eval_suite(specs, Path(args.output))
    print(
        f"Saved {len(specs)} duplicate pairs ({len(specs) * 2} total hands) to {args.output}"
    )


if __name__ == "__main__":
    main()
