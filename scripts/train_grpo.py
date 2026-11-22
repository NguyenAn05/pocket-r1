"""CLI entrypoint for Phase 4 GRPO RL Training (Exp 2, Exp 3, Exp 4, Exp 5)."""

from __future__ import annotations

import argparse

from pocket_r1.training.grpo_trainer import GRPOTrainConfig, train_grpo_model
from pocket_r1.training.sft_trainer import merge_lora_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPO RL training for Pocket-R1.")
    parser.add_argument("--config", type=str, default="configs/rl/exp4_grpo_hybrid.yaml")
    parser.add_argument("--init-lora-dir", type=str, default=None)
    parser.add_argument("--prm-model-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--merged-dir", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    cfg = GRPOTrainConfig.from_yaml(args.config)
    if args.init_lora_dir is not None:
        cfg.init_lora_dir = args.init_lora_dir
    if args.prm_model_dir is not None:
        cfg.prm_model_dir = args.prm_model_dir
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.num_steps is not None:
        cfg.num_steps = args.num_steps
    if args.group_size is not None:
        cfg.group_size = args.group_size

    adapter_dir = train_grpo_model(cfg)
    if args.merge:
        if not args.merged_dir:
            raise ValueError("--merge requires --merged-dir")
        merge_lora_adapter(cfg.base_model, adapter_dir, args.merged_dir)


if __name__ == "__main__":
    main()
