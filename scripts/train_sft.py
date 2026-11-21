"""CLI entrypoint to train Exp 1 / Exp 1b Cold-start SFT with optional LoRA merge."""

from __future__ import annotations

import argparse

from pocket_r1.training import SFTTrainConfig, merge_lora_adapter, train_sft_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Cold-start SFT (QLoRA) for Pocket-R1.")
    parser.add_argument("--config", type=str, default="configs/training/sft_nlhe_fim.yaml")
    parser.add_argument("--train-file", type=str, default=None)
    parser.add_argument("--val-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--merged-dir", type=str, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--merge", action="store_true", help="Merge LoRA into FP16 weights after train")
    args = parser.parse_args()

    cfg = SFTTrainConfig.from_yaml(args.config)
    if args.train_file:
        cfg.train_file = args.train_file
    if args.val_file:
        cfg.val_file = args.val_file
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.merged_dir:
        cfg.merged_output_dir = args.merged_dir
    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr

    adapter_dir = train_sft_model(cfg)
    if args.merge:
        if not cfg.merged_output_dir:
            raise ValueError("--merge requires merged_output_dir in config or --merged-dir")
        merge_lora_adapter(cfg.base_model, adapter_dir, cfg.merged_output_dir)


if __name__ == "__main__":
    main()
