"""CLI entrypoint to train, calibrate, and evaluate DeBERTa-v3 Step-level PRM."""

from __future__ import annotations

import argparse

from pocket_r1.prm.trainer import PRMTrainConfig, train_deberta_prm


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeBERTa-v3 Step-level PRM.")
    parser.add_argument("--config", type=str, default="configs/prm/deberta_v3_base.yaml")
    parser.add_argument("--train-file", type=str, default=None)
    parser.add_argument("--val-synthetic-file", type=str, default=None)
    parser.add_argument("--val-onpolicy-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = PRMTrainConfig.from_yaml(args.config)
    if args.train_file:
        cfg.train_file = args.train_file
    if args.val_synthetic_file:
        cfg.val_synthetic_file = args.val_synthetic_file
    if args.val_onpolicy_file:
        cfg.val_onpolicy_file = args.val_onpolicy_file
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    train_deberta_prm(cfg)


if __name__ == "__main__":
    main()
