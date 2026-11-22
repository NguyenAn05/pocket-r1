"""Warm-start upgrade of DeBERTa-v3 PRM (v1 -> v2) using mined False Positives + Replay Buffer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pocket_r1.data import synthesize_gym_dataset
from pocket_r1.prm import (
    compute_step_metrics,
    encode_prm_sample,
    fit_temperature_scaling,
    generate_synthetic_prm_samples,
)
from pocket_r1.prm.calibration import sigmoid_np
from pocket_r1.prm.modeling_deberta import build_deberta_prm_model
from pocket_r1.utils.seeding import make_rng


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade DeBERTa-v3 PRM v1 -> v2 with mined FPs.")
    parser.add_argument("--init-prm-dir", type=str, required=True)
    parser.add_argument("--false-positives-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--replay-clean", type=int, default=1500)
    parser.add_argument("--replay-corrupt", type=int, default=1500)
    parser.add_argument("--fp-repeat", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=1500000)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load DeBERTa PRM v1 weights & tokenizer
    init_root = Path(args.init_prm_dir)
    meta = json.loads((init_root / "prm_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(init_root / "tokenizer"))
    model = build_deberta_prm_model(str(init_root / "encoder"), vocab_size=meta["vocab_size"])
    model.head.load_state_dict(torch.load(init_root / "prm_head.pt", map_location="cpu"))
    model = model.float().to(device)
    print(f"Loaded DeBERTa PRM v1 from {init_root} (T={meta.get('temperature', 1.0)})", flush=True)

    # 2. Load mined False Positives + build Replay Buffer (prevent catastrophic forgetting)
    fp_rows = _load_jsonl(Path(args.false_positives_file))
    rng = make_rng(args.seed)
    rng.shuffle(fp_rows)
    n_val_fp = max(1, int(round(len(fp_rows) * 0.20))) if fp_rows else 0
    val_fp_rows = fp_rows[:n_val_fp]
    train_fp_rows = fp_rows[n_val_fp:] * max(1, args.fp_repeat)

    sft_ex = synthesize_gym_dataset(
        env_name="hu_nlhe",
        target_examples=args.replay_clean,
        base_seed=args.seed,
        equity_samples=40,
    )
    clean_s, corrupt_s = generate_synthetic_prm_samples(
        sft_ex, n_corruptions=args.replay_corrupt, seed=args.seed + 1
    )
    replay_rows = [s.model_dump() for s in (clean_s + corrupt_s)]
    rng.shuffle(replay_rows)
    n_val_rep = max(1, int(round(len(replay_rows) * 0.15)))
    val_rep_rows = replay_rows[:n_val_rep]
    train_rep_rows = replay_rows[n_val_rep:]

    train_rows = train_fp_rows + train_rep_rows
    val_rows = val_fp_rows + val_rep_rows
    rng.shuffle(train_rows)
    print(
        f"Co-evolution PRM v2 data -> train: {len(train_rows)} (FPs x{args.fp_repeat}: {len(train_fp_rows)}, "
        f"replay: {len(train_rep_rows)}) | val: {len(val_rows)} (val_fp: {len(val_fp_rows)})",
        flush=True,
    )

    class _DS(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.items: list[dict[str, Any]] = []
            for r in rows:
                try:
                    enc = encode_prm_sample(
                        tokenizer, r["prompt_text"], r["steps"], r["action"], max_length=1024
                    )
                    enc["labels"] = [float(x) for x in r["labels"]]
                    self.items.append(enc)
                except ValueError:
                    continue

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return self.items[idx]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        ids, mask, pos, lbls = [], [], [], []
        for x in batch:
            pad_n = max_len - len(x["input_ids"])
            ids.append(x["input_ids"] + [pad_id] * pad_n)
            mask.append(x["attention_mask"] + [0] * pad_n)
            pos.append(x["step_positions"])
            lbls.append(x["labels"])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "step_positions": torch.tensor(pos, dtype=torch.long),
            "labels": torch.tensor(lbls, dtype=torch.float32),
        }

    train_ds = _DS(train_rows)
    val_ds = _DS(val_rows)
    val_fp_ds = _DS(val_fp_rows)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)
    val_fp_loader = (
        DataLoader(val_fp_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)
        if len(val_fp_ds) > 0
        else None
    )

    def eval_loader(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        lgs, yts = [], []
        with torch.no_grad():
            for b in loader:
                out = model(
                    b["input_ids"].to(device),
                    b["attention_mask"].to(device),
                    b["step_positions"].to(device),
                )
                lgs.append(out.cpu().numpy())
                yts.append(b["labels"].numpy())
        return np.concatenate(lgs, axis=0), np.concatenate(yts, axis=0)

    # Measure v1 BEFORE upgrade on held-out mined False Positives
    before_fp_action_err_recall = None
    if val_fp_loader is not None:
        lg_before, yt_before = eval_loader(val_fp_loader)
        m_before = compute_step_metrics(sigmoid_np(lg_before / float(meta.get("temperature", 1.0))), yt_before)
        before_fp_action_err_recall = m_before["per_step"]["action"]["recall_error"]
        print(f"DeBERTa v1 BEFORE upgrade -> mined FP action error recall: {before_fp_action_err_recall}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, total_steps // 10), total_steps)
    bce = nn.BCEWithLogitsLoss()
    eps = 0.05

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for step_idx, b in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            logits = model(
                b["input_ids"].to(device),
                b["attention_mask"].to(device),
                b["step_positions"].to(device),
            )
            smooth_y = b["labels"].to(device) * (1.0 - 2.0 * eps) + eps
            loss = bce(logits, smooth_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.item())
        val_lg, val_yt = eval_loader(val_loader)
        val_m = compute_step_metrics(sigmoid_np(val_lg), val_yt)
        print(
            f"[DeBERTa v2 Epoch {ep}/{args.epochs}] loss={loss_sum / max(1, len(train_loader)):.4f} "
            f"| val AUC={val_m['overall']['auc_roc']} | F1_err={val_m['overall']['f1_error']}",
            flush=True,
        )

    # Fit temperature scaling on val_loader & evaluate AFTER upgrade on mined FPs
    val_lg, val_yt = eval_loader(val_loader)
    t_opt = fit_temperature_scaling(val_lg, val_yt)
    val_cal = compute_step_metrics(sigmoid_np(val_lg / t_opt), val_yt)

    after_fp_action_err_recall = None
    if val_fp_loader is not None:
        lg_after, yt_after = eval_loader(val_fp_loader)
        m_after = compute_step_metrics(sigmoid_np(lg_after / t_opt), yt_after)
        after_fp_action_err_recall = m_after["per_step"]["action"]["recall_error"]

    out_dir = Path(args.output_dir)
    model.save_pretrained(out_dir, tokenizer, temperature=t_opt)
    report = {
        "upgrade": "deberta_prm_v1 -> deberta_prm_v2",
        "mined_fps_total": len(fp_rows),
        "temperature_v2": t_opt,
        "fp_action_error_recall_v1": before_fp_action_err_recall,
        "fp_action_error_recall_v2": after_fp_action_err_recall,
        "val_calibrated_v2": val_cal,
    }
    (out_dir / "prm_v2_upgrade_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Saved DeBERTa PRM v2 & upgrade report:", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
