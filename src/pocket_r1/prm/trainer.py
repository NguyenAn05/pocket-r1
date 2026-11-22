"""Training, Temperature Calibration, Quality Gate, and Throughput Benchmark for DeBERTa-v3 PRM."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pocket_r1.prm.calibration import (
    compute_step_metrics,
    fit_temperature_scaling,
    sigmoid_np,
)
from pocket_r1.prm.modeling_deberta import (
    STEP_TOKEN,
    build_deberta_prm_model,
    encode_prm_sample,
)


@dataclass
class PRMTrainConfig:
    experiment_name: str = "deberta_prm_v1"
    backbone: str = "microsoft/deberta-v3-base"
    train_file: str = "data/prm_v1/train.jsonl"
    val_synthetic_file: str = "data/prm_v1/val_synthetic.jsonl"
    val_onpolicy_file: str = "data/prm_v1/val_onpolicy_real.jsonl"
    output_dir: str = "outputs/deberta_prm_v1"
    max_length: int = 1024
    batch_size: int = 8
    grad_accum_steps: int = 4
    epochs: int = 3
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    label_smoothing: float = 0.05
    fp16: bool = True
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> PRMTrainConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in valid})


def _load_prm_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def train_deberta_prm(cfg: PRMTrainConfig) -> dict[str, Any]:
    """Train DeBERTa-v3 Step-level PRM, calibrate temperature, and evaluate Quality Gate."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    tokenizer.add_special_tokens({"additional_special_tokens": [STEP_TOKEN]})

    class PRMDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.items: list[dict[str, Any]] = []
            for r in rows:
                try:
                    enc = encode_prm_sample(
                        tokenizer, r["prompt_text"], r["steps"], r["action"], cfg.max_length
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

    train_rows = _load_prm_jsonl(cfg.train_file)
    val_syn_rows = _load_prm_jsonl(cfg.val_synthetic_file)
    val_op_rows = _load_prm_jsonl(cfg.val_onpolicy_file)

    train_ds = PRMDataset(train_rows)
    val_syn_ds = PRMDataset(val_syn_rows)
    val_op_ds = PRMDataset(val_op_rows)
    print(
        f"PRM datasets -> train: {len(train_ds)}, "
        f"val_synthetic: {len(val_syn_ds)}, val_onpolicy_real: {len(val_op_ds)}"
    )
    if len(train_ds) == 0:
        raise ValueError(f"Empty PRM train dataset: {cfg.train_file}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_syn_loader = DataLoader(val_syn_ds, batch_size=cfg.batch_size * 2, shuffle=False, collate_fn=collate_fn) if len(val_syn_ds) else None
    val_op_loader = DataLoader(val_op_ds, batch_size=cfg.batch_size * 2, shuffle=False, collate_fn=collate_fn) if len(val_op_ds) else None

    model = build_deberta_prm_model(cfg.backbone, len(tokenizer)).to(device)

    # Mild per-step weighting for negative (error) examples so rare step errors get strong gradient
    all_train_lbls = np.array([x["labels"] for x in train_ds.items], dtype=np.float64)
    err_rates = np.clip(np.mean(all_train_lbls < 0.5, axis=0), 0.02, 0.50)
    # Weight errors slightly higher inversely proportional to error frequency (capped at 3.5x)
    err_weights = torch.tensor(
        np.clip(0.25 / err_rates, 1.0, 3.5), dtype=torch.float32, device=device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    total_steps = max(1, (len(train_loader) // max(1, cfg.grad_accum_steps)) * cfg.epochs)
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_amp = cfg.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def evaluate_loader(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        batch["step_positions"].to(device),
                    )
                all_logits.append(logits.float().cpu().numpy())
                all_labels.append(batch["labels"].numpy())
        return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    eps = cfg.label_smoothing

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step_idx, batch in enumerate(train_loader, start=1):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            pos = batch["step_positions"].to(device)
            raw_y = batch["labels"].to(device)
            # Label smoothing: 0 -> eps (0.05), 1 -> 1 - eps (0.95)
            smooth_y = raw_y * (1.0 - 2.0 * eps) + eps
            sample_w = torch.where(raw_y < 0.5, err_weights.unsqueeze(0), torch.ones_like(raw_y))

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(ids, mask, pos)
                loss_mat = bce(logits, smooth_y) * sample_w
                loss = loss_mat.mean() / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * cfg.grad_accum_steps

            if step_idx % cfg.grad_accum_steps == 0 or step_idx == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if step_idx % 200 == 0:
                print(
                    f"[Epoch {epoch}/{cfg.epochs} | Step {step_idx}/{len(train_loader)}] "
                    f"loss={running_loss / step_idx:.4f}"
                )

        # Evaluate on both validation sets; select best checkpoint by val_onpolicy_real AUC!
        primary_loader = val_op_loader if val_op_loader is not None else val_syn_loader
        if primary_loader is not None:
            val_logits, val_labels = evaluate_loader(primary_loader)
            metrics = compute_step_metrics(sigmoid_np(val_logits), val_labels)
            cur_auc = metrics["overall"]["auc_roc"] or 0.0
            print(
                f"Epoch {epoch} primary val -> AUC={cur_auc:.4f}, "
                f"F1_valid={metrics['overall']['f1_valid']:.4f}, "
                f"F1_err={metrics['overall']['f1_error']:.4f}, ECE={metrics['overall']['ece']:.4f}"
            )
            if cur_auc > best_auc:
                best_auc = cur_auc
                model.save_pretrained(out_dir, tokenizer, temperature=1.0)
        else:
            model.save_pretrained(out_dir, tokenizer, temperature=1.0)

    # Reload best checkpoint, fit Temperature Scaling on val_onpolicy_real, and save final report
    best_model, best_tok, _ = model.__class__.from_pretrained(out_dir)
    best_model = best_model.to(device)
    model = best_model

    report: dict[str, Any] = {"experiment": cfg.experiment_name, "backbone": cfg.backbone}
    temperature = 1.0

    if val_op_loader is not None:
        op_logits, op_labels = evaluate_loader(val_op_loader)
        temperature = fit_temperature_scaling(op_logits, op_labels)
        best_model.save_pretrained(out_dir, best_tok, temperature=temperature)
        report["temperature"] = temperature
        report["val_onpolicy_real_raw"] = compute_step_metrics(sigmoid_np(op_logits), op_labels)
        report["val_onpolicy_real_calibrated"] = compute_step_metrics(
            sigmoid_np(op_logits / temperature), op_labels
        )

    if val_syn_loader is not None:
        syn_logits, syn_labels = evaluate_loader(val_syn_loader)
        report["val_synthetic_calibrated"] = compute_step_metrics(
            sigmoid_np(syn_logits / temperature), syn_labels
        )

    # Inference throughput benchmark (batch_size=32)
    bench_ds = val_op_ds if len(val_op_ds) >= 64 else val_syn_ds
    if len(bench_ds) > 0:
        bench_loader = DataLoader(bench_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        n_seen = 0
        with torch.no_grad():
            for batch in bench_loader:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    _ = best_model(
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        batch["step_positions"].to(device),
                    )
                n_seen += batch["input_ids"].size(0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_gb = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
        else:
            peak_gb = 0.0
        elapsed = max(1e-6, time.perf_counter() - t0)
        report["inference_benchmark"] = {
            "num_sequences": n_seen,
            "elapsed_sec": round(elapsed, 3),
            "sequences_per_sec": round(n_seen / elapsed, 1),
            "peak_vram_gb": peak_gb,
        }

    (out_dir / "prm_eval_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Saved DeBERTa PRM report:", json.dumps(report, indent=2, ensure_ascii=False))
    return report
