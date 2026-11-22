"""Fast batched FP16 DeBERTa-v3 PRM Scorer for RL (Phase 4 GRPO) and Demo Visualizer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from pocket_r1.prm.modeling_deberta import build_deberta_prm_model, encode_prm_sample


@dataclass
class StepScoreResult:
    step_probs: list[float]
    min_score: float
    geom_mean: float
    first_error_step: int  # 1-indexed (1..6) of lowest-probability checkpoint


class DeBERTaPRMScorer:
    """Loads a calibrated DeBERTaStepPRM checkpoint and scores batches at high throughput."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        fp16: bool = True,
        max_length: int = 1024,
    ) -> None:
        import torch

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_amp = fp16 and self.device.type == "cuda"
        self.max_length = max_length

        # Load via factory classmethod
        dummy_cls = build_deberta_prm_model.__code__  # noqa: F841
        from transformers import AutoModel, AutoTokenizer

        root = Path(model_dir)
        import json

        meta = json.loads((root / "prm_config.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
        self.temperature = float(meta.get("temperature", 1.0))

        model = build_deberta_prm_model(str(root / "encoder"), vocab_size=meta["vocab_size"])
        model.head.load_state_dict(torch.load(root / "prm_head.pt", map_location="cpu"))
        self.model = model.to(self.device).eval()

    def score_batch(
        self,
        prompts: list[str],
        steps_list: list[list[str]],
        actions: list[str],
        batch_size: int = 32,
    ) -> list[StepScoreResult]:
        import torch

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        encoded: list[dict[str, list[int]] | None] = []
        for p, s, a in zip(prompts, steps_list, actions):
            try:
                encoded.append(encode_prm_sample(self.tokenizer, p, s, a, self.max_length))
            except ValueError:
                encoded.append(None)

        results: list[StepScoreResult | None] = [None] * len(encoded)
        valid_indices = [i for i, e in enumerate(encoded) if e is not None]

        with torch.no_grad():
            for start in range(0, len(valid_indices), batch_size):
                chunk_idx = valid_indices[start : start + batch_size]
                chunk = [encoded[i] for i in chunk_idx]
                max_len = max(len(x["input_ids"]) for x in chunk if x is not None)

                ids, mask, pos = [], [], []
                for x in chunk:
                    assert x is not None
                    pad_n = max_len - len(x["input_ids"])
                    ids.append(x["input_ids"] + [pad_id] * pad_n)
                    mask.append(x["attention_mask"] + [0] * pad_n)
                    pos.append(x["step_positions"])

                t_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
                t_mask = torch.tensor(mask, dtype=torch.long, device=self.device)
                t_pos = torch.tensor(pos, dtype=torch.long, device=self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits = self.model(t_ids, t_mask, t_pos)
                probs = torch.sigmoid(logits.float() / self.temperature).cpu().tolist()

                for i, p_vec in zip(chunk_idx, probs):
                    clamped = [max(1e-6, min(1.0 - 1e-6, float(v))) for v in p_vec]
                    min_p = min(clamped)
                    gmean = math.exp(sum(math.log(v) for v in clamped) / len(clamped))
                    worst_step = int(clamped.index(min_p)) + 1
                    results[i] = StepScoreResult(
                        step_probs=[round(v, 4) for v in clamped],
                        min_score=round(min_p, 4),
                        geom_mean=round(gmean, 4),
                        first_error_step=worst_step,
                    )

        fallback = StepScoreResult([0.0] * 6, 0.0, 0.0, 1)
        return [r if r is not None else fallback for r in results]
