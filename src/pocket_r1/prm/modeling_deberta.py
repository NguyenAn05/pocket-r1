"""DeBERTa-v3 Step-level PRM architecture with 6 [STEP] checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STEP_TOKEN = "[STEP]"
NUM_STEP_CHECKPOINTS = 6  # Steps 1..5 + Action


def format_steps_segment(steps: list[str], action: str) -> str:
    """Format 5 reasoning steps + action with [STEP] markers at each of the 6 checkpoints."""
    padded = list(steps[:5])
    while len(padded) < 5:
        padded.append("")
    parts = [f"{s.strip()} {STEP_TOKEN}" for s in padded]
    parts.append(f"Action: {action.strip()} {STEP_TOKEN}")
    return "\n".join(parts)


def encode_prm_sample(
    tokenizer: Any,
    prompt_text: str,
    steps: list[str],
    action: str,
    max_length: int = 1024,
) -> dict[str, list[int]]:
    """Tokenize (prompt_text, steps, action) ensuring ALL 6 [STEP] tokens are never truncated."""
    step_token_id = tokenizer.convert_tokens_to_ids(STEP_TOKEN)
    if step_token_id is None or step_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Special token {STEP_TOKEN} must be added to tokenizer first.")

    steps_seg = format_steps_segment(steps, action)
    steps_ids: list[int] = list(tokenizer.encode(steps_seg, add_special_tokens=False))
    prompt_ids: list[int] = list(tokenizer.encode(prompt_text, add_special_tokens=False))

    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 1
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 2

    # Reserve 3 special tokens: [CLS] prompt [SEP] steps [SEP]
    max_prompt_len = max(32, max_length - len(steps_ids) - 3)
    if len(prompt_ids) > max_prompt_len:
        # Keep the start (Game/Hole/Board/Pot) and end (Legal Actions), trim middle history
        keep_head = max_prompt_len // 2
        keep_tail = max_prompt_len - keep_head
        prompt_ids = prompt_ids[:keep_head] + prompt_ids[-keep_tail:]

    input_ids = [cls_id] + prompt_ids + [sep_id] + steps_ids + [sep_id]
    if len(input_ids) > max_length:
        # Extreme fallback: keep tail containing all 6 [STEP] markers
        input_ids = [cls_id] + input_ids[-(max_length - 1) :]

    step_positions = [idx for idx, tid in enumerate(input_ids) if tid == step_token_id]
    if len(step_positions) != NUM_STEP_CHECKPOINTS:
        raise ValueError(
            f"Expected {NUM_STEP_CHECKPOINTS} {STEP_TOKEN} tokens, found {len(step_positions)}."
        )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "step_positions": step_positions,
    }


def build_deberta_prm_model(backbone_name: str, vocab_size: int) -> Any:
    """Construct DeBERTaStepPRM nn.Module (lazy torch/transformers import for CPU tests)."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class DeBERTaStepPRM(nn.Module):
        def __init__(self, model_name: str, new_vocab_size: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.backbone_name = model_name
            self.encoder = AutoModel.from_pretrained(model_name)
            self.encoder.resize_token_embeddings(new_vocab_size)
            hidden_size = int(self.encoder.config.hidden_size)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            step_positions: torch.Tensor,
        ) -> torch.Tensor:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # (B, L, H)
            bsz = hidden.size(0)
            batch_idx = torch.arange(bsz, device=hidden.device).unsqueeze(1)
            step_hidden = hidden[batch_idx, step_positions]  # (B, 6, H)
            logits = self.head(step_hidden).squeeze(-1)  # (B, 6)
            return logits

        def save_pretrained(
            self, save_dir: str | Path, tokenizer: Any, temperature: float = 1.0
        ) -> None:
            out = Path(save_dir)
            out.mkdir(parents=True, exist_ok=True)
            self.encoder.save_pretrained(str(out / "encoder"))
            torch.save(self.head.state_dict(), out / "prm_head.pt")
            tokenizer.save_pretrained(str(out / "tokenizer"))
            meta = {
                "backbone_name": self.backbone_name,
                "vocab_size": int(self.encoder.get_input_embeddings().weight.shape[0]),
                "hidden_size": int(self.encoder.config.hidden_size),
                "step_token": STEP_TOKEN,
                "num_checkpoints": NUM_STEP_CHECKPOINTS,
                "temperature": float(temperature),
            }
            (out / "prm_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        @classmethod
        def from_pretrained(cls, load_dir: str | Path) -> tuple[Any, Any, float]:
            from transformers import AutoTokenizer

            root = Path(load_dir)
            meta = json.loads((root / "prm_config.json").read_text(encoding="utf-8"))
            tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
            model = cls(str(root / "encoder"), new_vocab_size=meta["vocab_size"])
            model.head.load_state_dict(torch.load(root / "prm_head.pt", map_location="cpu"))
            return model, tokenizer, float(meta.get("temperature", 1.0))

    return DeBERTaStepPRM(backbone_name, vocab_size)
