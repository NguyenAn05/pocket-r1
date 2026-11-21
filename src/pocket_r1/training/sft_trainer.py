"""T4-friendly 4-bit QLoRA SFT trainer with exact Assistant-only Loss Masking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SFTTrainConfig:
    experiment_name: str = "exp1b_sft_fim"
    base_model: str = "Qwen/Qwen3-4B"
    enable_thinking: bool = False
    train_file: str = "data/sft/processed/train_mixed_fim.jsonl"
    val_file: str = "data/sft/processed/val_sft.jsonl"
    output_dir: str = "outputs/exp1b_sft_fim_lora"
    merged_output_dir: str | None = None
    use_4bit: bool = True
    fp16: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    learning_rate: float = 1.5e-4
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 1024
    warmup_ratio: float = 0.05
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTTrainConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**filtered)


def build_assistant_only_features(
    prompt_ids: list[int],
    full_ids: list[int],
    max_seq_length: int = 1024,
) -> dict[str, list[int]]:
    """Build input_ids, attention_mask, and labels masking all prompt tokens with -100."""
    ids = full_ids[:max_seq_length]
    prompt_len = min(len(prompt_ids), len(ids))
    labels = [-100] * prompt_len + ids[prompt_len:]
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
    }


def _load_messages_jsonl(path: str | Path) -> list[list[dict[str, str]]]:
    rows: list[list[dict[str, str]]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                rows.append(obj["messages"])
    return rows


def _apply_chat_template_safe(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> list[int]:
    """Apply chat template (tokenize=False -> encode) so return type is always list[int]."""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    return list(tokenizer.encode(rendered, add_special_tokens=False))


def train_sft_model(cfg: SFTTrainConfig) -> Path:
    """Train a 4-bit QLoRA SFT adapter on chat JSONL with assistant-only loss masking."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class PokerSFTDataset(Dataset):
        def __init__(self, jsonl_path: str | Path) -> None:
            raw_msgs = _load_messages_jsonl(jsonl_path)
            self.features: list[dict[str, list[int]]] = []
            for msgs in raw_msgs:
                prompt_ids = _apply_chat_template_safe(
                    tokenizer, msgs[:-1], add_generation_prompt=True, enable_thinking=cfg.enable_thinking
                )
                full_ids = _apply_chat_template_safe(
                    tokenizer, msgs, add_generation_prompt=False, enable_thinking=cfg.enable_thinking
                )
                feat = build_assistant_only_features(prompt_ids, full_ids, cfg.max_seq_length)
                # Only keep examples with at least 1 supervised assistant token
                if any(lbl != -100 for lbl in feat["labels"]):
                    self.features.append(feat)

        def __len__(self) -> int:
            return len(self.features)

        def __getitem__(self, idx: int) -> dict[str, list[int]]:
            return self.features[idx]

    def collate_fn(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, attn_mask, labels = [], [], []
        for x in batch:
            pad_n = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad_n)
            attn_mask.append(x["attention_mask"] + [0] * pad_n)
            labels.append(x["labels"] + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    train_ds = PokerSFTDataset(cfg.train_file)
    val_ds = PokerSFTDataset(cfg.val_file) if Path(cfg.val_file).exists() else None
    print(f"Loaded SFT datasets -> train: {len(train_ds)}, val: {len(val_ds) if val_ds else 0}")
    if len(train_ds) == 0:
        raise ValueError(f"Empty training dataset loaded from {cfg.train_file}")

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "device_map": "auto",
    }
    if cfg.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16 if cfg.fp16 else torch.bfloat16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16 if cfg.fp16 else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
    model.config.use_cache = False
    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        bf16=not cfg.fp16,
        logging_steps=25,
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=1,
        optim="paged_adamw_8bit" if cfg.use_4bit else "adamw_torch",
        seed=cfg.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )
    trainer.train()

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved LoRA adapter to {out_dir}")
    return out_dir


def merge_lora_adapter(
    base_model: str,
    adapter_dir: str | Path,
    merged_output_dir: str | Path,
) -> Path:
    """Merge a trained LoRA adapter into standalone FP16 weights for fast vLLM serving."""
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged_path = Path(merged_output_dir)
    merged_path.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(merged_path), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_path))
    print(f"Merged FP16 model saved to {merged_path}")
    return merged_path
