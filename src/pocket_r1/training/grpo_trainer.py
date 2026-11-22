"""Dual-GPU T4-optimized PyTorch + PEFT QLoRA GRPO Trainer for Pocket-R1 (Exp 2-5)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pocket_r1.agents.league_manager import LeagueManager
from pocket_r1.env import DiscreteAction, HeadsUpNLHEEnv, Observation
from pocket_r1.prompts import SYSTEM_PROMPT, format_observation, parse_poker_cot
from pocket_r1.rewards import (
    HackingMonitor,
    HybridRewardAggregator,
    HybridRewardConfig,
)
from pocket_r1.training.coevolution import (
    mine_false_positives,
    rejection_sample_top_trajectories,
    save_jsonl_dicts,
)
from pocket_r1.utils.seeding import make_rng
from pocket_r1.verifiers.ev_calculator import EVCalculator


@dataclass
class GRPOTrainConfig:
    experiment_name: str = "exp4_grpo_hybrid"
    reward_mode: str = "hybrid"  # outcome | ev | hybrid
    base_model: str = "Qwen/Qwen3-4B"
    init_lora_dir: str | None = None
    prm_model_dir: str | None = None
    output_dir: str = "outputs/exp4_grpo_hybrid_lora"
    alpha_ev: float = 0.6
    beta_prm: float = 0.3
    gamma_format: float = 0.1
    prm_mode: str = "geom_mean"
    anti_hack_regret_bb: float = -5.0
    anti_hack_prm_scale: float = 0.25
    use_tier3_league: bool = False
    group_size: int = 4
    num_steps: int = 120
    prompts_per_step: int = 4
    learning_rate: float = 1.0e-5
    kl_coef: float = 0.03
    clip_eps: float = 0.2
    temperature: float = 0.85
    max_new_tokens: int = 260
    ev_rollouts: int = 16
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> GRPOTrainConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in valid})


def compute_group_advantages(rewards: list[float]) -> list[float]:
    """GRPO group-relative advantage normalization: (R_i - mean) / (std + 1e-6)."""
    if not rewards:
        return []
    arr = np.asarray(rewards, dtype=np.float64)
    std = float(np.std(arr))
    if std < 1e-6:
        return [0.0] * len(rewards)
    mean = float(np.mean(arr))
    return [float((r - mean) / (std + 1e-6)) for r in arr]


def _sample_decision_spot(
    league: LeagueManager,
    ev_calc: EVCalculator,
    spot_seed: int,
    ev_rollouts: int,
) -> dict[str, Any]:
    """Play a partial hand to a realistic decision point and precompute its 6-action EV table."""
    rng = make_rng(spot_seed)
    opp_name = league.sample_opponent_name()
    opp_bot = league.build_bot(opp_name, seed=spot_seed + 1, equity_samples=24)
    hero_guide = league.build_bot("tag", seed=spot_seed + 2, equity_samples=24)

    env = HeadsUpNLHEEnv(starting_stack_bb=100.0)
    hand_id = f"grpo_spot_{spot_seed}"
    obs = env.reset(seed=spot_seed, hand_id=hand_id)
    hero_seat = spot_seed % 2
    target_decision = rng.randint(0, 3)
    hero_dec_seen = 0

    chosen_env: HeadsUpNLHEEnv | None = None
    chosen_obs: Observation | None = None

    while not env.is_terminal:
        assert obs is not None
        if obs.seat == hero_seat:
            if hero_dec_seen == target_decision:
                chosen_env = env.clone()
                chosen_obs = obs
                break
            hero_dec_seen += 1
            act = hero_guide.act(obs).action
        else:
            act = opp_bot.act(obs).action
        obs, _, _, _ = env.step(act)

    if chosen_env is None or chosen_obs is None:
        # Fallback to first hero decision point of a fresh hand
        env = HeadsUpNLHEEnv(starting_stack_bb=100.0)
        obs = env.reset(seed=spot_seed + 77777, hand_id=hand_id)
        while not env.is_terminal and obs is not None and obs.seat != hero_seat:
            obs, _, _, _ = env.step(opp_bot.act(obs).action)
        chosen_env = env.clone()
        chosen_obs = env.get_observation(hero_seat)

    # Precompute 6-action EV table ONCE -> O(1) lookup for all G rollouts!
    ev_table = ev_calc.compute_action_evs(
        env=chosen_env,
        hero_seat=hero_seat,
        opponent=opp_bot,
        n_rollouts=ev_rollouts,
        seed=spot_seed + 17,
    )
    return {
        "hand_id": hand_id,
        "opponent_name": opp_name,
        "opponent_bot": opp_bot,
        "hero_seat": hero_seat,
        "env": chosen_env,
        "obs": chosen_obs,
        "ev_table": ev_table,
    }


def _simulate_single_outcome(
    env: HeadsUpNLHEEnv,
    hero_seat: int,
    action: DiscreteAction,
    opp_bot: Any,
    seed: int,
) -> float:
    """Roll out 1 single noisy hand outcome (used strictly for Exp 2 Outcome-only ablation)."""
    from pocket_r1.agents.rule_bots import create_bot

    sim = env.clone()
    sim.step(action)
    hero_follow = create_bot("tag", seed=seed + 3, equity_samples=20)
    steps = 0
    while not sim.is_terminal and steps < 30:
        seat = sim.current_seat
        assert seat is not None
        obs = sim.get_observation(seat)
        ag = hero_follow if seat == hero_seat else opp_bot
        sim.step(ag.act(obs).action)
        steps += 1
    return sim.payoffs_bb[hero_seat]


def _render_prompt_text(tokenizer: Any, obs: Observation) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(obs)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def train_grpo_model(cfg: GRPOTrainConfig) -> Path:
    """Run GRPO with Dual-GPU 2x T4 placement (GPU 0: QLoRA Actor, GPU 1: DeBERTa PRM)."""
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    actor_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prm_device = (
        "cuda:1"
        if torch.cuda.device_count() >= 2
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_cfg,
        attn_implementation="sdpa",
        device_map={"": actor_device},
        trust_remote_code=True,
    )
    base_model.config.use_cache = True
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    if cfg.init_lora_dir and Path(cfg.init_lora_dir).exists():
        print(f"Initializing GRPO actor from LoRA adapter: {cfg.init_lora_dir}")
        model = PeftModel.from_pretrained(base_model, cfg.init_lora_dir, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # Load DeBERTa-v3 PRM Scorer on GPU 1 (FP16, ~1.5GB VRAM) if hybrid mode
    prm_scorer = None
    if cfg.reward_mode == "hybrid" and cfg.prm_model_dir and Path(cfg.prm_model_dir).exists():
        from pocket_r1.prm.scorer import DeBERTaPRMScorer

        print(f"Loading DeBERTa-v3 PRM Scorer on {prm_device} from {cfg.prm_model_dir}")
        prm_scorer = DeBERTaPRMScorer(cfg.prm_model_dir, device=prm_device, fp16=True)

    reward_agg = HybridRewardAggregator(
        HybridRewardConfig(
            mode=cfg.reward_mode,
            alpha_ev=cfg.alpha_ev,
            beta_prm=cfg.beta_prm,
            gamma_format=cfg.gamma_format,
            prm_mode=cfg.prm_mode,
            anti_hack_regret_bb=cfg.anti_hack_regret_bb,
            anti_hack_prm_scale=cfg.anti_hack_prm_scale,
        )
    )
    monitor = HackingMonitor()
    league = LeagueManager(include_gto=cfg.use_tier3_league, seed=cfg.seed)
    ev_calc = EVCalculator(default_rollouts=cfg.ev_rollouts, rollout_equity_samples=16)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=0.01,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "grpo_metrics.jsonl"
    all_rollout_logs: list[dict[str, Any]] = []

    def _completion_logprobs(seq_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Compute per-token logprobs over completion tokens seq_ids[:, prompt_len:]."""
        logits = model(input_ids=seq_ids).logits[:, prompt_len - 1 : -1, :]
        target = seq_ids[:, prompt_len:]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

    for step in range(1, cfg.num_steps + 1):
        import time

        t_step = time.perf_counter()
        optimizer.zero_grad()
        step_rewards: list[float] = []
        step_ev_rewards: list[float] = []
        step_prm_rewards: list[float] = []
        step_kls: list[float] = []
        step_loss_val = 0.0
        n_updates = 0

        for p_idx in range(cfg.prompts_per_step):
            spot_seed = 1_200_000 + cfg.seed * 10_000 + step * 100 + p_idx
            spot = _sample_decision_spot(league, ev_calc, spot_seed, cfg.ev_rollouts)
            obs: Observation = spot["obs"]
            prompt_text_plain = format_observation(obs)
            rendered = _render_prompt_text(tokenizer, obs)

            enc = tokenizer(rendered, return_tensors="pt").to(actor_device)
            prompt_len = int(enc["input_ids"].shape[1])

            # 1. Generate G rollouts on GPU 0 (disable grad checkpointing so KV-cache works at full speed!)
            model.eval()
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
            with torch.no_grad():
                gen_ids = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=0.95,
                    max_new_tokens=cfg.max_new_tokens,
                    num_return_sequences=cfg.group_size,
                    pad_token_id=tokenizer.pad_token_id,
                )

            comp_texts = [
                tokenizer.decode(gen_ids[i, prompt_len:], skip_special_tokens=True)
                for i in range(cfg.group_size)
            ]
            parsed_list = [parse_poker_cot(txt, obs.legal_actions) for txt in comp_texts]

            # 2. Score PRM in one batch on GPU 1
            prm_results = None
            if prm_scorer is not None:
                prm_results = prm_scorer.score_batch(
                    prompts=[prompt_text_plain] * cfg.group_size,
                    steps_list=[p.steps for p in parsed_list],
                    actions=[p.action.value if p.action else "INVALID" for p in parsed_list],
                )

            # 3. Compute Hybrid Rewards
            group_rewards: list[float] = []
            for i, parsed in enumerate(parsed_list):
                outcome_bb = None
                if cfg.reward_mode == "outcome" and parsed.action is not None:
                    outcome_bb = _simulate_single_outcome(
                        spot["env"], spot["hero_seat"], parsed.action, spot["opponent_bot"], spot_seed + i
                    )
                step_probs = prm_results[i].step_probs if prm_results is not None else None
                bd = reward_agg.score(
                    parsed=parsed,
                    pot_bb=obs.pot_bb,
                    ev_table=spot["ev_table"],
                    step_probs=step_probs,
                    hand_outcome_bb=outcome_bb,
                )
                monitor.update(bd)
                group_rewards.append(bd.total)
                step_rewards.append(bd.total)
                step_ev_rewards.append(bd.r_ev)
                step_prm_rewards.append(bd.r_prm)

                all_rollout_logs.append(
                    {
                        "step": step,
                        "hand_id": spot["hand_id"],
                        "opponent": spot["opponent_name"],
                        "prompt_text": prompt_text_plain,
                        "steps": parsed.steps,
                        "action": parsed.action.value if parsed.action else None,
                        "total_reward": bd.total,
                        "r_ev": bd.r_ev,
                        "r_prm": bd.r_prm,
                        "r_format": bd.r_format,
                        "regret_bb": bd.regret_bb,
                        "hard_gate_failed": bd.hard_gate_failed,
                    }
                )

            advantages = compute_group_advantages(group_rewards)
            if all(abs(a) < 1e-6 for a in advantages):
                continue

            # 4. Compute reference logprobs (base model with LoRA disabled) & GRPO policy gradient
            model.train()
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            for i in range(cfg.group_size):
                adv = advantages[i]
                if abs(adv) < 1e-6:
                    continue
                seq = gen_ids[i : i + 1]
                # Mask padding tokens after EOS in completion
                comp_tokens = seq[:, prompt_len:]
                valid_mask = (comp_tokens != tokenizer.pad_token_id).float()
                denom = valid_mask.sum().clamp(min=1.0)

                with torch.no_grad():
                    with model.disable_adapter():
                        ref_lp = _completion_logprobs(seq, prompt_len)

                cur_lp = _completion_logprobs(seq, prompt_len)
                # Approximate KL(pi || ref) = exp(ref - cur) - (ref - cur) - 1 (Schulman unbiased est.)
                log_ratio = (ref_lp - cur_lp).clamp(-10.0, 10.0)
                kl = (torch.exp(log_ratio) - log_ratio - 1.0) * valid_mask
                kl_mean = kl.sum() / denom

                pg_loss = -(adv * (cur_lp * valid_mask).sum() / denom)
                loss = (pg_loss + cfg.kl_coef * kl_mean) / (cfg.prompts_per_step * cfg.group_size)
                loss.backward()

                step_loss_val += float(loss.item())
                step_kls.append(float(kl_mean.item()))
                n_updates += 1

        if n_updates > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

        mon = monitor.summary()
        log_row = {
            "step": step,
            "mean_reward": round(float(np.mean(step_rewards)) if step_rewards else 0.0, 4),
            "mean_ev_reward": round(float(np.mean(step_ev_rewards)) if step_ev_rewards else 0.0, 4),
            "mean_prm_reward": round(float(np.mean(step_prm_rewards)) if step_prm_rewards else 0.0, 4),
            "mean_kl": round(float(np.mean(step_kls)) if step_kls else 0.0, 4),
            "loss": round(step_loss_val, 4),
            "corr_ev_prm": mon["corr_ev_prm"],
            "false_positive_rate": mon["false_positive_rate"],
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_row) + "\n")

        elapsed_step = time.perf_counter() - t_step
        print(
            f"[GRPO Step {step}/{cfg.num_steps} | {elapsed_step:.1f}s] "
            f"R={log_row['mean_reward']:+.3f} (EV={log_row['mean_ev_reward']:+.3f}, "
            f"PRM={log_row['mean_prm_reward']:.3f}) | KL={log_row['mean_kl']:.4f} | "
            f"corr={log_row['corr_ev_prm']:+.3f} | FP={log_row['false_positive_rate']:.3f}",
            flush=True,
        )

    # Save trained GRPO LoRA adapter + Co-evolution artifacts (mined FPs & RFT candidates)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    fps = mine_false_positives(all_rollout_logs)
    save_jsonl_dicts([s.model_dump() for s in fps], out_dir / "mined_false_positives.jsonl")
    rft = rejection_sample_top_trajectories(all_rollout_logs)
    save_jsonl_dicts(rft, out_dir / "rft_candidates.jsonl")
    print(
        f"GRPO complete! Saved adapter to {out_dir} | "
        f"mined FPs: {len(fps)} | RFT candidates: {len(rft)}"
    )
    return out_dir
