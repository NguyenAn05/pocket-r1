"""Hybrid Reward Aggregator with Anti-Reward-Hacking Gate and HackingMonitor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pocket_r1.env.action_abstraction import DiscreteAction
from pocket_r1.prompts.parser import ParseResult
from pocket_r1.rewards.ev_reward import compute_outcome_reward, compute_relative_ev_reward
from pocket_r1.rewards.format_reward import compute_format_reward
from pocket_r1.rewards.prm_reward import aggregate_prm_score


@dataclass
class HybridRewardConfig:
    mode: str = "hybrid"  # outcome (Exp 2) | ev (Exp 3) | hybrid (Exp 4/5)
    alpha_ev: float = 0.6
    beta_prm: float = 0.3
    gamma_format: float = 0.1
    prm_mode: str = "geom_mean"  # geom_mean | min
    anti_hack_regret_bb: float = -5.0
    anti_hack_prm_scale: float = 0.25
    soft_max_tokens: int = 550
    hard_max_tokens: int = 800


@dataclass
class RewardBreakdown:
    total: float
    r_ev: float
    r_prm: float
    r_format: float
    regret_bb: float
    hard_gate_failed: bool
    anti_hack_triggered: bool


class HackingMonitor:
    """Tracks corr(R_EV, R_DeBERTa) and False Positive rate during GRPO to catch reward hacking."""

    def __init__(self) -> None:
        self.ev_vals: list[float] = []
        self.prm_vals: list[float] = []
        self.fp_count: int = 0
        self.total_count: int = 0

    def update(self, breakdown: RewardBreakdown) -> None:
        self.total_count += 1
        self.ev_vals.append(breakdown.r_ev)
        self.prm_vals.append(breakdown.r_prm)
        if breakdown.r_prm > 0.80 and breakdown.regret_bb < -3.0:
            self.fp_count += 1

    def summary(self) -> dict[str, float]:
        if self.total_count < 2:
            return {"corr_ev_prm": 0.0, "false_positive_rate": 0.0, "count": float(self.total_count)}
        ev_arr = np.asarray(self.ev_vals, dtype=np.float64)
        prm_arr = np.asarray(self.prm_vals, dtype=np.float64)
        if np.std(ev_arr) < 1e-8 or np.std(prm_arr) < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(ev_arr, prm_arr)[0, 1])
        return {
            "corr_ev_prm": round(corr, 4),
            "false_positive_rate": round(self.fp_count / max(1, self.total_count), 4),
            "count": float(self.total_count),
        }


class HybridRewardAggregator:
    """Computes R_total = alpha * R_EV + beta * R_DeBERTa + gamma * R_format with safety gates."""

    def __init__(self, cfg: HybridRewardConfig | None = None) -> None:
        self.cfg = cfg or HybridRewardConfig()

    def score(
        self,
        parsed: ParseResult,
        pot_bb: float,
        ev_table: dict[DiscreteAction, float] | None = None,
        step_probs: list[float] | None = None,
        hand_outcome_bb: float | None = None,
    ) -> RewardBreakdown:
        r_format, hard_failed = compute_format_reward(
            parsed,
            soft_max_tokens=self.cfg.soft_max_tokens,
            hard_max_tokens=self.cfg.hard_max_tokens,
        )

        # Hard Format Gate: illegal action or broken tags -> zero PRM bonus & negative EV
        if hard_failed or parsed.action is None:
            r_ev = -1.0
            r_prm = 0.0
            total = (
                self.cfg.alpha_ev * r_ev
                + self.cfg.beta_prm * r_prm
                + self.cfg.gamma_format * r_format
            )
            return RewardBreakdown(
                total=round(total, 4),
                r_ev=r_ev,
                r_prm=r_prm,
                r_format=r_format,
                regret_bb=-10.0,
                hard_gate_failed=True,
                anti_hack_triggered=False,
            )

        # Compute primary outcome/EV signal
        regret_bb = 0.0
        if self.cfg.mode == "outcome":
            r_ev = compute_outcome_reward(hand_outcome_bb or 0.0)
            if ev_table and parsed.action in ev_table:
                _, regret_bb = compute_relative_ev_reward(parsed.action, ev_table, pot_bb)
        else:
            r_ev, regret_bb = compute_relative_ev_reward(parsed.action, ev_table or {}, pot_bb)

        # Compute DeBERTa PRM signal (only active in hybrid mode)
        anti_hack = False
        if self.cfg.mode == "hybrid" and self.cfg.beta_prm > 0.0 and step_probs:
            raw_prm = aggregate_prm_score(step_probs, mode=self.cfg.prm_mode)
            # Map [0, 1] -> [-1, +1] so scale matches R_EV and R_format cleanly
            r_prm = raw_prm
            if regret_bb <= self.cfg.anti_hack_regret_bb:
                r_prm = round(r_prm * self.cfg.anti_hack_prm_scale, 4)
                anti_hack = True
        else:
            r_prm = 0.0

        total = (
            self.cfg.alpha_ev * r_ev
            + self.cfg.beta_prm * r_prm
            + self.cfg.gamma_format * r_format
        )
        return RewardBreakdown(
            total=round(total, 4),
            r_ev=r_ev,
            r_prm=r_prm,
            r_format=r_format,
            regret_bb=regret_bb,
            hard_gate_failed=False,
            anti_hack_triggered=anti_hack,
        )
