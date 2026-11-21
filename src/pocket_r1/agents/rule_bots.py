"""Four deterministic exploitative personality bots for Tier-2 Arena League."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pocket_r1.agents.base import AgentResult, BaseAgent
from pocket_r1.env.action_abstraction import DiscreteAction
from pocket_r1.env.base import Observation
from pocket_r1.utils.seeding import make_rng
from pocket_r1.verifiers.card_math import (
    compute_pot_odds,
    estimate_equity,
    preflop_hand_strength,
)


@dataclass
class BotConfig:
    name: str
    description: str = ""
    preflop_vpip_threshold: float = 0.50
    preflop_raise_threshold: float = 0.60
    postflop_call_margin: float = 0.0
    postflop_value_threshold: float = 0.65
    bluff_freq: float = 0.12
    fold_to_bet_tightness: float = 0.45
    preferred_bet_sizes: list[str] = field(default_factory=lambda: ["BET_33", "BET_75"])


DEFAULT_BOT_CONFIGS: dict[str, BotConfig] = {
    "calling_station": BotConfig(
        name="calling_station",
        description="Rarely folds, rarely raises, calls down light.",
        preflop_vpip_threshold=0.14,
        preflop_raise_threshold=0.56,
        postflop_call_margin=-0.22,
        postflop_value_threshold=0.78,
        bluff_freq=0.04,
        fold_to_bet_tightness=0.15,
        preferred_bet_sizes=["BET_33", "BET_75"],
    ),
    "nit": BotConfig(
        name="nit",
        description="Ultra-tight preflop and folds quickly postflop when missing.",
        preflop_vpip_threshold=0.44,
        preflop_raise_threshold=0.48,
        postflop_call_margin=0.14,
        postflop_value_threshold=0.72,
        bluff_freq=0.02,
        fold_to_bet_tightness=0.78,
        preferred_bet_sizes=["BET_75", "BET_150"],
    ),
    "maniac": BotConfig(
        name="maniac",
        description="Hyper-aggressive with frequent large bets and bluffs.",
        preflop_vpip_threshold=0.12,
        preflop_raise_threshold=0.24,
        postflop_call_margin=-0.12,
        postflop_value_threshold=0.50,
        bluff_freq=0.45,
        fold_to_bet_tightness=0.18,
        preferred_bet_sizes=["BET_75", "BET_150", "ALL_IN"],
    ),
    "tag": BotConfig(
        name="tag",
        description="Balanced Tight-Aggressive baseline respecting pot odds and equity.",
        preflop_vpip_threshold=0.30,
        preflop_raise_threshold=0.36,
        postflop_call_margin=0.01,
        postflop_value_threshold=0.62,
        bluff_freq=0.15,
        fold_to_bet_tightness=0.45,
        preferred_bet_sizes=["BET_33", "BET_75"],
    ),
}


def load_bot_configs(config_path: str | Path | None = None) -> dict[str, BotConfig]:
    if config_path is None:
        return dict(DEFAULT_BOT_CONFIGS)
    path = Path(config_path)
    if not path.exists():
        return dict(DEFAULT_BOT_CONFIGS)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bots_raw = raw.get("bots", {})
    configs: dict[str, BotConfig] = {}
    for name, params in bots_raw.items():
        configs[name] = BotConfig(name=name, **params)
    return configs or dict(DEFAULT_BOT_CONFIGS)


class RulePersonalityBot(BaseAgent):
    """Parameterized poker bot supporting distinct exploitable & balanced styles."""

    def __init__(
        self,
        config: BotConfig,
        seed: int = 42,
        equity_samples: int = 60,
    ) -> None:
        self.config = config
        self.name = config.name
        self.rng = make_rng(seed)
        self.equity_samples = equity_samples

    def _pick_bet_action(
        self, legal: list[DiscreteAction], strong_value: bool = False
    ) -> DiscreteAction:
        preferred = [DiscreteAction(a) for a in self.config.preferred_bet_sizes]
        candidates = [a for a in preferred if a in legal]
        if strong_value and DiscreteAction.ALL_IN in legal and self.rng.random() < 0.25:
            return DiscreteAction.ALL_IN
        if candidates:
            return self.rng.choice(candidates)
        for fallback in (
            DiscreteAction.BET_75,
            DiscreteAction.BET_33,
            DiscreteAction.BET_150,
            DiscreteAction.ALL_IN,
        ):
            if fallback in legal:
                return fallback
        return DiscreteAction.CHECK_CALL

    def _act_preflop(self, obs: Observation) -> DiscreteAction:
        legal = obs.legal_actions
        cfg = self.config
        strength = preflop_hand_strength(obs.hero_cards)
        # Positional bonus on BTN
        if obs.position == "BTN":
            strength = min(1.0, strength + 0.05)

        facing_bet = obs.to_call_bb > 0.5  # facing a raise beyond the initial 0.5BB blind
        pot_odds = compute_pot_odds(obs.pot_bb, obs.to_call_bb)

        # Facing an all-in or large raise
        if obs.to_call_bb >= 15.0:
            required = max(0.55, cfg.preflop_raise_threshold + cfg.postflop_call_margin)
            if strength >= required:
                return DiscreteAction.CHECK_CALL
            if DiscreteAction.FOLD in legal:
                return DiscreteAction.FOLD
            return DiscreteAction.CHECK_CALL

        if strength >= cfg.preflop_raise_threshold:
            if not facing_bet or strength >= cfg.preflop_raise_threshold + 0.12:
                return self._pick_bet_action(legal, strong_value=(strength >= 0.88))
            return DiscreteAction.CHECK_CALL

        if strength >= cfg.preflop_vpip_threshold:
            if facing_bet and strength < pot_odds + cfg.postflop_call_margin + 0.25:
                if self.rng.random() < cfg.fold_to_bet_tightness and DiscreteAction.FOLD in legal:
                    return DiscreteAction.FOLD
            if not facing_bet and self.rng.random() < cfg.bluff_freq:
                return self._pick_bet_action(legal)
            return DiscreteAction.CHECK_CALL

        # Weak hand preflop
        if obs.to_call_bb == 0.0:
            if self.rng.random() < cfg.bluff_freq * 0.5:
                return self._pick_bet_action(legal)
            return DiscreteAction.CHECK_CALL

        if self.rng.random() < cfg.bluff_freq * 0.6 and not facing_bet:
            return self._pick_bet_action(legal)
        if DiscreteAction.FOLD in legal:
            return DiscreteAction.FOLD
        return DiscreteAction.CHECK_CALL

    def _act_postflop(self, obs: Observation) -> DiscreteAction:
        legal = obs.legal_actions
        cfg = self.config
        eq_seed = self.rng.randint(0, 10**9)
        equity = estimate_equity(
            obs.hero_cards, obs.board, n_samples=self.equity_samples, seed=eq_seed
        )
        pot_odds = compute_pot_odds(obs.pot_bb, obs.to_call_bb)

        if obs.to_call_bb == 0.0:
            if equity >= cfg.postflop_value_threshold:
                return self._pick_bet_action(legal, strong_value=(equity >= 0.85))
            if self.rng.random() < cfg.bluff_freq:
                return self._pick_bet_action(legal)
            return DiscreteAction.CHECK_CALL

        # Facing a bet
        required_equity = max(0.05, pot_odds + cfg.postflop_call_margin)
        if equity >= cfg.postflop_value_threshold + 0.10:
            # Raise for value (calling stations mostly just call)
            if self.name == "calling_station" and self.rng.random() < 0.75:
                return DiscreteAction.CHECK_CALL
            return self._pick_bet_action(legal, strong_value=(equity >= 0.88))

        if equity >= required_equity:
            if self.rng.random() < cfg.bluff_freq * 0.35 and equity < 0.55:
                return self._pick_bet_action(legal)
            return DiscreteAction.CHECK_CALL

        # Below required equity: fold or occasional maniac bluff-raise / station call
        if self.name == "calling_station" and self.rng.random() > cfg.fold_to_bet_tightness:
            return DiscreteAction.CHECK_CALL
        if self.rng.random() < cfg.bluff_freq * 0.25 and obs.to_call_bb < obs.hero_stack_bb * 0.4:
            return self._pick_bet_action(legal)
        if DiscreteAction.FOLD in legal:
            return DiscreteAction.FOLD
        return DiscreteAction.CHECK_CALL

    def act(self, obs: Observation) -> AgentResult:
        if obs.street == "PREFLOP":
            action = self._act_preflop(obs)
        else:
            action = self._act_postflop(obs)
        if action not in obs.legal_actions:
            action = (
                DiscreteAction.CHECK_CALL
                if DiscreteAction.CHECK_CALL in obs.legal_actions
                else obs.legal_actions[0]
            )
        return AgentResult(action=action, metadata={"bot_type": self.name})


def create_bot(
    name: str,
    seed: int = 42,
    config_path: str | Path | None = None,
    equity_samples: int = 60,
) -> RulePersonalityBot:
    configs = load_bot_configs(config_path)
    if name not in configs:
        raise KeyError(f"Unknown bot '{name}'. Available: {list(configs)}")
    return RulePersonalityBot(configs[name], seed=seed, equity_samples=equity_samples)
