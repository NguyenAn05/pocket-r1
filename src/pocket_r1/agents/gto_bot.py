"""Tier-3 GTO / Minimum Defense Frequency (MDF) Approximated Bot for HU-NLHE."""

from __future__ import annotations

from pocket_r1.agents.base import AgentResult, BaseAgent
from pocket_r1.env.action_abstraction import DiscreteAction
from pocket_r1.env.base import Observation
from pocket_r1.utils.seeding import make_rng
from pocket_r1.verifiers.card_math import (
    compute_pot_odds,
    count_draws_and_outs,
    estimate_equity,
    preflop_hand_strength,
)


class GTOMDFBot(BaseAgent):
    """Balanced Tier-3 opponent using GTO preflop ranges, MDF defense, and polarized sizing."""

    def __init__(self, seed: int = 42, equity_samples: int = 64) -> None:
        self.name = "gto_mdf"
        self.rng = make_rng(seed)
        self.equity_samples = equity_samples

    def _pick_size(self, legal: list[DiscreteAction], preferred: tuple[DiscreteAction, ...]) -> DiscreteAction:
        for act in preferred:
            if act in legal:
                return act
        return DiscreteAction.CHECK_CALL

    def _act_preflop(self, obs: Observation) -> DiscreteAction:
        legal = obs.legal_actions
        s = preflop_hand_strength(obs.hero_cards)
        facing_raise = obs.to_call_bb > 0.5

        if obs.to_call_bb >= 18.0:
            # Defend vs 4-bet/all-in with top ~12% range
            if s >= 0.62:
                return DiscreteAction.CHECK_CALL
            return DiscreteAction.FOLD if DiscreteAction.FOLD in legal else DiscreteAction.CHECK_CALL

        if obs.position == "BTN" and not facing_raise:
            # GTO HU BTN opens ~75% (strong value + balanced suited/connected hands)
            if s >= 0.24:
                return self._pick_size(legal, (DiscreteAction.BET_75, DiscreteAction.BET_33))
            if self.rng.random() < 0.20:
                return DiscreteAction.CHECK_CALL
            return DiscreteAction.FOLD if DiscreteAction.FOLD in legal else DiscreteAction.CHECK_CALL

        if facing_raise:
            # BB defense vs open: polarized 3-bet top ~14% + 5% suited bluffs, call middle ~45%
            if s >= 0.58:
                return self._pick_size(legal, (DiscreteAction.BET_75, DiscreteAction.BET_150))
            if s >= 0.26:
                return DiscreteAction.CHECK_CALL
            if s >= 0.20 and self.rng.random() < 0.18:
                return self._pick_size(legal, (DiscreteAction.BET_75, DiscreteAction.BET_33))
            return DiscreteAction.FOLD if DiscreteAction.FOLD in legal else DiscreteAction.CHECK_CALL

        # Unopened / limped BB
        if s >= 0.45:
            return self._pick_size(legal, (DiscreteAction.BET_75, DiscreteAction.BET_33))
        return DiscreteAction.CHECK_CALL

    def _act_postflop(self, obs: Observation) -> DiscreteAction:
        legal = obs.legal_actions
        eq = estimate_equity(
            obs.hero_cards, obs.board, n_samples=self.equity_samples, seed=self.rng.randint(0, 10**9)
        )
        draws = count_draws_and_outs(obs.hero_cards, obs.board)
        has_strong_draw = draws["total_draw_outs"] >= 8
        pot_odds = compute_pot_odds(obs.pot_bb, obs.to_call_bb)

        if obs.to_call_bb == 0.0:
            # Polarized c-bet / value bet strategy
            if eq >= 0.72:
                return self._pick_size(legal, (DiscreteAction.BET_75, DiscreteAction.BET_150, DiscreteAction.ALL_IN))
            if eq >= 0.56:
                return self._pick_size(legal, (DiscreteAction.BET_33, DiscreteAction.BET_75))
            # Balanced semi-bluff with strong draws (2:1 value-to-bluff on 75% pot)
            if has_strong_draw and self.rng.random() < 0.45:
                return self._pick_size(legal, (DiscreteAction.BET_75, DiscreteAction.BET_33))
            if obs.street == "RIVER" and eq < 0.22 and self.rng.random() < 0.18:
                return self._pick_size(legal, (DiscreteAction.BET_75, DiscreteAction.BET_33))
            return DiscreteAction.CHECK_CALL

        # Facing a bet: apply Minimum Defense Frequency (MDF) + Equity threshold
        # MDF = Pot_before_bet / (Pot_before_bet + Bet)
        pot_before = max(0.5, obs.pot_bb - obs.to_call_bb)
        mdf = pot_before / (pot_before + obs.to_call_bb)
        defense_threshold = max(pot_odds, 1.0 - mdf)

        if eq >= 0.78:
            # Value raise / jam with monsters
            if obs.spr <= 1.5 and DiscreteAction.ALL_IN in legal:
                return DiscreteAction.ALL_IN
            return self._pick_size(legal, (DiscreteAction.BET_75, DiscreteAction.BET_150, DiscreteAction.CHECK_CALL))

        if eq >= defense_threshold or (has_strong_draw and pot_odds <= 0.36):
            # Occasional semi-bluff check-raise
            if has_strong_draw and obs.street != "RIVER" and self.rng.random() < 0.22:
                return self._pick_size(legal, (DiscreteAction.BET_75, DiscreteAction.CHECK_CALL))
            return DiscreteAction.CHECK_CALL

        return DiscreteAction.FOLD if DiscreteAction.FOLD in legal else DiscreteAction.CHECK_CALL

    def act(self, obs: Observation) -> AgentResult:
        act = self._act_preflop(obs) if obs.street == "PREFLOP" else self._act_postflop(obs)
        if act not in obs.legal_actions:
            act = (
                DiscreteAction.CHECK_CALL
                if DiscreteAction.CHECK_CALL in obs.legal_actions
                else obs.legal_actions[0]
            )
        return AgentResult(action=act, metadata={"bot_type": self.name})
