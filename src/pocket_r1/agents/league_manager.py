"""Tier-3 Arena League Manager with Prioritized Fictitious Self-Play (PFSP)."""

from __future__ import annotations

from dataclasses import dataclass

from pocket_r1.agents.base import BaseAgent
from pocket_r1.agents.gto_bot import GTOMDFBot
from pocket_r1.agents.rule_bots import create_bot
from pocket_r1.utils.seeding import make_rng

TIER2_BOTS: tuple[str, ...] = ("calling_station", "nit", "maniac", "tag")


@dataclass
class OpponentRecord:
    name: str
    tier: int  # 2 = exploitative bot, 3 = GTO or historical SLM
    hands: int = 0
    total_bb: float = 0.0

    @property
    def win_rate_est(self) -> float:
        """Map rolling BB/100 into a smooth [0.05, 0.95] win-rate proxy."""
        if self.hands == 0:
            return 0.5
        bb100 = (self.total_bb / self.hands) * 100.0
        # +100 BB/100 -> ~0.85, 0 -> 0.50, -100 -> ~0.15
        import math

        return max(0.05, min(0.95, 1.0 / (1.0 + math.exp(-bb100 / 60.0))))


class LeagueManager:
    """Samples opponents via PFSP while reserving >=30% weight for Tier-2 anti-forgetting."""

    def __init__(
        self,
        include_gto: bool = True,
        min_tier2_weight: float = 0.30,
        pfsp_power: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.include_gto = include_gto
        self.min_tier2_weight = min_tier2_weight
        self.pfsp_power = pfsp_power
        self.rng = make_rng(seed)

        self.records: dict[str, OpponentRecord] = {
            name: OpponentRecord(name=name, tier=2) for name in TIER2_BOTS
        }
        if include_gto:
            self.records["gto_mdf"] = OpponentRecord(name="gto_mdf", tier=3)

    def add_historical_opponent(self, name: str) -> None:
        if name not in self.records:
            self.records[name] = OpponentRecord(name=name, tier=3)

    def record_result(self, opponent_name: str, hero_payoff_bb: float) -> None:
        if opponent_name in self.records:
            rec = self.records[opponent_name]
            rec.hands += 1
            rec.total_bb += hero_payoff_bb

    def sampling_probabilities(self) -> dict[str, float]:
        """Compute PFSP weights (harder opponents sampled more) with >=30% Tier-2 floor."""
        raw: dict[str, float] = {
            name: (1.0 - rec.win_rate_est) ** self.pfsp_power
            for name, rec in self.records.items()
        }
        t2_names = [n for n, r in self.records.items() if r.tier == 2]
        t3_names = [n for n, r in self.records.items() if r.tier == 3]

        if not t3_names:
            total = sum(raw.values()) or 1.0
            return {n: raw[n] / total for n in self.records}

        t2_sum = sum(raw[n] for n in t2_names) or 1.0
        t3_sum = sum(raw[n] for n in t3_names) or 1.0
        raw_t2_share = t2_sum / (t2_sum + t3_sum)
        t2_share = max(self.min_tier2_weight, raw_t2_share)
        t3_share = 1.0 - t2_share

        probs: dict[str, float] = {}
        for n in t2_names:
            probs[n] = t2_share * (raw[n] / t2_sum)
        for n in t3_names:
            probs[n] = t3_share * (raw[n] / t3_sum)
        return probs

    def sample_opponent_name(self) -> str:
        probs = self.sampling_probabilities()
        names = list(probs.keys())
        weights = [probs[n] for n in names]
        return self.rng.choices(names, weights=weights, k=1)[0]

    def build_bot(self, name: str, seed: int, equity_samples: int = 40) -> BaseAgent:
        if name == "gto_mdf":
            return GTOMDFBot(seed=seed, equity_samples=equity_samples)
        return create_bot(name, seed=seed, equity_samples=equity_samples)
