"""Base interface and canonical observation schema for Poker environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pocket_r1.env.action_abstraction import DiscreteAction


@dataclass(frozen=True)
class ActionHistoryEntry:
    street: str
    position: str
    action: DiscreteAction
    amount_bb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "street": self.street,
            "position": self.position,
            "action": self.action.value,
            "amount_bb": round(self.amount_bb, 2),
        }


@dataclass
class Observation:
    """Canonical decision-point observation shared by Leduc and HU-NLHE."""

    env_name: str
    hand_id: str
    seed: int
    street: str
    seat: int
    position: str
    hero_cards: list[str]
    board: list[str]
    pot_bb: float
    to_call_bb: float
    hero_stack_bb: float
    villain_stack_bb: float
    hero_street_bet_bb: float
    villain_street_bet_bb: float
    legal_actions: list[DiscreteAction]
    action_history: list[ActionHistoryEntry] = field(default_factory=list)
    villain_stats: dict[str, float] | None = None

    @property
    def effective_stack_bb(self) -> float:
        return min(self.hero_stack_bb, self.villain_stack_bb)

    @property
    def spr(self) -> float:
        denom = max(self.pot_bb, 1.0)
        return round(self.effective_stack_bb / denom, 2)

    @property
    def pot_odds(self) -> float:
        """Standard Pot Odds: to_call / (current_pot_including_villain_bet + to_call)."""
        if self.to_call_bb <= 0.0:
            return 0.0
        return self.to_call_bb / (self.pot_bb + self.to_call_bb)

    def to_state_json(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "street": self.street,
            "hero_cards": list(self.hero_cards),
            "board": list(self.board),
            "pot_bb": round(self.pot_bb, 2),
            "to_call_bb": round(self.to_call_bb, 2),
            "hero_stack_bb": round(self.hero_stack_bb, 2),
            "villain_stack_bb": round(self.villain_stack_bb, 2),
            "spr": self.spr,
            "legal_actions": [a.value for a in self.legal_actions],
            "action_history": [h.to_dict() for h in self.action_history],
            "villain_stats": self.villain_stats,
        }


class BasePokerEnv(ABC):
    """Abstract interface for 2-player Poker Gym environments."""

    env_name: str = "base"

    @abstractmethod
    def reset(
        self,
        seed: int | None = None,
        deck: list[str] | None = None,
        hand_id: str | None = None,
    ) -> Observation:
        """Reset environment for a new hand and return initial actor's observation."""

    @abstractmethod
    def step(self, action: DiscreteAction) -> tuple[Observation | None, list[float], bool, dict[str, Any]]:
        """Execute a discrete action for the current actor.

        Returns:
            (next_obs_or_none, payoffs_bb_per_seat, is_terminal, info)
        """

    @abstractmethod
    def get_observation(self, seat: int | None = None) -> Observation:
        """Build canonical Observation for the given seat (defaults to current actor)."""

    @property
    @abstractmethod
    def current_seat(self) -> int | None:
        """Return seat index (0 or 1) of current actor, or None if terminal."""

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the hand has ended."""

    @property
    @abstractmethod
    def payoffs_bb(self) -> list[float]:
        """Return net payoffs in BB for [seat_0, seat_1] (zero-sum)."""

    @abstractmethod
    def clone(self) -> BasePokerEnv:
        """Return an independent deep copy of the environment at the exact current state."""
