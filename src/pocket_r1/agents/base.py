"""Base interface for all Poker agents (Rule Bots, SLMs, GTO Bots)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pocket_r1.env.action_abstraction import DiscreteAction
from pocket_r1.env.base import Observation
from pocket_r1.prompts.parser import ParseResult


@dataclass
class AgentResult:
    """Output returned by an agent at a decision point."""

    action: DiscreteAction
    raw_response: str = ""
    prompt_text: str = ""
    parsed: ParseResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    name: str = "base_agent"

    @abstractmethod
    def act(self, obs: Observation) -> AgentResult:
        """Choose a legal DiscreteAction given an Observation."""
