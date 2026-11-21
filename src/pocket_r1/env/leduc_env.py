"""Prototype 2-player Leduc Poker environment sharing BasePokerEnv interface."""

from __future__ import annotations

from typing import Any

from pocket_r1.env.action_abstraction import DiscreteAction
from pocket_r1.env.base import ActionHistoryEntry, BasePokerEnv, Observation
from pocket_r1.utils.seeding import make_rng

LEDUC_DECK: tuple[str, ...] = ("Js", "Jh", "Qs", "Qh", "Ks", "Kh")
LEDUC_RANK_ORDER: dict[str, int] = {"J": 0, "Q": 1, "K": 2}


class LeducPokerEnv(BasePokerEnv):
    """Exact 2-player Leduc Hold'em for fast pipeline & RL prototyping.

    Rules:
    - 6-card deck: Js, Jh, Qs, Qh, Ks, Kh.
    - Both players post 1.0 BB ante.
    - Round 0 (PREFLOP): 1 private card each; bet size = 2.0 BB, max 2 raises/round.
    - Round 1 (FLOP): 1 public board card; bet size = 4.0 BB, max 2 raises/round.
    - Showdown: pair with board card beats high card; ties split pot equally.
    - Uses DiscreteAction.BET_75 to represent the fixed Leduc BET/RAISE action.
    """

    env_name = "leduc"

    def __init__(
        self,
        starting_stack_bb: float = 100.0,
        ante_bb: float = 1.0,
        preflop_bet_bb: float = 2.0,
        flop_bet_bb: float = 4.0,
        max_raises_per_round: int = 2,
    ) -> None:
        self.starting_stack_bb = starting_stack_bb
        self.ante_bb = ante_bb
        self.bet_sizes = (preflop_bet_bb, flop_bet_bb)
        self.max_raises_per_round = max_raises_per_round

        self._seed: int = 0
        self._hand_id: str = "leduc_0"
        self._deck: list[str] = list(LEDUC_DECK)
        self._hole_cards: list[str] = ["Js", "Qs"]
        self._board: list[str] = []
        self._street_idx: int = 0  # 0=PREFLOP, 1=FLOP
        self._stacks: list[float] = [starting_stack_bb, starting_stack_bb]
        self._street_bets: list[float] = [0.0, 0.0]
        self._total_contrib: list[float] = [0.0, 0.0]
        self._pot: float = 0.0
        self._raises_in_round: int = 0
        self._actions_in_round: int = 0
        self._actor: int | None = 0
        self._terminal: bool = False
        self._payoffs: list[float] = [0.0, 0.0]
        self._history: list[ActionHistoryEntry] = []

    def reset(
        self,
        seed: int | None = None,
        deck: list[str] | None = None,
        hand_id: str | None = None,
    ) -> Observation:
        self._seed = seed if seed is not None else 0
        self._hand_id = hand_id or f"leduc_{self._seed}"
        if deck is not None:
            if len(deck) < 3:
                raise ValueError("Leduc deck must contain at least 3 cards.")
            self._deck = list(deck)
        else:
            rng = make_rng(self._seed)
            self._deck = list(LEDUC_DECK)
            rng.shuffle(self._deck)

        self._hole_cards = [self._deck[0], self._deck[1]]
        self._board = []
        self._street_idx = 0
        self._stacks = [
            self.starting_stack_bb - self.ante_bb,
            self.starting_stack_bb - self.ante_bb,
        ]
        self._street_bets = [0.0, 0.0]
        self._total_contrib = [self.ante_bb, self.ante_bb]
        self._pot = 2.0 * self.ante_bb
        self._raises_in_round = 0
        self._actions_in_round = 0
        self._actor = 0  # Seat 0 acts first in Leduc
        self._terminal = False
        self._payoffs = [0.0, 0.0]
        self._history = []
        return self.get_observation(self._actor)

    @property
    def current_seat(self) -> int | None:
        return self._actor

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    @property
    def payoffs_bb(self) -> list[float]:
        return list(self._payoffs)

    def _street_name(self) -> str:
        return "PREFLOP" if self._street_idx == 0 else "FLOP"

    @staticmethod
    def _position_name(seat: int) -> str:
        return "BTN" if seat == 0 else "BB"

    def _legal_actions(self) -> list[DiscreteAction]:
        if self._terminal or self._actor is None:
            return []
        seat = self._actor
        opp = 1 - seat
        to_call = self._street_bets[opp] - self._street_bets[seat]
        actions: list[DiscreteAction] = []
        if to_call > 0:
            actions.append(DiscreteAction.FOLD)
        actions.append(DiscreteAction.CHECK_CALL)
        if self._raises_in_round < self.max_raises_per_round:
            actions.append(DiscreteAction.BET_75)
        return actions

    def get_observation(self, seat: int | None = None) -> Observation:
        target_seat = seat if seat is not None else (self._actor if self._actor is not None else 0)
        opp = 1 - target_seat
        to_call = max(0.0, self._street_bets[opp] - self._street_bets[target_seat])
        legal = self._legal_actions() if target_seat == self._actor else []
        return Observation(
            env_name=self.env_name,
            hand_id=self._hand_id,
            seed=self._seed,
            street=self._street_name(),
            seat=target_seat,
            position=self._position_name(target_seat),
            hero_cards=[self._hole_cards[target_seat]],
            board=list(self._board),
            pot_bb=self._pot,
            to_call_bb=to_call,
            hero_stack_bb=self._stacks[target_seat],
            villain_stack_bb=self._stacks[opp],
            hero_street_bet_bb=self._street_bets[target_seat],
            villain_street_bet_bb=self._street_bets[opp],
            legal_actions=legal,
            action_history=list(self._history),
        )

    def _resolve_showdown(self) -> None:
        board_rank = self._board[0][0]
        r0 = self._hole_cards[0][0]
        r1 = self._hole_cards[1][0]
        score0 = (1 if r0 == board_rank else 0, LEDUC_RANK_ORDER[r0])
        score1 = (1 if r1 == board_rank else 0, LEDUC_RANK_ORDER[r1])
        if score0 > score1:
            self._payoffs = [self._pot - self._total_contrib[0], -self._total_contrib[1]]
        elif score1 > score0:
            self._payoffs = [-self._total_contrib[0], self._pot - self._total_contrib[1]]
        else:
            half = self._pot / 2.0
            self._payoffs = [half - self._total_contrib[0], half - self._total_contrib[1]]
        self._terminal = True
        self._actor = None

    def step(
        self, action: DiscreteAction
    ) -> tuple[Observation | None, list[float], bool, dict[str, Any]]:
        if self._terminal or self._actor is None:
            raise RuntimeError("Cannot step a terminal LeducPokerEnv.")
        legal = self._legal_actions()
        if action not in legal:
            raise ValueError(f"Illegal Leduc action {action}. Legal: {legal}")

        seat = self._actor
        opp = 1 - seat
        to_call = self._street_bets[opp] - self._street_bets[seat]
        street_name = self._street_name()

        if action == DiscreteAction.FOLD:
            self._history.append(
                ActionHistoryEntry(street_name, self._position_name(seat), action, 0.0)
            )
            self._payoffs = [0.0, 0.0]
            self._payoffs[seat] = -self._total_contrib[seat]
            self._payoffs[opp] = self._pot - self._total_contrib[opp]
            self._terminal = True
            self._actor = None
            return None, list(self._payoffs), True, {"winner": opp, "reason": "fold"}

        if action == DiscreteAction.CHECK_CALL:
            paid = to_call
            self._stacks[seat] -= paid
            self._street_bets[seat] += paid
            self._total_contrib[seat] += paid
            self._pot += paid
            self._actions_in_round += 1
            self._history.append(
                ActionHistoryEntry(street_name, self._position_name(seat), action, paid)
            )
            # Round ends if facing a bet (call closes round) or both players checked
            if to_call > 0 or self._actions_in_round >= 2:
                if self._street_idx == 0:
                    self._street_idx = 1
                    self._board = [self._deck[2]]
                    self._street_bets = [0.0, 0.0]
                    self._raises_in_round = 0
                    self._actions_in_round = 0
                    self._actor = 0
                    return self.get_observation(self._actor), [0.0, 0.0], False, {}
                self._resolve_showdown()
                return None, list(self._payoffs), True, {"reason": "showdown"}
            self._actor = opp
            return self.get_observation(self._actor), [0.0, 0.0], False, {}

        # BET / RAISE (DiscreteAction.BET_75)
        bet_size = self.bet_sizes[self._street_idx]
        paid = to_call + bet_size
        self._stacks[seat] -= paid
        self._street_bets[seat] += paid
        self._total_contrib[seat] += paid
        self._pot += paid
        self._raises_in_round += 1
        self._actions_in_round += 1
        self._history.append(
            ActionHistoryEntry(street_name, self._position_name(seat), action, paid)
        )
        self._actor = opp
        return self.get_observation(self._actor), [0.0, 0.0], False, {}

    def clone(self) -> LeducPokerEnv:
        copy = LeducPokerEnv(
            starting_stack_bb=self.starting_stack_bb,
            ante_bb=self.ante_bb,
            preflop_bet_bb=self.bet_sizes[0],
            flop_bet_bb=self.bet_sizes[1],
            max_raises_per_round=self.max_raises_per_round,
        )
        copy._seed = self._seed
        copy._hand_id = self._hand_id
        copy._deck = list(self._deck)
        copy._hole_cards = list(self._hole_cards)
        copy._board = list(self._board)
        copy._street_idx = self._street_idx
        copy._stacks = list(self._stacks)
        copy._street_bets = list(self._street_bets)
        copy._total_contrib = list(self._total_contrib)
        copy._pot = self._pot
        copy._raises_in_round = self._raises_in_round
        copy._actions_in_round = self._actions_in_round
        copy._actor = self._actor
        copy._terminal = self._terminal
        copy._payoffs = list(self._payoffs)
        copy._history = list(self._history)
        return copy
