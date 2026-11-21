"""Heads-Up No-Limit Texas Hold'em (HU-NLHE) environment built on PokerKit."""

from __future__ import annotations

from typing import Any

from pokerkit import Automation, NoLimitTexasHoldem

from pocket_r1.env.action_abstraction import (
    ActionSpec,
    DiscreteAction,
    get_legal_nlhe_actions,
)
from pocket_r1.env.base import ActionHistoryEntry, BasePokerEnv, Observation
from pocket_r1.utils.seeding import make_rng

RANKS: tuple[str, ...] = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
SUITS: tuple[str, ...] = ("c", "d", "h", "s")
STANDARD_DECK: tuple[str, ...] = tuple(f"{r}{s}" for r in RANKS for s in SUITS)

_AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)

# Internal representation: 2 chips == 1.0 BB (so SB = 1 chip = 0.5 BB, BB = 2 chips = 1.0 BB)
_CHIPS_PER_BB = 2


def _card_to_str(card: Any) -> str:
    return f"{card.rank.value}{card.suit.value}"


class HeadsUpNLHEEnv(BasePokerEnv):
    """Heads-Up (1v1) No-Limit Texas Hold'em with 6-action discrete abstraction.

    Seat convention (matching PokerKit 2-player heads-up):
    - Seat 0: BB (posts 1.0 BB, acts 2nd pre-flop, acts 1st post-flop)
    - Seat 1: BTN / SB (posts 0.5 BB, acts 1st pre-flop, acts 2nd post-flop)
    """

    env_name = "hu_nlhe"

    def __init__(
        self,
        starting_stack_bb: float = 100.0,
        all_in_merge_threshold: float = 0.85,
    ) -> None:
        self.starting_stack_bb = starting_stack_bb
        self.starting_stack_chips = int(round(starting_stack_bb * _CHIPS_PER_BB))
        self.all_in_merge_threshold = all_in_merge_threshold

        self._seed: int = 0
        self._hand_id: str = "nlhe_0"
        self._deck: list[str] = list(STANDARD_DECK)
        self._state: Any = None
        self._board_deal_cursor: int = 4  # cards 0..3 are hole cards; 4..8 are board
        self._burn_cursor: int = 9  # cards 9..11 used for deterministic burns
        self._history: list[ActionHistoryEntry] = []
        self._applied_actions: list[DiscreteAction] = []
        self._action_spec: ActionSpec = ActionSpec(legal_actions=[], bet_targets={})

    def _create_fresh_state(self) -> Any:
        return NoLimitTexasHoldem.create_state(
            _AUTOMATIONS,
            True,  # ante_trimming_status
            0,  # raw_antes
            (1, 2),  # raw_blinds_or_straddles (SB=1, BB=2)
            2,  # min_bet
            (self.starting_stack_chips, self.starting_stack_chips),
            2,  # player_count
        )

    def _advance_dealing(self) -> None:
        """Deal hole and board cards deterministically from self._deck until actor or terminal."""
        while self._state.status:
            if self._state.can_burn_card():
                # Burn a card outside the first 9 cards (0..3 hole, 4..8 board) so no collision
                burn_card = (
                    self._deck[self._burn_cursor]
                    if self._burn_cursor < len(self._deck)
                    else None
                )
                self._burn_cursor += 1
                if burn_card is not None:
                    self._state.burn_card(burn_card)
                else:
                    self._state.burn_card()
            elif self._state.can_deal_hole():
                # PokerKit deals 1 player's hole cards per deal_hole call (seat 0 then seat 1)
                if len(self._state.hole_cards[0]) == 0:
                    self._state.deal_hole(f"{self._deck[0]}{self._deck[1]}")
                else:
                    self._state.deal_hole(f"{self._deck[2]}{self._deck[3]}")
            elif self._state.can_deal_board():
                # Flop deals 3 cards; Turn/River deal 1 card
                count = 3 if len(self._state.board_cards) == 0 else 1
                cards = "".join(self._deck[self._board_deal_cursor : self._board_deal_cursor + count])
                self._board_deal_cursor += count
                self._state.deal_board(cards)
            else:
                break
        self._refresh_action_spec()

    def _refresh_action_spec(self) -> None:
        if not self._state.status or self._state.actor_index is None:
            self._action_spec = ActionSpec(legal_actions=[], bet_targets={})
            return
        actor = self._state.actor_index
        self._action_spec = get_legal_nlhe_actions(
            can_fold=self._state.can_fold(),
            can_check_or_call=self._state.can_check_or_call(),
            can_bet_or_raise=self._state.can_complete_bet_or_raise_to(),
            to_call_chips=int(self._state.checking_or_calling_amount or 0),
            total_pot_chips=int(self._state.total_pot_amount),
            hero_street_bet_chips=int(self._state.bets[actor]),
            hero_stack_chips=int(self._state.stacks[actor]),
            min_bet_to_chips=self._state.min_completion_betting_or_raising_to_amount,
            max_bet_to_chips=self._state.max_completion_betting_or_raising_to_amount,
            all_in_merge_threshold=self.all_in_merge_threshold,
        )

    def reset(
        self,
        seed: int | None = None,
        deck: list[str] | None = None,
        hand_id: str | None = None,
    ) -> Observation:
        self._seed = seed if seed is not None else 0
        self._hand_id = hand_id or f"nlhe_{self._seed}"
        if deck is not None:
            if len(deck) < 9:
                raise ValueError("NLHE deck must contain at least 9 cards (4 hole + 5 board).")
            self._deck = list(deck)
        else:
            rng = make_rng(self._seed)
            self._deck = list(STANDARD_DECK)
            rng.shuffle(self._deck)

        self._state = self._create_fresh_state()
        self._board_deal_cursor = 4
        self._burn_cursor = 9
        self._history = []
        self._applied_actions = []
        self._advance_dealing()
        return self.get_observation(self.current_seat)

    @property
    def current_seat(self) -> int | None:
        if self._state is None or not self._state.status:
            return None
        return self._state.actor_index

    @property
    def is_terminal(self) -> bool:
        return self._state is None or not self._state.status

    @property
    def payoffs_bb(self) -> list[float]:
        if self._state is None or self._state.status:
            return [0.0, 0.0]
        return [
            (self._state.stacks[0] - self.starting_stack_chips) / _CHIPS_PER_BB,
            (self._state.stacks[1] - self.starting_stack_chips) / _CHIPS_PER_BB,
        ]

    @property
    def deck(self) -> list[str]:
        return list(self._deck)

    def _street_name(self) -> str:
        n_board = len(self._state.board_cards)
        if n_board == 0:
            return "PREFLOP"
        if n_board == 3:
            return "FLOP"
        if n_board == 4:
            return "TURN"
        return "RIVER"

    @staticmethod
    def position_name(seat: int) -> str:
        return "BB" if seat == 0 else "BTN"

    def get_observation(self, seat: int | None = None) -> Observation:
        target_seat = seat if seat is not None else (self.current_seat if self.current_seat is not None else 0)
        opp = 1 - target_seat
        hero_cards = [_card_to_str(c) for c in self._state.hole_cards[target_seat]]
        board = [_card_to_str(c) for sub in self._state.board_cards for c in sub]
        to_call_chips = (
            int(self._state.checking_or_calling_amount or 0)
            if target_seat == self.current_seat
            else max(0, int(self._state.bets[opp]) - int(self._state.bets[target_seat]))
        )
        legal = list(self._action_spec.legal_actions) if target_seat == self.current_seat else []
        return Observation(
            env_name=self.env_name,
            hand_id=self._hand_id,
            seed=self._seed,
            street=self._street_name(),
            seat=target_seat,
            position=self.position_name(target_seat),
            hero_cards=hero_cards,
            board=board,
            pot_bb=self._state.total_pot_amount / _CHIPS_PER_BB,
            to_call_bb=to_call_chips / _CHIPS_PER_BB,
            hero_stack_bb=self._state.stacks[target_seat] / _CHIPS_PER_BB,
            villain_stack_bb=self._state.stacks[opp] / _CHIPS_PER_BB,
            hero_street_bet_bb=self._state.bets[target_seat] / _CHIPS_PER_BB,
            villain_street_bet_bb=self._state.bets[opp] / _CHIPS_PER_BB,
            legal_actions=legal,
            action_history=list(self._history),
        )

    def step(
        self, action: DiscreteAction
    ) -> tuple[Observation | None, list[float], bool, dict[str, Any]]:
        if self.is_terminal or self.current_seat is None:
            raise RuntimeError("Cannot step a terminal HeadsUpNLHEEnv.")
        if action not in self._action_spec.legal_actions:
            raise ValueError(
                f"Illegal NLHE action {action}. Legal: {self._action_spec.legal_actions}"
            )

        seat = self.current_seat
        street = self._street_name()
        pos = self.position_name(seat)
        prev_bet = int(self._state.bets[seat])

        if action == DiscreteAction.FOLD:
            self._state.fold()
            amount_bb = 0.0
        elif action == DiscreteAction.CHECK_CALL:
            call_chips = int(self._state.checking_or_calling_amount or 0)
            self._state.check_or_call()
            amount_bb = call_chips / _CHIPS_PER_BB
        else:
            target_chips = self._action_spec.bet_targets[action]
            self._state.complete_bet_or_raise_to(target_chips)
            amount_bb = (target_chips - prev_bet) / _CHIPS_PER_BB

        self._history.append(ActionHistoryEntry(street, pos, action, amount_bb))
        self._applied_actions.append(action)
        self._advance_dealing()

        if self.is_terminal:
            return None, self.payoffs_bb, True, {"street": street}
        return self.get_observation(self.current_seat), [0.0, 0.0], False, {}

    def clone(self, override_deck: list[str] | None = None) -> HeadsUpNLHEEnv:
        """Clone environment by deterministically replaying applied actions.

        Passing `override_deck` allows counterfactual rollouts (e.g., replacing
        unseen villain hole cards & future board cards for Monte Carlo EV).
        """
        copy = HeadsUpNLHEEnv(
            starting_stack_bb=self.starting_stack_bb,
            all_in_merge_threshold=self.all_in_merge_threshold,
        )
        copy.reset(
            seed=self._seed,
            deck=override_deck if override_deck is not None else list(self._deck),
            hand_id=self._hand_id,
        )
        for act in self._applied_actions:
            copy.step(act)
        return copy
