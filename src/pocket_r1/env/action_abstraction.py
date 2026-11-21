"""Discrete 6-action abstraction and legal action mapping for Heads-Up NLHE."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DiscreteAction(str, Enum):
    FOLD = "FOLD"
    CHECK_CALL = "CHECK_CALL"
    BET_33 = "BET_33"
    BET_75 = "BET_75"
    BET_150 = "BET_150"
    ALL_IN = "ALL_IN"


ALL_ACTIONS: tuple[DiscreteAction, ...] = (
    DiscreteAction.FOLD,
    DiscreteAction.CHECK_CALL,
    DiscreteAction.BET_33,
    DiscreteAction.BET_75,
    DiscreteAction.BET_150,
    DiscreteAction.ALL_IN,
)

BET_POT_RATIOS: dict[DiscreteAction, float] = {
    DiscreteAction.BET_33: 0.33,
    DiscreteAction.BET_75: 0.75,
    DiscreteAction.BET_150: 1.50,
}


@dataclass(frozen=True)
class ActionSpec:
    """Resolved legal actions and their exact chip raise-to targets."""

    legal_actions: list[DiscreteAction]
    bet_targets: dict[DiscreteAction, int]


def compute_bet_target_chips(
    ratio: float,
    total_pot_chips: int,
    to_call_chips: int,
    hero_street_bet_chips: int,
    min_bet_to_chips: int,
    max_bet_to_chips: int,
) -> int:
    """Compute a raise-to chip target corresponding to `ratio * pot_after_call`.

    Formula:
        pot_after_call = total_pot_chips + to_call_chips
        extra_bet = round(ratio * pot_after_call)
        target_to = hero_street_bet_chips + to_call_chips + extra_bet
    Clamped to [min_bet_to_chips, max_bet_to_chips].
    """
    pot_after_call = total_pot_chips + to_call_chips
    extra = max(1, int(round(ratio * pot_after_call)))
    raw_target = hero_street_bet_chips + to_call_chips + extra
    return max(min_bet_to_chips, min(max_bet_to_chips, raw_target))


def get_legal_nlhe_actions(
    can_fold: bool,
    can_check_or_call: bool,
    can_bet_or_raise: bool,
    to_call_chips: int,
    total_pot_chips: int,
    hero_street_bet_chips: int,
    hero_stack_chips: int,
    min_bet_to_chips: int | None,
    max_bet_to_chips: int | None,
    all_in_merge_threshold: float = 0.85,
) -> ActionSpec:
    """Compute legal discrete actions and exact raise-to chip targets."""
    legal: list[DiscreteAction] = []
    targets: dict[DiscreteAction, int] = {}

    # Never allow FOLD when checking is free (to_call_chips == 0)
    if can_fold and to_call_chips > 0:
        legal.append(DiscreteAction.FOLD)

    if can_check_or_call:
        legal.append(DiscreteAction.CHECK_CALL)

    if (
        can_bet_or_raise
        and min_bet_to_chips is not None
        and max_bet_to_chips is not None
        and max_bet_to_chips >= min_bet_to_chips
    ):
        all_in_target = max_bet_to_chips
        merge_cutoff = hero_street_bet_chips + int(round(all_in_merge_threshold * hero_stack_chips))
        seen_targets: set[int] = set()

        for action, ratio in BET_POT_RATIOS.items():
            target = compute_bet_target_chips(
                ratio=ratio,
                total_pot_chips=total_pot_chips,
                to_call_chips=to_call_chips,
                hero_street_bet_chips=hero_street_bet_chips,
                min_bet_to_chips=min_bet_to_chips,
                max_bet_to_chips=max_bet_to_chips,
            )
            # Merge into ALL_IN if target is already close to or at max stack
            if target >= all_in_target or target >= merge_cutoff:
                continue
            if target in seen_targets:
                continue
            seen_targets.add(target)
            legal.append(action)
            targets[action] = target

        legal.append(DiscreteAction.ALL_IN)
        targets[DiscreteAction.ALL_IN] = all_in_target

    return ActionSpec(legal_actions=legal, bet_targets=targets)
