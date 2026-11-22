"""Low-variance Relative Action EV reward (R_EV) and raw hand outcome reward (Exp 2)."""

from __future__ import annotations

import math

from pocket_r1.env.action_abstraction import DiscreteAction


def compute_relative_ev_reward(
    chosen_action: DiscreteAction,
    ev_table: dict[DiscreteAction, float],
    pot_bb: float,
) -> tuple[float, float]:
    """Compute bounded Relative Action EV reward in [-1.0, +1.0] and EV regret in BB.

    Using centered advantage (EV(a) - mean_EV) removes card-luck bias so folding 72o
    correctly and value-betting AA correctly both earn positive R_EV.
    """
    if not ev_table or chosen_action not in ev_table:
        return 0.0, 0.0

    chosen_ev = ev_table[chosen_action]
    max_ev = max(ev_table.values())
    mean_ev = sum(ev_table.values()) / len(ev_table)
    regret_bb = chosen_ev - max_ev  # <= 0.0

    scale = max(2.0, 0.5 * pot_bb)
    r_ev = math.tanh((chosen_ev - mean_ev) / scale)
    return round(r_ev, 4), round(regret_bb, 4)


def compute_outcome_reward(hand_outcome_bb: float, scale_bb: float = 15.0) -> float:
    """Raw single-hand win/loss outcome reward mapped to [-1.0, +1.0] (for Exp 2 ablation)."""
    return round(math.tanh(hand_outcome_bb / max(1.0, scale_bb)), 4)
