"""Deterministic poker hand evaluation, outs counting, Pot Odds, and fast Monte Carlo equity."""

from __future__ import annotations

from collections import Counter
from typing import Any

from treys import Card, Evaluator

from pocket_r1.env.nlhe_env import STANDARD_DECK
from pocket_r1.utils.seeding import make_rng

_EVALUATOR = Evaluator()

RANK_VALUE: dict[str, int] = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

# Canonical treys rank class -> standard name
TREYS_CLASS_NAMES: dict[int, str] = {
    0: "Royal Flush",
    1: "Straight Flush",
    2: "Four of a Kind",
    3: "Full House",
    4: "Flush",
    5: "Straight",
    6: "Three of a Kind",
    7: "Two Pair",
    8: "Pair",
    9: "High Card",
}

_CHEN_BASE: dict[str, float] = {
    "A": 10.0,
    "K": 8.0,
    "Q": 7.0,
    "J": 6.0,
    "T": 5.0,
    "9": 4.5,
    "8": 4.0,
    "7": 3.5,
    "6": 3.0,
    "5": 2.5,
    "4": 2.0,
    "3": 1.5,
    "2": 1.0,
}


def compute_pot_odds(pot_bb: float, to_call_bb: float) -> float:
    """Locked Pot Odds formula: to_call / (pot_including_villain_bet + to_call)."""
    if to_call_bb <= 0.0:
        return 0.0
    return to_call_bb / (pot_bb + to_call_bb)


def preflop_hand_strength(hole_cards: list[str]) -> float:
    """Fast Chen-formula-based preflop strength normalized to [0.05, 1.0]."""
    if len(hole_cards) == 1:
        # Leduc Poker 1-card strength
        rank = hole_cards[0][0]
        return {"J": 0.30, "Q": 0.55, "K": 0.85}.get(rank, 0.5)
    if len(hole_cards) != 2:
        return 0.5

    r1, s1 = hole_cards[0][0], hole_cards[0][1]
    r2, s2 = hole_cards[1][0], hole_cards[1][1]
    v1, v2 = RANK_VALUE[r1], RANK_VALUE[r2]
    if v2 > v1:
        r1, r2 = r2, r1
        v1, v2 = v2, v1

    score = _CHEN_BASE[r1]
    if r1 == r2:
        score = max(5.0, score * 2.0)
    else:
        if s1 == s2:
            score += 2.0
        gap = v1 - v2 - 1
        if gap == 1:
            score -= 1.0
        elif gap == 2:
            score -= 2.0
        elif gap == 3:
            score -= 4.0
        elif gap >= 4:
            score -= 5.0
        if gap <= 1 and v1 < 12:
            score += 1.0

    # Map Chen range [-1.0, 20.0] -> [0.05, 1.0]
    norm = (score + 1.0) / 21.0
    return max(0.05, min(1.0, norm))


def evaluate_hand_category(hole_cards: list[str], board: list[str]) -> str:
    """Evaluate current made hand category for NLHE or Leduc."""
    if len(hole_cards) == 1:
        if board and hole_cards[0][0] == board[0][0]:
            return "Pair"
        return "High Card"

    if len(board) < 3:
        if hole_cards[0][0] == hole_cards[1][0]:
            return "Pair"
        return "High Card"

    t_hole = [Card.new(c) for c in hole_cards]
    t_board = [Card.new(c) for c in board]
    score = _EVALUATOR.evaluate(t_board, t_hole)
    rank_class = _EVALUATOR.get_rank_class(score)
    return TREYS_CLASS_NAMES.get(rank_class, "High Card")


def _has_straight_from_ranks(rank_set: set[int]) -> bool:
    ranks = set(rank_set)
    if 14 in ranks:
        ranks.add(1)
    for start in range(1, 11):
        if all((start + d) in ranks for d in range(5)):
            return True
    return False


def count_draws_and_outs(hole_cards: list[str], board: list[str]) -> dict[str, Any]:
    """Count flush draw, straight draw (OESD/gutshot), and exact unique draw outs on Flop/Turn."""
    if len(hole_cards) != 2 or len(board) not in (3, 4):
        return {
            "flush_draw": False,
            "straight_draw": None,
            "flush_outs": 0,
            "straight_outs": 0,
            "total_draw_outs": 0,
        }

    known = set(hole_cards) | set(board)
    all_cards = hole_cards + board
    suit_counts = Counter(c[1] for c in all_cards)
    hero_suits = {c[1] for c in hole_cards}

    flush_suit: str | None = None
    for suit, cnt in suit_counts.items():
        if cnt == 4 and suit in hero_suits:
            flush_suit = suit
            break

    unseen = [c for c in STANDARD_DECK if c not in known]
    flush_out_cards = (
        {c for c in unseen if c[1] == flush_suit} if flush_suit is not None else set()
    )

    current_ranks = {RANK_VALUE[c[0]] for c in all_cards}
    already_straight = _has_straight_from_ranks(current_ranks)
    straight_out_cards: set[str] = set()
    if not already_straight:
        for c in unseen:
            if _has_straight_from_ranks(current_ranks | {RANK_VALUE[c[0]]}):
                straight_out_cards.add(c)

    straight_draw_type: str | None = None
    if len(straight_out_cards) >= 8:
        straight_draw_type = "OESD"
    elif len(straight_out_cards) >= 4:
        straight_draw_type = "GUTSHOT"

    combined_outs = flush_out_cards | straight_out_cards
    return {
        "flush_draw": flush_suit is not None,
        "straight_draw": straight_draw_type,
        "flush_outs": len(flush_out_cards),
        "straight_outs": len(straight_out_cards),
        "total_draw_outs": len(combined_outs),
    }


def estimate_equity(
    hole_cards: list[str],
    board: list[str],
    n_samples: int = 200,
    seed: int | None = None,
) -> float:
    """Fast Monte Carlo equity vs random villain hand using treys."""
    if len(hole_cards) == 1:
        # Leduc fast approximation
        return preflop_hand_strength(hole_cards)

    if len(board) == 0 and n_samples <= 64:
        # Fast preflop proxy blends Chen strength toward [0.32, 0.85] HU equity range
        chen = preflop_hand_strength(hole_cards)
        return 0.30 + 0.55 * chen

    known = set(hole_cards) | set(board)
    remaining = [c for c in STANDARD_DECK if c not in known]
    t_hole = [Card.new(c) for c in hole_cards]
    t_board = [Card.new(c) for c in board]
    t_remaining = [Card.new(c) for c in remaining]

    need_board = 5 - len(board)
    need_total = 2 + need_board
    rng = make_rng(seed)

    wins = 0.0
    for _ in range(n_samples):
        drawn = rng.sample(t_remaining, need_total)
        opp_hole = drawn[:2]
        full_board = t_board + drawn[2:]
        hero_rank = _EVALUATOR.evaluate(full_board, t_hole)
        opp_rank = _EVALUATOR.evaluate(full_board, opp_hole)
        if hero_rank < opp_rank:  # lower score is stronger in treys
            wins += 1.0
        elif hero_rank == opp_rank:
            wins += 0.5
    return wins / max(1, n_samples)
