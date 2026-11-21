"""Deterministic Python verifiers for Steps 1-3 and Logic-Action Consistency."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pocket_r1.env.action_abstraction import DiscreteAction
from pocket_r1.env.base import Observation
from pocket_r1.verifiers.card_math import (
    compute_pot_odds,
    count_draws_and_outs,
    estimate_equity,
    evaluate_hand_category,
)

_UNICODE_SUIT_MAP = {
    "♠": "s",
    "♥": "h",
    "♦": "d",
    "♣": "c",
}

# Match cards like As, Kh, Td, 7c or A♠, K♥, 10♠
_CARD_RE = re.compile(r"(?<![A-Za-z0-9])(10|[2-9TJQKA])([shdc♠♥♦♣])(?![A-Za-z0-9])")

_HAND_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Straight Flush", ["straight flush", "thùng phá sảnh", "royal flush"]),
    ("Four of a Kind", ["four of a kind", "quads", "tứ quý"]),
    ("Full House", ["full house", "cù lũ"]),
    ("Flush", ["flush", "thùng"]),
    ("Straight", ["straight", "sảnh"]),
    ("Three of a Kind", ["three of a kind", "trips", "set", "sám cô", "bộ ba"]),
    ("Two Pair", ["two pair", "hai đôi", "2 đôi"]),
    ("Pair", ["pair", "đôi", "cặp"]),
    ("High Card", ["high card", "ace-high", "king-high", "queen-high", "mậu thầu", "bài cao"]),
]

_DRAW_CONTEXT_RE = re.compile(
    r"(?:draw|mua\s+(?:thùng|sảnh)|chờ\s+(?:thùng|sảnh)|flush\s+draw|straight\s+draw)",
    re.IGNORECASE,
)

_OUTS_RE = re.compile(r"(\d{1,2})\s*(?:outs?|lá\s+cơ\s+hội)", re.IGNORECASE)
_POT_ODDS_PCT_RE = re.compile(
    r"pot\s*odds[^%\n]{0,60}?(\d{1,3}(?:\.\d+)?)\s*%", re.IGNORECASE
)
_EQUITY_CMP_RE = re.compile(
    r"equity[^%\n]{0,30}?(\d{1,3}(?:\.\d+)?)\s*%\s*([<>]=?)\s*(\d{1,3}(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def extract_cards(text: str) -> list[str]:
    """Extract normalized 2-char card strings (e.g., 'As', 'Td') from text."""
    cards: list[str] = []
    for rank, suit in _CARD_RE.findall(text):
        r = "T" if rank == "10" else rank.upper()
        s = _UNICODE_SUIT_MAP.get(suit, suit.lower())
        cards.append(f"{r}{s}")
    return cards


@dataclass
class StepVerificationResult:
    step1_card_exact: bool
    step1_hallucinated_cards: list[str] = field(default_factory=list)
    step2_hand_eval_ok: bool = True
    step3_pot_odds_ok: bool = True
    step3_expected_pot_odds: float = 0.0
    step3_equity_est: float = 0.5
    logic_action_consistent: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step1_card_exact": self.step1_card_exact,
            "step1_hallucinated_cards": self.step1_hallucinated_cards,
            "step2_hand_eval_ok": self.step2_hand_eval_ok,
            "step3_pot_odds_ok": self.step3_pot_odds_ok,
            "step3_expected_pot_odds": round(self.step3_expected_pot_odds, 4),
            "step3_equity_est": round(self.step3_equity_est, 4),
            "logic_action_consistent": self.logic_action_consistent,
        }


def _verify_step1(obs: Observation, step1: str) -> tuple[bool, list[str], dict[str, Any]]:
    mentioned = extract_cards(step1)
    true_cards = set(obs.hero_cards) | set(obs.board)
    hallucinated = sorted({c for c in mentioned if c not in true_cards})
    missing_hero = [c for c in obs.hero_cards if c not in mentioned]

    # Check position contradiction if explicitly stated
    pos_ok = True
    upper = step1.upper()
    if obs.position == "BTN" and "VỊ TRÍ BB" in upper or obs.position == "BB" and "VỊ TRÍ BTN" in upper:
        pos_ok = False

    ok = (len(hallucinated) == 0) and (len(missing_hero) == 0) and pos_ok
    return ok, hallucinated, {"missing_hero": missing_hero, "pos_ok": pos_ok}


def _verify_step2(obs: Observation, step2: str) -> tuple[bool, dict[str, Any]]:
    true_cat = evaluate_hand_category(obs.hero_cards, obs.board)
    draws = count_draws_and_outs(obs.hero_cards, obs.board)
    lower = step2.lower()

    # Remove draw phrases before checking claimed made hand (so "Flush Draw" != "Flush")
    cleaned = _DRAW_CONTEXT_RE.sub("", lower)
    claimed_cats: list[str] = []
    for cat, kws in _HAND_KEYWORDS:
        if any(kw in cleaned for kw in kws):
            claimed_cats.append(cat)

    # Check if a higher made hand was hallucinated
    hand_order = [
        "High Card",
        "Pair",
        "Two Pair",
        "Three of a Kind",
        "Straight",
        "Flush",
        "Full House",
        "Four of a Kind",
        "Straight Flush",
        "Royal Flush",
    ]
    true_idx = hand_order.index(true_cat) if true_cat in hand_order else 0

    cat_ok = True
    if claimed_cats:
        # Must not claim a strictly stronger made hand than reality
        for claimed in claimed_cats:
            c_idx = hand_order.index(claimed) if claimed in hand_order else 0
            if c_idx > true_idx:
                cat_ok = False
                break

    # Check outs if mentioned
    outs_ok = True
    outs_matches = _OUTS_RE.findall(step2)
    if outs_matches and len(obs.board) in (3, 4):
        stated_outs = int(outs_matches[0])
        # Allow draw outs OR draw outs + up to 6 overcard outs, tolerance +-1
        min_expected = max(0, draws["total_draw_outs"] - 1)
        max_expected = draws["total_draw_outs"] + 6 + 1
        if draws["total_draw_outs"] == 0 and stated_outs > 7 or draws["total_draw_outs"] > 0 and not (min_expected <= stated_outs <= max_expected):
            outs_ok = False

    # Check hallucinated flush draw when no flush draw exists
    if ("flush draw" in lower or "mua thùng" in lower) and not draws["flush_draw"]:
        if "backdoor" not in lower:
            outs_ok = False

    return (cat_ok and outs_ok), {
        "true_category": true_cat,
        "claimed_categories": claimed_cats,
        "draws": draws,
        "cat_ok": cat_ok,
        "outs_ok": outs_ok,
    }


def _verify_step3(
    obs: Observation, step3: str, equity_samples: int = 120
) -> tuple[bool, float, float, dict[str, Any]]:
    expected_po = compute_pot_odds(obs.pot_bb, obs.to_call_bb)
    equity_est = estimate_equity(
        obs.hero_cards, obs.board, n_samples=equity_samples, seed=obs.seed
    )

    po_ok = True
    stated_po_pct: float | None = None
    m_po = _POT_ODDS_PCT_RE.search(step3)
    if m_po:
        stated_po_pct = float(m_po.group(1))
        expected_pct = expected_po * 100.0
        if abs(stated_po_pct - expected_pct) > 2.5:
            po_ok = False
    elif obs.to_call_bb > 0:
        # Step 3 should mention Pot Odds percentage when facing a bet
        po_ok = False

    # Verify inequality sign consistency when written like "Equity ~45% > 25%"
    cmp_ok = True
    m_cmp = _EQUITY_CMP_RE.search(step3)
    if m_cmp:
        lhs = float(m_cmp.group(1))
        op = m_cmp.group(2)
        rhs = float(m_cmp.group(3))
        if op.startswith(">") and lhs < rhs - 0.5 or op.startswith("<") and lhs > rhs + 0.5:
            cmp_ok = False

    return (po_ok and cmp_ok), expected_po, equity_est, {
        "stated_pot_odds_pct": stated_po_pct,
        "po_ok": po_ok,
        "cmp_ok": cmp_ok,
    }


def _verify_logic_action_consistency(
    step5: str, action: DiscreteAction | None
) -> bool:
    if action is None or not step5:
        return False
    lower = step5.lower()
    # Explicit fold recommendation vs aggressive/call action
    recommends_fold = any(
        kw in lower for kw in ("chọn fold", "nên fold", "bỏ bài", "quyết định fold", "action fold")
    )
    recommends_allin = any(
        kw in lower for kw in ("chọn all_in", "chọn all-in", "đẩy all-in", "nên all-in")
    )
    recommends_call = any(
        kw in lower for kw in ("chọn call", "chọn check_call", "nên call", "chọn check")
    )

    if recommends_fold and action != DiscreteAction.FOLD:
        return False
    if recommends_allin and action == DiscreteAction.FOLD:
        return False
    if recommends_call and action == DiscreteAction.FOLD and not recommends_fold:
        return False
    return True


def verify_cot_steps(
    obs: Observation,
    steps: list[str],
    action: DiscreteAction | None,
    equity_samples: int = 120,
) -> StepVerificationResult:
    """Run deterministic Python checks on Steps 1-3 and Step 5 <-> Action consistency."""
    step1 = steps[0] if len(steps) >= 1 else ""
    step2 = steps[1] if len(steps) >= 2 else ""
    step3 = steps[2] if len(steps) >= 3 else ""
    step5 = steps[4] if len(steps) >= 5 else ""

    s1_ok, hallucinated, d1 = _verify_step1(obs, step1)
    s2_ok, d2 = _verify_step2(obs, step2)
    s3_ok, expected_po, equity_est, d3 = _verify_step3(obs, step3, equity_samples=equity_samples)
    consistent = _verify_logic_action_consistency(step5, action)

    return StepVerificationResult(
        step1_card_exact=s1_ok,
        step1_hallucinated_cards=hallucinated,
        step2_hand_eval_ok=s2_ok,
        step3_pot_odds_ok=s3_ok,
        step3_expected_pot_odds=expected_po,
        step3_equity_est=equity_est,
        logic_action_consistent=consistent,
        details={"step1": d1, "step2": d2, "step3": d3},
    )
