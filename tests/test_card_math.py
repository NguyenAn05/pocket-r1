"""Comprehensive unit tests (30+ cases) for CardMath and StepVerifier."""

from __future__ import annotations

import pytest

from pocket_r1.env import DiscreteAction, Observation
from pocket_r1.verifiers import (
    compute_pot_odds,
    count_draws_and_outs,
    estimate_equity,
    evaluate_hand_category,
    preflop_hand_strength,
    verify_cot_steps,
)


def _make_obs(
    hero_cards: list[str],
    board: list[str],
    pot_bb: float = 18.0,
    to_call_bb: float = 6.0,
    position: str = "BTN",
    street: str = "FLOP",
) -> Observation:
    return Observation(
        env_name="hu_nlhe",
        hand_id="test_hand",
        seed=42,
        street=street,
        seat=1 if position == "BTN" else 0,
        position=position,
        hero_cards=hero_cards,
        board=board,
        pot_bb=pot_bb,
        to_call_bb=to_call_bb,
        hero_stack_bb=89.0,
        villain_stack_bb=83.0,
        hero_street_bet_bb=0.0,
        villain_street_bet_bb=to_call_bb,
        legal_actions=[DiscreteAction.FOLD, DiscreteAction.CHECK_CALL, DiscreteAction.BET_75, DiscreteAction.ALL_IN],
    )


@pytest.mark.parametrize(
    ("hole", "board", "expected_cat"),
    [
        (["As", "Ks"], ["Qs", "7s", "2h"], "High Card"),
        (["As", "Ah"], ["Qs", "7s", "2h"], "Pair"),
        (["Qs", "7d"], ["Qh", "7s", "2c"], "Two Pair"),
        (["7c", "7d"], ["Qs", "7s", "2h"], "Three of a Kind"),
        (["9s", "8h"], ["7d", "6c", "5s"], "Straight"),
        (["As", "Ks"], ["Qs", "7s", "2s"], "Flush"),
        (["Qc", "Qd"], ["Qs", "7s", "7h"], "Full House"),
        (["7c", "7d"], ["7h", "7s", "2h"], "Four of a Kind"),
        (["9s", "8s"], ["7s", "6s", "5s"], "Straight Flush"),
        (["As", "Ks"], [], "High Card"),
        (["Qd", "Qh"], [], "Pair"),
    ],
)
def test_evaluate_hand_category_all_ranks(
    hole: list[str], board: list[str], expected_cat: str
) -> None:
    assert evaluate_hand_category(hole, board) == expected_cat


def test_draws_outs_and_preflop_strength_and_pot_odds() -> None:
    # Nut Flush Draw = 9 outs
    d1 = count_draws_and_outs(["As", "Ks"], ["Qs", "7s", "2h"])
    assert d1["flush_draw"] is True
    assert d1["flush_outs"] == 9
    assert d1["total_draw_outs"] == 9

    # Open-Ended Straight Draw = 8 outs
    d2 = count_draws_and_outs(["9d", "8c"], ["7h", "6s", "2d"])
    assert d2["straight_draw"] == "OESD"
    assert d2["straight_outs"] == 8

    # Monster Combo Draw (Flush Draw + OESD) = 15 unique outs
    d3 = count_draws_and_outs(["9s", "8s"], ["7s", "6s", "2d"])
    assert d3["flush_draw"] is True
    assert d3["straight_draw"] == "OESD"
    assert d3["total_draw_outs"] == 15

    # Rainbow dry board = 0 draw outs
    d4 = count_draws_and_outs(["As", "Kd"], ["Qh", "7c", "2s"])
    assert d4["total_draw_outs"] == 0

    # Preflop ordering: AA > AKs > 72o
    assert preflop_hand_strength(["As", "Ah"]) > preflop_hand_strength(["As", "Ks"])
    assert preflop_hand_strength(["As", "Ks"]) > preflop_hand_strength(["7c", "2d"])

    # Pot odds locked formula
    assert compute_pot_odds(18.0, 6.0) == pytest.approx(0.25)
    assert compute_pot_odds(10.0, 0.0) == pytest.approx(0.0)

    # Equity sanity: AA on dry board >> 72o on dry board
    eq_aa = estimate_equity(["As", "Ah"], ["Kd", "7c", "2s"], n_samples=150, seed=1)
    eq_72 = estimate_equity(["7c", "2d"], ["As", "Kh", "Qs"], n_samples=150, seed=1)
    assert eq_aa > 0.75
    assert eq_72 < 0.25


@pytest.mark.parametrize(
    ("s1", "s2", "s3", "s5", "act", "exp_s1", "exp_s2", "exp_s3", "exp_cons"),
    [
        # 1. All valid (ASCII cards)
        (
            "Vị trí BTN, bài tẩy As Ks, mặt bàn Qs 7s 2h, Pot 18BB.",
            "Ace-high với Nut Flush Draw (9 outs mua thùng).",
            "Pot Odds = 6 / (18 + 6) = 25.0%. Equity ~45% > 25.0%.",
            "Có equity cao, chọn Call.",
            DiscreteAction.CHECK_CALL,
            True, True, True, True,
        ),
        # 2. All valid (Unicode suits)
        (
            "Vị trí BTN, bài tẩy A♠ K♠, mặt bàn Q♠ 7♠ 2♥, Pot 18BB.",
            "High card kết hợp flush draw 9 outs.",
            "Pot Odds = 25%. Equity 46% > 25%.",
            "Nên call lấy lá Turn.",
            DiscreteAction.CHECK_CALL,
            True, True, True, True,
        ),
        # 3. Step 1 card hallucination (wrong suit Kh instead of Ks)
        (
            "Vị trí BTN, bài tẩy As Kh, mặt bàn Qs 7s 2h.",
            "Ace-high với Flush Draw 9 outs.",
            "Pot Odds = 25%. Equity 45% > 25%.",
            "Chọn Call.",
            DiscreteAction.CHECK_CALL,
            False, True, True, True,
        ),
        # 4. Step 1 position contradiction
        (
            "Vị trí BB, bài tẩy As Ks, mặt bàn Qs 7s 2h.",
            "Ace-high với Flush Draw 9 outs.",
            "Pot Odds = 25%. Equity 45% > 25%.",
            "Chọn Call.",
            DiscreteAction.CHECK_CALL,
            False, True, True, True,
        ),
        # 5. Step 2 hallucinated made Flush (only 4 spades, not 5!)
        (
            "Vị trí BTN, bài tẩy As Ks, mặt bàn Qs 7s 2h.",
            "Hiện tại đã hoàn thành Flush (Thùng) lớn nhất.",
            "Pot Odds = 25%. Equity 85% > 25%.",
            "Chọn Raise.",
            DiscreteAction.BET_75,
            True, False, True, True,
        ),
        # 6. Step 2 hallucinated Two Pair
        (
            "Vị trí BTN, bài tẩy As Ks, mặt bàn Qs 7s 2h.",
            "Hiện đang có Two Pair mạnh.",
            "Pot Odds = 25%. Equity 60% > 25%.",
            "Chọn Call.",
            DiscreteAction.CHECK_CALL,
            True, False, True, True,
        ),
        # 7. Step 2 wildly wrong outs (22 outs)
        (
            "Vị trí BTN, bài tẩy As Ks, mặt bàn Qs 7s 2h.",
            "Ace-high có 22 outs mua bài.",
            "Pot Odds = 25%. Equity 45% > 25%.",
            "Chọn Call.",
            DiscreteAction.CHECK_CALL,
            True, False, True, True,
        ),
        # 8. Step 3 wrong Pot Odds math (40% instead of 25%)
        (
            "Vị trí BTN, bài tẩy As Ks, mặt bàn Qs 7s 2h.",
            "Ace-high với Flush Draw 9 outs.",
            "Pot Odds = 40.0%. Equity 45% > 40.0%.",
            "Chọn Call.",
            DiscreteAction.CHECK_CALL,
            True, True, False, True,
        ),
        # 9. Step 3 flipped inequality sign (45% < 25%)
        (
            "Vị trí BTN, bài tẩy As Ks, mặt bàn Qs 7s 2h.",
            "Ace-high với Flush Draw 9 outs.",
            "Pot Odds = 25.0%. Equity 45% < 25.0%.",
            "Chọn Call.",
            DiscreteAction.CHECK_CALL,
            True, True, False, True,
        ),
        # 10. Step 3 missing Pot Odds when facing bet
        (
            "Vị trí BTN, bài tẩy As Ks, mặt bàn Qs 7s 2h.",
            "Ace-high với Flush Draw 9 outs.",
            "Bài này khá đẹp nên chơi tiếp.",
            "Chọn Call.",
            DiscreteAction.CHECK_CALL,
            True, True, False, True,
        ),
        # 11. Step 5 <-> Action contradiction (recommends Fold but outputs ALL_IN)
        (
            "Vị trí BTN, bài tẩy As Ks, mặt bàn Qs 7s 2h.",
            "Ace-high với Flush Draw 9 outs.",
            "Pot Odds = 25%. Equity 45% > 25%.",
            "Bài quá yếu nên chọn Fold để bảo toàn chip.",
            DiscreteAction.ALL_IN,
            True, True, True, False,
        ),
        # 12. Step 5 <-> Action contradiction (recommends Call but outputs FOLD)
        (
            "Vị trí BTN, bài tẩy As Ks, mặt bàn Qs 7s 2h.",
            "Ace-high với Flush Draw 9 outs.",
            "Pot Odds = 25%. Equity 45% > 25%.",
            "Đủ odds nên chọn Call.",
            DiscreteAction.FOLD,
            True, True, True, False,
        ),
    ],
)
def test_step_verifier_matrix(
    s1: str,
    s2: str,
    s3: str,
    s5: str,
    act: DiscreteAction,
    exp_s1: bool,
    exp_s2: bool,
    exp_s3: bool,
    exp_cons: bool,
) -> None:
    obs = _make_obs(["As", "Ks"], ["Qs", "7s", "2h"], pot_bb=18.0, to_call_bb=6.0)
    res = verify_cot_steps(obs, [s1, s2, s3, "4. Đọc vị: bình thường.", s5], act, equity_samples=60)
    assert res.step1_card_exact is exp_s1
    assert res.step2_hand_eval_ok is exp_s2
    assert res.step3_pot_odds_ok is exp_s3
    assert res.logic_action_consistent is exp_cons
