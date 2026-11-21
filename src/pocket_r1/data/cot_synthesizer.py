"""Verified 5-step Poker CoT synthesizer grounded in PokerKit + CardMath."""

from __future__ import annotations

import random
from typing import Any

from pydantic import BaseModel, Field

from pocket_r1.agents.rule_bots import create_bot
from pocket_r1.env import DiscreteAction, HeadsUpNLHEEnv, LeducPokerEnv, Observation
from pocket_r1.prompts.parser import parse_poker_cot
from pocket_r1.prompts.state_formatter import estimate_tokens, format_observation
from pocket_r1.prompts.templates import SYSTEM_PROMPT
from pocket_r1.utils.seeding import make_rng
from pocket_r1.verifiers.card_math import (
    compute_pot_odds,
    count_draws_and_outs,
    estimate_equity,
    evaluate_hand_category,
    preflop_hand_strength,
)
from pocket_r1.verifiers.step_verifier import verify_cot_steps


class SFTExample(BaseModel):
    """Canonical SFT training example (standard 5-step CoT or Masked Step Infilling)."""

    example_id: str
    hand_id: str
    env_name: str
    task_type: str = "standard_cot"  # standard_cot | fim_step2 | fim_step3
    seed: int
    street: str
    action: str
    opponent_type: str
    messages: list[dict[str, str]]
    steps: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


_HAND_VI_NAMES: dict[str, str] = {
    "High Card": "High Card (mậu thầu)",
    "Pair": "Pair (một đôi)",
    "Two Pair": "Two Pair (hai đôi)",
    "Three of a Kind": "Three of a Kind (sám cô/set)",
    "Straight": "Straight (sảnh)",
    "Flush": "Flush (thùng)",
    "Full House": "Full House (cù lũ)",
    "Four of a Kind": "Four of a Kind (tứ quý)",
    "Straight Flush": "Straight Flush (thùng phá sảnh)",
    "Royal Flush": "Royal Flush (thùng phá sảnh lớn)",
}

_OPP_READ_TEMPLATES: dict[str, list[str]] = {
    "calling_station": [
        "Đối thủ có xu hướng theo bài rất rộng và ít bỏ bài; ưu tiên value bet khi có bài mạnh và hạn chế bluff.",
        "Lịch sử cược cho thấy đối thủ hay call với bài yếu hoặc draw; chiến thuật tối ưu là khai thác giá trị mỏng.",
        "Đối thủ thiên về phòng thủ thụ động (calling station), nên tập trung đánh theo sức mạnh bài thật.",
    ],
    "nit": [
        "Đối thủ chơi rất chặt (nit), thường chỉ tiếp tục khi có bài mạnh; có thể gây áp lực cướp pot khi đối thủ tỏ ra yếu.",
        "Range đối thủ hẹp và nhạy với áp lực cược; tôn trọng khi đối thủ tố mạnh nhưng tận dụng vị trí để tấn công pot nhỏ.",
        "Đối thủ kén bài, dễ bỏ bài trên mặt bàn khó; cân nhắc semi-bluff có kiểm soát khi có equity hoặc vị trí.",
    ],
    "maniac": [
        "Đối thủ chơi rất hổ báo (maniac) với tần suất bet/bluff cao; có thể bluff-catch với bài trung bình-mạnh hoặc gài bẫy.",
        "Áp lực cược từ đối thủ chứa nhiều bài bluff và bán bluff; không nên bỏ bài quá dễ khi equity vượt Pot Odds.",
        "Đối thủ thường xuyên đẩy kích cỡ cược lớn; kiên nhẫn chọn lọc tay bài đủ mạnh để call hoặc phản công.",
    ],
    "tag": [
        "Đối thủ chơi cân bằng theo lối Tight-Aggressive (TAG), range kết hợp cả value lẫn draw hợp lý.",
        "Lịch sử hành động phản ánh đối thủ có kỷ luật về vị trí và kích cỡ cược; cần bám sát Pot Odds và equity.",
        "Đối thủ cân bằng giữa value bet và semi-bluff; ưu tiên quyết định chuẩn toán học và kiểm soát pot.",
    ],
}


def _describe_draws(draws: dict[str, Any]) -> str:
    parts: list[str] = []
    if draws["flush_draw"]:
        parts.append(f"Flush Draw ({draws['flush_outs']} outs mua thùng)")
    if draws["straight_draw"] == "OESD":
        parts.append(f"Open-Ended Straight Draw ({draws['straight_outs']} outs mua sảnh 2 đầu)")
    elif draws["straight_draw"] == "GUTSHOT":
        parts.append(f"Gutshot Straight Draw ({draws['straight_outs']} outs mua sảnh lọt khe)")
    if not parts:
        return "không có draw mua thùng/sảnh đáng kể (0 outs draw)"
    return "kết hợp " + " và ".join(parts) + f", tổng {draws['total_draw_outs']} outs mua bài"


def _choose_action_rationale(
    obs: Observation,
    action: DiscreteAction,
    equity: float,
    pot_odds: float,
    rng: random.Random,
) -> str:
    pos = obs.position
    if action == DiscreteAction.FOLD:
        opts = [
            f"Equity (~{equity * 100:.1f}%) thấp hơn ngưỡng an toàn so với Pot Odds ({pot_odds * 100:.1f}%), chọn FOLD để bảo toàn stack.",
            f"Bất lợi về xác suất thắng trước áp lực cược {obs.to_call_bb:g}BB, quyết định FOLD kỷ luật để tránh mất thêm EV.",
            f"Bài hiện tại không đủ mạnh để tiếp tục vào pot {obs.pot_bb:g}BB, chọn FOLD.",
        ]
    elif action == DiscreteAction.CHECK_CALL:
        if obs.to_call_bb == 0.0:
            opts = [
                f"Không phải đối mặt cược (To Call 0BB), chọn CHECK_CALL để kiểm soát pot và xem lá tiếp theo miễn phí.",
                f"Ở vị trí {pos} với mức equity ~{equity * 100:.1f}%, chọn CHECK_CALL nhằm giữ pot gọn và bảo vệ range check.",
            ]
        else:
            opts = [
                f"Equity (~{equity * 100:.1f}%) đủ cạnh tranh so với Pot Odds ({pot_odds * 100:.1f}%), chọn CHECK_CALL để hiện thực hóa equity.",
                f"Có mức odds hợp lý khi cần call {obs.to_call_bb:g}BB vào pot {obs.pot_bb:g}BB, chọn CHECK_CALL.",
            ]
    elif action == DiscreteAction.ALL_IN:
        opts = [
            f"Với equity vượt trội (~{equity * 100:.1f}%) và SPR {obs.spr:g}, chọn ALL_IN để tối đa hóa giá trị kỳ vọng (+EV).",
            f"Tình huống thuận lợi để gây áp lực tối đa lên stack đối thủ, chọn ALL_IN.",
        ]
    else:
        size_label = action.value
        if equity >= 0.58:
            opts = [
                f"Sở hữu lợi thế equity rõ rệt (~{equity * 100:.1f}%), chọn {size_label} để value bet và xây dựng pot.",
                f"Bài đủ mạnh tại vị trí {pos}, chọn {size_label} nhằm trích xuất giá trị từ các tay bài yếu hơn của đối thủ.",
            ]
        else:
            opts = [
                f"Tận dụng vị trí {pos} và khả năng gây áp lực (fold equity), chọn {size_label} như một nước semi-bluff có tính toán.",
                f"Kích cỡ {size_label} tạo áp lực tốt lên range yếu của đối thủ, chọn {size_label}.",
            ]
    return rng.choice(opts)


def synthesize_verified_cot(
    obs: Observation,
    action: DiscreteAction,
    opponent_type: str,
    rng: random.Random,
    equity_samples: int = 120,
) -> tuple[str, list[str]] | None:
    """Build a diverse, 100%-verifier-grounded 5-step Poker CoT for (obs, action)."""
    hole_str = " ".join(obs.hero_cards)
    board_str = " ".join(obs.board) if obs.board else "None"
    hand_cat = evaluate_hand_category(obs.hero_cards, obs.board)
    hand_vi = _HAND_VI_NAMES.get(hand_cat, hand_cat)
    draws = count_draws_and_outs(obs.hero_cards, obs.board)
    pot_odds = compute_pot_odds(obs.pot_bb, obs.to_call_bb)
    equity = estimate_equity(obs.hero_cards, obs.board, n_samples=equity_samples, seed=obs.seed)

    # Step 1 (diverse phrasings, exact ground-truth numbers & cards)
    s1_templates = [
        (
            f"1. Tình huống: Street {obs.street}, vị trí {obs.position}, bài tẩy {hole_str}, "
            f"mặt bàn {board_str}. Current Pot {obs.pot_bb:g}BB, To Call {obs.to_call_bb:g}BB, "
            f"Hero stack {obs.hero_stack_bb:g}BB (SPR {obs.spr:g})."
        ),
        (
            f"1. Tình huống: Đang ở vòng {obs.street} tại vị trí {obs.position} với stack {obs.hero_stack_bb:g}BB, "
            f"cầm {hole_str} trên mặt bàn {board_str}; Pot hiện tại {obs.pot_bb:g}BB, cần call {obs.to_call_bb:g}BB."
        ),
        (
            f"1. Tình huống: Vị trí {obs.position} ({obs.street}), bài tẩy {hole_str}, board {board_str}, "
            f"Pot {obs.pot_bb:g}BB, đối mặt mức call {obs.to_call_bb:g}BB, hiệu dụng SPR {obs.spr:g}."
        ),
    ]
    step1 = rng.choice(s1_templates)

    # Step 2 (exact made hand + draws/outs)
    if obs.street == "PREFLOP":
        strength = preflop_hand_strength(obs.hero_cards)
        tier = "rất mạnh" if strength >= 0.65 else ("khá tốt" if strength >= 0.40 else ("trung bình" if strength >= 0.25 else "yếu"))
        suited_note = (
            "đồng chất"
            if len(obs.hero_cards) == 2 and obs.hero_cards[0][1] == obs.hero_cards[1][1]
            else "khác chất"
        )
        s2_templates = [
            f"2. Sức mạnh bài: Tay bài Pre-flop là {hand_vi} ({suited_note}), sức mạnh khởi đầu ở mức {tier}.",
            f"2. Sức mạnh bài: Hiện giữ {hand_vi} trước flop ({suited_note}), nhóm bài {tier} cho thế trận Heads-Up.",
        ]
    else:
        draw_desc = _describe_draws(draws)
        s2_templates = [
            f"2. Sức mạnh bài: Hiện có {hand_vi}, {draw_desc}.",
            f"2. Sức mạnh bài: Trên mặt bàn {board_str}, tay bài đạt {hand_vi} và {draw_desc}.",
        ]
    step2 = rng.choice(s2_templates)

    # Step 3 (exact Pot Odds formula + Equity comparison)
    cmp_op = ">" if equity >= pot_odds else "<"
    if obs.to_call_bb == 0.0:
        s3_templates = [
            f"3. Toán học: To Call = 0BB nên Pot Odds = 0.0%. Equity ước tính ~{equity * 100:.1f}% > 0.0%, có thể hành động không mất phí call.",
            f"3. Toán học: Pot Odds = 0.0% (được check miễn phí). Equity ước tính khoảng {equity * 100:.1f}% > 0.0%.",
        ]
    else:
        s3_templates = [
            (
                f"3. Toán học: Pot Odds = {obs.to_call_bb:g} / ({obs.pot_bb:g} + {obs.to_call_bb:g}) "
                f"= {pot_odds * 100:.1f}%. Equity ước tính ~{equity * 100:.1f}% {cmp_op} {pot_odds * 100:.1f}%."
            ),
            (
                f"3. Toán học: Cần call {obs.to_call_bb:g}BB vào tổng pot sau call {obs.pot_bb + obs.to_call_bb:g}BB "
                f"→ Pot Odds = {pot_odds * 100:.1f}%. So sánh Equity ~{equity * 100:.1f}% {cmp_op} {pot_odds * 100:.1f}%."
            ),
        ]
    step3 = rng.choice(s3_templates)

    # Step 4 (opponent & context read)
    opp_reads = _OPP_READ_TEMPLATES.get(opponent_type, _OPP_READ_TEMPLATES["tag"])
    step4 = f"4. Đọc vị: {rng.choice(opp_reads)}"

    # Step 5 (grounded decision)
    step5 = f"5. Quyết định: {_choose_action_rationale(obs, action, equity, pot_odds, rng)}"

    steps = [step1, step2, step3, step4, step5]
    raw_response = "\n".join(f"<step>{s}</step>" for s in steps) + f"\n<action>{action.value}</action>"

    # Strict self-verification gate (100% pass required)
    parsed = parse_poker_cot(raw_response, legal_actions=obs.legal_actions, max_tokens=550)
    if not parsed.is_valid or parsed.action != action:
        return None
    ver = verify_cot_steps(obs, parsed.steps, parsed.action, equity_samples=60)
    if not (
        ver.step1_card_exact
        and ver.step2_hand_eval_ok
        and ver.step3_pot_odds_ok
        and ver.logic_action_consistent
    ):
        return None

    return raw_response, steps


def synthesize_gym_dataset(
    env_name: str = "hu_nlhe",
    target_examples: int = 12000,
    base_seed: int = 700000,
    opponents: list[str] | None = None,
    equity_samples: int = 80,
) -> list[SFTExample]:
    """Generate verified 5-step SFT examples by playing a strong TAG teacher in the Gym."""
    opp_list = opponents or ["calling_station", "nit", "maniac", "tag"]
    rng = make_rng(base_seed)
    teacher = create_bot("tag", seed=base_seed + 1, equity_samples=equity_samples)
    opp_bots = {
        name: create_bot(name, seed=base_seed + 100 + idx, equity_samples=40)
        for idx, name in enumerate(opp_list)
    }

    examples: list[SFTExample] = []
    hand_counter = 0

    while len(examples) < target_examples:
        opp_name = opp_list[hand_counter % len(opp_list)]
        opp_bot = opp_bots[opp_name]
        hand_seed = base_seed + hand_counter
        hero_seat = hand_counter % 2
        hand_id = f"sft_{env_name}_{opp_name}_{hand_counter:06d}"

        env = LeducPokerEnv() if env_name == "leduc" else HeadsUpNLHEEnv(starting_stack_bb=100.0)
        obs = env.reset(seed=hand_seed, hand_id=hand_id)
        decision_idx = 0

        while not env.is_terminal and len(examples) < target_examples:
            assert obs is not None
            seat = obs.seat
            if seat == hero_seat:
                act = teacher.act(obs).action
                cot_res = synthesize_verified_cot(
                    obs=obs,
                    action=act,
                    opponent_type=opp_name,
                    rng=rng,
                    equity_samples=equity_samples,
                )
                if cot_res is not None:
                    raw_response, steps = cot_res
                    user_prompt = format_observation(obs)
                    total_tokens = estimate_tokens(SYSTEM_PROMPT + user_prompt + raw_response)
                    if total_tokens <= 850:
                        examples.append(
                            SFTExample(
                                example_id=f"{hand_id}_d{decision_idx}",
                                hand_id=hand_id,
                                env_name=env_name,
                                task_type="standard_cot",
                                seed=hand_seed,
                                street=obs.street,
                                action=act.value,
                                opponent_type=opp_name,
                                messages=[
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": user_prompt},
                                    {"role": "assistant", "content": raw_response},
                                ],
                                steps=steps,
                                meta={
                                    "hero_cards": list(obs.hero_cards),
                                    "board": list(obs.board),
                                    "position": obs.position,
                                    "total_tokens_est": total_tokens,
                                },
                            )
                        )
                decision_idx += 1
            else:
                act = opp_bot.act(obs).action

            obs, _, _, _ = env.step(act)

        hand_counter += 1

    return examples
