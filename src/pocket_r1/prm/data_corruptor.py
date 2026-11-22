"""RTD-style Synthetic Corruption generator for 6-step DeBERTa-v3 PRM training."""

from __future__ import annotations

import random
import re
from typing import Any

from pydantic import BaseModel, Field

from pocket_r1.data.cot_synthesizer import SFTExample
from pocket_r1.env.action_abstraction import DiscreteAction
from pocket_r1.env.nlhe_env import STANDARD_DECK
from pocket_r1.utils.seeding import make_rng
from pocket_r1.verifiers.step_verifier import extract_cards

CORRUPTION_TYPES: tuple[str, ...] = (
    "step1_card_hallucination",
    "step2_hand_or_outs_hallucination",
    "step3_pot_odds_or_cmp_error",
    "step4_context_contradiction",
    "action_logic_contradiction",
    "cascading_hand_to_action_error",
)

_PCT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
_OUTS_RE = re.compile(r"(\d{1,2})\s*outs", re.IGNORECASE)


class PRMSample(BaseModel):
    """Canonical 6-checkpoint step-level PRM sample (Steps 1-5 + Action)."""

    sample_id: str
    hand_id: str
    source: str  # clean | synthetic:<type> | onpolicy:<exp>
    prompt_text: str
    steps: list[str]
    action: str
    labels: list[float] = Field(min_length=6, max_length=6)
    meta: dict[str, Any] = Field(default_factory=dict)


def sft_to_clean_prm_sample(ex: SFTExample) -> PRMSample | None:
    if len(ex.steps) != 5:
        return None
    return PRMSample(
        sample_id=f"{ex.example_id}_clean",
        hand_id=ex.hand_id,
        source="clean",
        prompt_text=ex.messages[1]["content"],
        steps=list(ex.steps),
        action=ex.action,
        labels=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        meta={"street": ex.street, "opponent_type": ex.opponent_type},
    )


def _corrupt_step1_card(steps: list[str], meta: dict[str, Any], rng: random.Random) -> tuple[list[str], list[float]] | None:
    s1 = steps[0]
    cards = extract_cards(s1)
    if not cards:
        return None
    target = rng.choice(cards)
    known = set(meta.get("hero_cards", [])) | set(meta.get("board", []))
    pool = [c for c in STANDARD_DECK if c not in known and c != target]
    if not pool:
        return None
    # Prefer same rank different suit (subtle hallucination!)
    same_rank = [c for c in pool if c[0] == target[0]]
    fake = rng.choice(same_rank) if same_rank else rng.choice(pool)
    new_s1 = s1.replace(target, fake, 1)
    if new_s1 == s1:
        return None
    out = list(steps)
    out[0] = new_s1
    return out, [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def _corrupt_step2_hand_or_outs(steps: list[str], rng: random.Random) -> tuple[list[str], list[float]] | None:
    s2 = steps[1]
    out = list(steps)
    # Variant A: corrupt outs number if present
    m_outs = _OUTS_RE.search(s2)
    if m_outs and rng.random() < 0.45:
        orig = int(m_outs.group(1))
        fake_outs = orig + rng.choice([7, 9, 12]) if orig < 10 else max(1, orig - 7)
        out[1] = s2[: m_outs.start(1)] + str(fake_outs) + s2[m_outs.end(1) :]
        return out, [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    # Variant B: hallucinate a stronger made hand
    replacements = [
        ("High Card (mậu thầu)", rng.choice(["Flush (thùng)", "Two Pair (hai đôi)", "Full House (cù lũ)"])),
        ("Pair (một đôi)", rng.choice(["Full House (cù lũ)", "Flush (thùng)", "Four of a Kind (tứ quý)"])),
        ("Two Pair (hai đôi)", "Straight Flush (thùng phá sảnh)"),
        ("không có draw mua thùng/sảnh đáng kể (0 outs draw)", "kết hợp Flush Draw (9 outs mua thùng) rất mạnh"),
    ]
    rng.shuffle(replacements)
    for src, dst in replacements:
        if src in s2:
            out[1] = s2.replace(src, dst, 1)
            return out, [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    out[1] = "2. Sức mạnh bài: Hiện đã hoàn thành Flush (thùng) lớn nhất trên bàn với 15 outs bổ sung."
    return out, [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def _corrupt_step3_math(steps: list[str], rng: random.Random) -> tuple[list[str], list[float]] | None:
    s3 = steps[2]
    out = list(steps)
    # Variant A: flip inequality sign '>' <-> '<'
    if (" > " in s3 or " < " in s3) and rng.random() < 0.45:
        out[2] = s3.replace(" > ", " < ", 1) if " > " in s3 else s3.replace(" < ", " > ", 1)
        return out, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]

    # Variant B: distort first percentage (Pot Odds) by +-18..28%
    m_pct = _PCT_RE.search(s3)
    if m_pct:
        val = float(m_pct.group(1))
        delta = rng.choice([18.0, 24.0, -18.0, -22.0])
        new_val = max(3.0, min(92.0, val + delta))
        if abs(new_val - val) < 10.0:
            new_val = min(92.0, val + 22.0)
        out[2] = s3[: m_pct.start(1)] + f"{new_val:.1f}" + s3[m_pct.end(1) :]
        return out, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    return None


def _corrupt_step4_context(steps: list[str], opponent_type: str, rng: random.Random) -> tuple[list[str], list[float]]:
    out = list(steps)
    contradictions = {
        "calling_station": "4. Đọc vị: Đối thủ là người chơi siêu chặt (nit), chỉ chơi top 5% bài và rất dễ bỏ bài khi bị bluff.",
        "nit": "4. Đọc vị: Đối thủ là maniac cực kỳ hổ báo, thường xuyên bluff bừa bãi mọi vòng nên ta nên call nhẹ với mọi bài.",
        "maniac": "4. Đọc vị: Đối thủ chơi cực kỳ thụ động, hầu như không bao giờ bluff nên khi họ bet là chắc chắn cầm bài mạnh nhất.",
        "tag": "4. Đọc vị: Đối thủ không quan tâm vị trí hay Pot Odds, luôn đẩy all-in ngẫu nhiên mọi tay bài yếu.",
    }
    out[3] = contradictions.get(opponent_type, rng.choice(list(contradictions.values())))
    return out, [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]


def _corrupt_action_contradiction(
    steps: list[str], orig_action: str, rng: random.Random
) -> tuple[list[str], str, list[float]]:
    opposites: dict[str, list[str]] = {
        DiscreteAction.FOLD.value: [DiscreteAction.ALL_IN.value, DiscreteAction.BET_150.value, DiscreteAction.BET_75.value],
        DiscreteAction.CHECK_CALL.value: [DiscreteAction.FOLD.value, DiscreteAction.ALL_IN.value],
        DiscreteAction.BET_33.value: [DiscreteAction.FOLD.value],
        DiscreteAction.BET_75.value: [DiscreteAction.FOLD.value],
        DiscreteAction.BET_150.value: [DiscreteAction.FOLD.value],
        DiscreteAction.ALL_IN.value: [DiscreteAction.FOLD.value],
    }
    fake_act = rng.choice(opposites.get(orig_action, [DiscreteAction.FOLD.value]))
    return list(steps), fake_act, [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]


def _corrupt_cascading(steps: list[str], rng: random.Random) -> tuple[list[str], str, list[float]]:
    out = list(steps)
    out[1] = "2. Sức mạnh bài: Đã hoàn thành Full House (cù lũ) cực lớn, hầu như chắc chắn thắng showdown."
    out[4] = "5. Quyết định: Với bài cù lũ mạnh tuyệt đối, chọn ALL_IN để ăn trọn stack đối thủ."
    return out, DiscreteAction.ALL_IN.value, [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def corrupt_sft_example(
    ex: SFTExample,
    corruption_type: str,
    rng: random.Random,
) -> PRMSample | None:
    """Create a synthetic RTD-corrupted PRMSample with exact per-step labels."""
    if len(ex.steps) != 5:
        return None
    prompt_text = ex.messages[1]["content"]
    action = ex.action

    if corruption_type == "step1_card_hallucination":
        res = _corrupt_step1_card(ex.steps, ex.meta, rng)
        if res is None:
            return None
        new_steps, labels = res
    elif corruption_type == "step2_hand_or_outs_hallucination":
        res = _corrupt_step2_hand_or_outs(ex.steps, rng)
        if res is None:
            return None
        new_steps, labels = res
    elif corruption_type == "step3_pot_odds_or_cmp_error":
        res = _corrupt_step3_math(ex.steps, rng)
        if res is None:
            return None
        new_steps, labels = res
    elif corruption_type == "step4_context_contradiction":
        new_steps, labels = _corrupt_step4_context(ex.steps, ex.opponent_type, rng)
    elif corruption_type == "action_logic_contradiction":
        new_steps, action, labels = _corrupt_action_contradiction(ex.steps, ex.action, rng)
    elif corruption_type == "cascading_hand_to_action_error":
        new_steps, action, labels = _corrupt_cascading(ex.steps, rng)
    else:
        raise ValueError(f"Unknown corruption_type: {corruption_type}")

    return PRMSample(
        sample_id=f"{ex.example_id}_{corruption_type}",
        hand_id=ex.hand_id,
        source=f"synthetic:{corruption_type}",
        prompt_text=prompt_text,
        steps=new_steps,
        action=action,
        labels=labels,
        meta={"corruption_type": corruption_type, "street": ex.street},
    )


def generate_synthetic_prm_samples(
    sft_examples: list[SFTExample],
    n_corruptions: int = 10000,
    seed: int = 42,
) -> tuple[list[PRMSample], list[PRMSample]]:
    """Build (clean_samples, corrupted_samples) from verified SFT examples."""
    rng = make_rng(seed)
    clean_samples: list[PRMSample] = []
    for ex in sft_examples:
        s = sft_to_clean_prm_sample(ex)
        if s is not None:
            clean_samples.append(s)

    corrupted_samples: list[PRMSample] = []
    if not sft_examples or n_corruptions <= 0:
        return clean_samples, corrupted_samples

    attempts = 0
    max_attempts = n_corruptions * 3
    while len(corrupted_samples) < n_corruptions and attempts < max_attempts:
        ex = rng.choice(sft_examples)
        ctype = CORRUPTION_TYPES[len(corrupted_samples) % len(CORRUPTION_TYPES)]
        sample = corrupt_sft_example(ex, ctype, rng)
        if sample is not None:
            # Make sample_id unique if same example chosen twice
            sample.sample_id = f"{sample.sample_id}_{len(corrupted_samples)}"
            corrupted_samples.append(sample)
        attempts += 1

    return clean_samples, corrupted_samples
