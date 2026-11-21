"""Masked Step Infilling (FIM) generator targeting Step 2 (Hand/Outs) and Step 3 (Pot Odds)."""

from __future__ import annotations

from pocket_r1.data.cot_synthesizer import SFTExample
from pocket_r1.utils.seeding import make_rng

FIM_SYSTEM_PROMPT = """Bạn là chuyên gia Poker chiến thuật.
Nhiệm vụ của bạn là điền chính xác duy nhất bước suy luận bị ẩn (<MASK_STEP>) sao cho khớp tuyệt đối với trạng thái bàn, các bước trước/sau và hành động cuối.
Chỉ trả về đúng 1 thẻ <step>...</step> cho bước bị ẩn, không viết thêm văn bản ngoài thẻ."""


def create_fim_examples(
    examples: list[SFTExample],
    ratio: float = 0.25,
    mask_steps: tuple[int, ...] = (2, 3),
    seed: int = 42,
) -> list[SFTExample]:
    """Build Masked Step Infilling examples by hiding Step 2 or Step 3 from verified CoT traces."""
    rng = make_rng(seed)
    eligible = [ex for ex in examples if ex.task_type == "standard_cot" and len(ex.steps) == 5]
    if not eligible or ratio <= 0.0:
        return []

    n_target = max(1, int(round(len(eligible) * ratio)))
    sampled = rng.sample(eligible, min(n_target, len(eligible)))

    fim_list: list[SFTExample] = []
    for ex in sampled:
        step_num = rng.choice(mask_steps)  # 2 or 3 (1-indexed)
        idx = step_num - 1
        target_step = f"<step>{ex.steps[idx]}</step>"

        masked_lines: list[str] = []
        for i, s in enumerate(ex.steps):
            if i == idx:
                masked_lines.append("<MASK_STEP>")
            else:
                masked_lines.append(f"<step>{s}</step>")
        masked_lines.append(f"<action>{ex.action}</action>")

        state_prompt = ex.messages[1]["content"]
        fim_user = (
            f"{state_prompt}\n\n"
            "[Nhiệm vụ: Điền bước suy luận bị thiếu (<MASK_STEP>) sao cho nhất quán với trạng thái bàn, "
            "các bước trước/sau và hành động cuối.]\n"
            + "\n".join(masked_lines)
        )

        fim_list.append(
            SFTExample(
                example_id=f"{ex.example_id}_fim_s{step_num}",
                hand_id=ex.hand_id,
                env_name=ex.env_name,
                task_type=f"fim_step{step_num}",
                seed=ex.seed,
                street=ex.street,
                action=ex.action,
                opponent_type=ex.opponent_type,
                messages=[
                    {"role": "system", "content": FIM_SYSTEM_PROMPT},
                    {"role": "user", "content": fim_user},
                    {"role": "assistant", "content": target_step},
                ],
                steps=ex.steps,
                meta={**ex.meta, "masked_step": step_num},
            )
        )

    return fim_list
