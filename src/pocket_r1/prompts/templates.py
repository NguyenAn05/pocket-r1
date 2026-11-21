"""System prompt and few-shot templates for 5-step Structured Poker CoT."""

from __future__ import annotations

from pocket_r1.env.base import Observation
from pocket_r1.prompts.state_formatter import format_observation

SYSTEM_PROMPT = """Bạn là chuyên gia Poker chiến thuật (Heads-Up No-Limit Hold'em / Leduc Poker).
Với mỗi trạng thái bàn được cung cấp, hãy suy luận ngắn gọn, chính xác theo ĐÚNG 5 thẻ <step> và kết thúc bằng ĐÚNG 1 thẻ <action> nằm trong danh sách Legal Actions. Không viết thêm chữ ngoài các thẻ.

Quy ước toán học bắt buộc:
- Pot Odds = To Call / (Current Pot + To Call). Nếu To Call = 0 thì Pot Odds = 0%.
- Giữ toàn bộ câu trả lời ngắn gọn (dưới 450 tokens).

Định dạng bắt buộc:
<step>1. Tình huống: Nêu đúng vị trí, stack, bài tẩy, mặt bàn, pot và lượng cần call.</step>
<step>2. Sức mạnh bài: Đánh giá tay bài hiện tại và số outs mua bài (nếu có).</step>
<step>3. Toán học: Tính chính xác Pot Odds, ước lượng Equity và so sánh hai chỉ số.</step>
<step>4. Đọc vị: Phân tích range đối thủ dựa trên lịch sử cược và chỉ số (nếu có).</step>
<step>5. Quyết định: Tổng hợp lý do chọn hành động tối ưu (+EV).</step>
<action>ONE_OF_LEGAL_ACTIONS</action>"""

ONE_SHOT_USER = """Game: HU_NLHE | Street: FLOP | Position: BTN
Hole Cards: As Ks | Board: Qs 7s 2h
Current Pot (including villain's bet): 18 BB | To Call: 6 BB
Hero Stack: 89 BB | Villain Stack: 83 BB | Effective SPR: 4.61
Action History: PREFLOP: BTN BET_75(2.5BB) -> BB CHECK_CALL(2BB) | FLOP: BB BET_75(6BB)
Legal Actions: [FOLD, CHECK_CALL, BET_75, BET_150, ALL_IN]"""

ONE_SHOT_ASSISTANT = """<step>1. Tình huống: Vị trí BTN, Hero stack 89BB, bài tẩy As Ks, mặt bàn Qs 7s 2h, Current Pot 18BB, To Call 6BB.</step>
<step>2. Sức mạnh bài: Hiện có Ace-high kết hợp Nut Flush Draw (9 outs mua thùng bích) và 2 overcards.</step>
<step>3. Toán học: Pot Odds = 6 / (18 + 6) = 25.0%. Equity ước tính ~45% > 25.0%, đủ lợi thế toán học để tiếp tục.</step>
<step>4. Đọc vị: Đối thủ donk-bet lớn trên mặt bàn có 2 lá bích, range có thể gồm cặp Q, cặp 7 hoặc draw yếu hơn.</step>
<step>5. Quyết định: Có ưu thế vị trí và equity cao vượt trội Pot Odds, chọn Call để hiện thực hóa equity ở lá Turn.</step>
<action>CHECK_CALL</action>"""


def build_chat_messages(obs: Observation, few_shot: bool = False) -> list[dict[str, str]]:
    """Build OpenAI-style chat messages for an Observation."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        messages.append({"role": "user", "content": ONE_SHOT_USER})
        messages.append({"role": "assistant", "content": ONE_SHOT_ASSISTANT})
    messages.append({"role": "user", "content": format_observation(obs)})
    return messages
