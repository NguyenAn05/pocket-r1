# Kế hoạch Chi tiết — Phase 1: Xây Phòng Gym, Đấu trường & Baseline (Exp 0)

> **Thời lượng dự kiến:** Tuần 1–2  
> **Tài liệu gốc:** [PROJECT_IDEA.md](./PROJECT_IDEA.md)  
> **Giai đoạn tiếp theo:** [Phase 2 — Cold-start SFT & Baseline](./PHASE_2_COLD_START_SFT.md)

---

## 1. Mục tiêu & Phạm vi (Goals & Scope)

Phase 1 đặt toàn bộ nền móng kỹ thuật cho `pocket-r1`: xây dựng một môi trường Poker chuẩn hóa, nhanh, tái lập được (*deterministic*), kèm hệ thống đối thủ, bộ tính kỳ vọng toán học (**EV**), bộ kiểm chứng lỗi suy luận tự động và bộ benchmark gốc **Exp 0 (Base SLM)**.

### Đầu ra bắt buộc của Phase 1
1. Cấu trúc repository chuẩn, môi trường Python quản lý đầy đủ dependencies (`PokerKit`, `eval7`/`treys`, `vLLM`, `pytest`, `wandb`).
2. Hai môi trường tương thích chung một interface:
   * **Prototype Env:** `Leduc Poker` (kiểm thông pipeline nhanh, dùng lại ở Phase 3–4 để debug RL).
   * **Main Env:** `Heads-Up (1v1) No-Limit Texas Hold'em (HU-NLHE)` với **Action Abstraction 6 hành động**.
3. Chuẩn hóa **Prompt Trạng thái bàn** và **Structured CoT Parser** cho khung 5 thẻ `<step>` + `<action>`.
4. **4 Bot tính cách** (Tầng 2): `Calling Station`, `Nit/Tight`, `Maniac/Aggressive`, `TAG` — có seed tái lập.
5. **EV Verifier** (Monte Carlo runout tối ưu tốc độ + giảm phương sai) và **Deterministic Code Verifiers** (bắt lỗi Bước 1–3).
6. **Held-out Evaluation Suite** (cố định bộ bài chia theo cơ chế *Duplicate Deals*) và kết quả benchmark **Exp 0 (Base SLM)**.

---

## 2. Cấu trúc Thư mục Khởi tạo (Repo Architecture)

Thiết kế cấu trúc thống nhất để mở rộng mượt sang Phase 2–4:

```text
pocket-r1/
├── configs/
│   ├── env/
│   │   ├── leduc.yaml
│   │   └── hu_nlhe.yaml
│   ├── bots/
│   │   └── personalities.yaml
│   └── eval/
│       └── exp0_baseline.yaml
├── src/pocket_r1/
│   ├── env/
│   │   ├── base.py                 # Interface chung cho Leduc & HU-NLHE
│   │   ├── leduc_env.py            # Prototype Leduc Poker wrapper
│   │   ├── nlhe_env.py             # Heads-Up NLHE wrapper trên PokerKit
│   │   └── action_abstraction.py   # Mapping 6 hành động rời rạc & legal masking
│   ├── prompts/
│   │   ├── state_formatter.py      # Chuyển trạng thái bàn -> text ngắn gọn
│   │   ├── templates.py            # System & user prompt cho khung CoT 5 bước
│   │   └── parser.py               # Parse <step>, <action> & phân loại lỗi format
│   ├── agents/
│   │   ├── base.py                 # Interface Agent.act(obs) -> ActionResult
│   │   ├── rule_bots.py            # Calling Station, Nit, Maniac, TAG
│   │   └── slm_agent.py            # Wrapper gọi SLM qua vLLM / Transformers
│   ├── verifiers/
│   │   ├── card_math.py            # Eval tay bài, đếm outs, tính Pot Odds, Equity
│   │   ├── step_verifier.py        # Kiểm chứng tự động Bước 1, 2, 3 bằng Python
│   │   └── ev_calculator.py        # Bộ tính Action EV bằng Monte Carlo Runout
│   ├── eval/
│   │   ├── duplicate_arena.py      # Đấu trường Duplicate Deals đo BB/100
│   │   └── metrics.py              # BB/100, Hallucination Rate, Pot Odds Acc, Consistency
│   └── utils/
│       ├── logging.py              # Trajectory JSONL logger + W&B/MLflow hook
│       └── seeding.py
├── scripts/
│   ├── generate_eval_suite.py      # Sinh tập tay bài Held-out cố định
│   ├── sanity_check_bots.py        # Cho 4 bot đấu vòng tròn kiểm chứng độ hợp lý
│   └── run_exp0_baseline.py        # Chạy benchmark Exp 0
├── tests/
│   ├── test_nlhe_env.py
│   ├── test_parser.py
│   ├── test_card_math.py
│   └── test_ev_calculator.py
└── data/
    └── eval_suites/                # Tập tay bài test cố định (không được dùng để train)
```

---

## 3. Thiết kế Kỹ thuật Chi tiết

### 3.1. Môi trường Poker & Action Abstraction (`src/pocket_r1/env/`)

* **Thông số chuẩn HU-NLHE (`configs/env/hu_nlhe.yaml`):**
  * Effective stack khởi đầu mỗi ván: `100 BB` (`SB = 0.5 BB`, `BB = 1.0 BB`), reset stack sau mỗi ván (Cash-game style) để đo `BB/100` ổn định và công bằng giữa các mô hình.
  * Vị trí: `BTN/SB` (hành động trước ở Pre-flop, hành động sau ở Post-flop) và `BB`.
* **Rời rạc hóa 6 hành động (`action_abstraction.py`):**
  1. `FOLD`
  2. `CHECK_CALL` (tự map sang `Check` nếu không đối mặt cược, ngược lại `Call`)
  3. `BET_33` (Bet hoặc Raise thêm sao cho tổng mức cược tương đương `0.33x Pot` sau khi gọi cược)
  4. `BET_75` (`0.75x Pot`)
  5. `BET_150` (`1.5x Pot`)
  6. `ALL_IN` (đẩy toàn bộ stack còn lại)
* **Xử lý biên & Legal Action Masking:**
  * Nếu mức bet tính ra nhỏ hơn `min_raise` của `PokerKit` → tự động kẹp về `min_raise` (hoặc loại khỏi danh sách legal action nếu trùng với action kế tiếp).
  * Nếu mức bet tính ra $\ge 85\%$ stack còn lại → gộp vào `ALL_IN` để tránh action dư thừa.
  * Khi `to_call == 0`, loại `FOLD` khỏi `legal_actions` (tránh SLM tự bỏ bài miễn phí).
  * Luôn trả về `legal_actions` tường minh trong observation để prompt liệt kê rõ cho SLM.

### 3.2. Chuẩn hóa Prompt Trạng thái & Parser (`src/pocket_r1/prompts/`)

* **Ràng buộc độ dài:** Toàn bộ `State Prompt + 5-step CoT + Action` phải nằm gọn dưới **`900–1024 tokens`** để tương thích tối đa với `DeBERTa-v3` ở Phase 3–4.
  * Ngân sách gợi ý: Prompt trạng thái $\le 250$ tokens; 5 bước CoT + Action $\le 500$ tokens.
* **Cấu trúc State Observation chuẩn hóa:**
  * Thông tin tĩnh & bàn: `Street (PREFLOP/FLOP/TURN/RIVER)`, `Position (BTN/BB)`, `Hero Stack`, `Villain Stack`, `Pot`, `To Call`, `Effective SPR`.
  * Bài: `Hole Cards` (ví dụ `As Ks`), `Board` (ví dụ `Qs 7s 2h` hoặc `None`).
  * Lịch sử cược nén gọn theo street: `PREFLOP: BTN BET_75 (2.5BB) -> BB CHECK_CALL | FLOP: BB CHECK -> BTN BET_33 (2.0BB)`.
  * Danh sách `Legal Actions`: `[FOLD, CHECK_CALL, BET_33, BET_75, BET_150, ALL_IN]`.
  * *(Tùy chọn cho đối thủ Tầng 2)*: Có thể cung cấp hoặc ẩn nhãn đối thủ; khuyến nghị **ẩn nhãn đối thủ** và cung cấp tóm tắt thống kê lịch sử gần đây (`Villain Stats: VPIP 62%, PFR 18%, AF 1.1` sau tối thiểu 20 ván) để SLM học cách *đọc vị* thực sự thay vì học vẹt tên bot.
* **Parser & Phân loại lỗi (`parser.py`):**
  * Bắt chính xác 5 thẻ `<step>...</step>` (đánh số 1–5) và đúng 1 thẻ `<action>...</action>`.
  * Trả về `ParseResult(is_valid, steps, action, error_codes, token_count)` với mã lỗi rõ ràng: `MISSING_STEPS`, `WRONG_STEP_COUNT`, `MISSING_ACTION`, `UNKNOWN_ACTION`, `ILLEGAL_ACTION`, `EXTRA_TEXT_OUTSIDE_TAGS`, `LENGTH_OVERFLOW`.
  * Khi SLM sinh hành động bất hợp lệ trong lúc đánh thực tế: log cờ `illegal_action=True`, áp fallback an toàn (`CHECK_CALL` nếu không mất tiền, ngược lại `FOLD`) để ván đấu không bị crash.

### 3.3. Dàn 4 Bot Tính cách (`src/pocket_r1/agents/rule_bots.py`)

Mỗi bot sử dụng **Hand Strength / Equity Estimator** (`eval7`) kết hợp tham số tính cách trong `configs/bots/personalities.yaml` và bộ sinh số ngẫu nhiên theo seed:

| Bot | Đặc trưng chiến thuật | Thông số hành vi mục tiêu | Bài học ép SLM phải học |
| :--- | :--- | :--- | :--- |
| **Calling Station** | Rất ít Fold, ít Raise, hay Call tới River với cặp yếu/draw yếu | VPIP ~65–75%, PFR ~10%, Fold-to-Bet ~15% | **Value Bet mỏng & dày**, gần như không Bluff |
| **Nit / Tight** | Chỉ chơi top 15% bài Pre-flop, Post-flop bỏ bài ngay khi trượt mặt bàn | VPIP ~15–20%, PFR ~12%, Fold-to-Bet ~65% | **Cướp Pot / Semi-bluff có tính toán**, tôn trọng khi Nit tố lại |
| **Maniac / Aggressive** | Tố/Bluff liên tục ở mọi street với kích cỡ lớn (`BET_75`, `BET_150`) | VPIP ~70%, PFR ~55%, AF > 3.5, Bluff freq ~45% | **Slow-play / Gài bẫy & Bluff-catch** với bài trung bình-mạnh |
| **TAG (Tight-Aggressive)** | Chọn lọc bài hợp lý theo vị trí, c-bet có chọn lọc, cân bằng value & draw | VPIP ~28–35%, PFR ~24%, AF ~2.2 | **Chơi căn bản vững chắc**, kiểm soát Pot Odds & vị trí |

* **Kiểm chứng chéo (Sanity Check):** Cho 4 bot đấu vòng tròn 5.000 ván (`scripts/sanity_check_bots.py`) để xác nhận `TAG` thắng dương bền vững trước `Calling Station`, `Nit`, `Maniac`, và không bot nào gặp lỗi crash hay vòng lặp vô hạn.

### 3.4. Bộ Kiểm chứng Toán học & Tính EV (`src/pocket_r1/verifiers/`)

#### A. Deterministic Code Verifier (`card_math.py`, `step_verifier.py`)
Tự động đối chiếu nội dung văn bản trong 3 bước đầu của CoT với ground truth từ `PokerKit` + `eval7`:
* **Bước 1 (Tình huống):** Trích xuất lá bài (`As`, `K♠`...), vị trí, Pot, Stack nhắc tới trong Step 1 → đối chiếu khớp 100% với trạng thái thật. Phát hiện **Card Hallucination**.
* **Bước 2 (Sức mạnh bài & Outs):** So khớp loại tay bài hiện tại (High card, Pair, Two Pair, Set, Straight, Flush...) và kiểm tra số *outs* (ví dụ: Nut Flush Draw = 9 outs, Open-Ended Straight Draw = 8 outs, sai số cho phép $\pm 1$ out cho các trường hợp trùng lá).
* **Bước 3 (Toán học):**
  * Công thức Pot Odds chuẩn: $\text{Pot Odds} = \frac{\text{ToCall}}{\text{Pot} + \text{ToCall}}$ (lưu ý thống nhất định nghĩa giữa prompt và verifier: `Pot` đã bao gồm cược hiện tại của đối thủ hay chưa — **cần ghi cứng định nghĩa này trong `state_formatter.py` để tránh nhầm lẫn toàn dự án!**).
  * Tính `Equity` bằng Monte Carlo nhanh (200–500 mẫu ngẫu nhiên bài đối thủ & lá còn lại) để kiểm tra so sánh `Equity > Pot Odds` hay `<` trong Step 3 có hợp lý không.

#### B. EV Calculator (`ev_calculator.py`)
Để giảm phương sai (High Variance) khi đánh giá nước đi và chuẩn bị cho `R_EV` ở Phase 4:
* **Với `Leduc Poker`:** Có thể tính **Exact EV** bằng duyệt cây trò chơi (rất nhỏ, chạy dưới 1ms).
* **Với `HU-NLHE`:** Thiết kế 2 chế độ EV linh hoạt:
  1. **Rollout EV (ưu tiên chính cho Gym đối đầu Bot):** Tại một decision point $(s, a)$, đóng băng trạng thái bàn, thực hiện hành động $a$, sau đó cho **Policy Rollout** (chính đối thủ hiện tại đấu với một fast equity-heuristic policy hoặc bản thân SLM ở bước sau) chạy $K$ kịch bản ngẫu nhiên (ví dụ $K = 32$ hoặc $64$) chia các lá bài ẩn khác nhau, lấy trung bình lợi nhuận ròng theo `BB`.
     * Áp dụng **Common Random Numbers (CRN):** Dùng chung tập $K$ bộ bài ẩn khi so sánh các hành động khác nhau $a_1, a_2, \dots$ tại cùng trạng thái $s$ để giảm mạnh phương sai ước lượng.
  2. **All-in / Showdown Equity EV (bổ trợ tốc độ cao):** Khi hành động dẫn tới All-in hoặc trên River, tính trực tiếp $\text{EV} = \text{Equity} \times (\text{Pot} + \text{Call}) - \text{Call}$ (với `FOLD` thì $\text{EV} = 0$ so với mốc quyết định hiện tại).
* **Mục tiêu tốc độ:** Vectorize hoặc dùng `eval7` (viết bằng Cython/C) + đa luồng Python để đạt **< 20–50 ms / decision point** cho $K=32$ rollouts.

### 3.5. Schema Lưu trữ Trajectory (JSONL)

Mọi ván đấu sinh ra trong Gym đều lưu chuẩn hóa theo schema sau (dùng xuyên suốt Phase 1–4):

```json
{
  "hand_id": "nlhe_eval_000142_seat0",
  "env_name": "hu_nlhe",
  "seed": 420142,
  "opponent_type": "calling_station",
  "street": "FLOP",
  "decision_idx": 2,
  "state_json": {
    "position": "BTN",
    "hero_cards": ["As", "Ks"],
    "board": ["Qs", "7s", "2h"],
    "pot_bb": 12.0,
    "to_call_bb": 6.0,
    "hero_stack_bb": 89.0,
    "villain_stack_bb": 83.0,
    "legal_actions": ["FOLD", "CHECK_CALL", "BET_75", "ALL_IN"]
  },
  "prompt_text": "...",
  "raw_response": "<step>1. ...</step>...<action>CHECK_CALL</action>",
  "parsed": {
    "is_valid": true,
    "steps": ["1. ...", "2. ...", "3. ...", "4. ...", "5. ..."],
    "action": "CHECK_CALL",
    "error_codes": [],
    "response_tokens": 214
  },
  "auto_checks": {
    "step1_card_exact": true,
    "step2_hand_eval_ok": true,
    "step3_pot_odds_ok": true,
    "step3_expected_pot_odds": 0.25,
    "step3_equity_est": 0.46
  },
  "ev_bb": 3.42,
  "hand_outcome_bb": -12.0
}
```

---

## 4. Kế hoạch Triển khai Từng Bước (Work Breakdown)

### Tuần 1: Môi trường, Prompt & Bot
- [ ] **Task 1.1 – Khởi tạo dự án:** Tạo `pyproject.toml`, cài đặt `pokerkit`, `eval7`, `pydantic`, `pyyaml`, `pytest`, `wandb`. Kiểm tra GPU/CUDA cục bộ.
- [ ] **Task 1.2 – Prototype `LeducEnv`:** Dựng xong `base.py` và `leduc_env.py`, viết unit test xác nhận chia bài, đặt cược, showdown, tính reward đúng 100%.
- [ ] **Task 1.3 – Main `NLHEEnv` & `ActionAbstraction`:** Dựng wrapper `PokerKit` cho HU-NLHE 100BB với 6 hành động rời rạc; viết test bao phủ các edge case (split pot, side pot / all-in lệch stack, min-raise, blinds all-in).
- [ ] **Task 1.4 – `StateFormatter` & `Parser`:** Viết formatter nén lịch sử cược + parser chặt chẽ cho 5 `<step>` và `<action>`. Kiểm tra độ dài token trên 1.000 trạng thái ngẫu nhiên.
- [ ] **Task 1.5 – 4 Bot tính cách:** Implement `CallingStationBot`, `NitBot`, `ManiacBot`, `TAGBot`. Chạy giải đấu vòng tròn 5.000 ván kiểm tra phân bố VPIP/PFR/AF và thứ hạng thắng thua.

### Tuần 2: Verifier, Đấu trường Duplicate & Benchmark Exp 0
- [ ] **Task 1.6 – `CardMath` & `StepVerifier`:** Implement bộ kiểm tra tự động Bước 1 (lá bài/vị trí/pot), Bước 2 (loại bài/outs), Bước 3 (Pot Odds & Equity). Viết bộ unit test gồm ít nhất 30 case đúng & sai tinh vi.
- [ ] **Task 1.7 – `EVCalculator`:** Implement Exact EV cho Leduc và Monte Carlo Rollout EV (dùng CRN) cho HU-NLHE. Benchmark tốc độ và độ ổn định phương sai với $K \in \{16, 32, 64\}$.
- [ ] **Task 1.8 – Held-out Evaluation Suite (`duplicate_arena.py`):**
  * Cố định tập seed cho **250 cặp Duplicate Deals (= 500 ván / đối thủ × 4 đối thủ = 2.000 ván)** lưu vào `data/eval_suites/hu_nlhe_heldout_v1.jsonl`.
  * Cơ chế Duplicate Deal: cùng một bộ bài đã xáo, ván 1 Hero ngồi BTN và Villain ngồi BB; ván 2 đổi bài cho nhau để triệt tiêu may rủi chia bài khi đo `BB/100`.
- [ ] **Task 1.9 – Chạy Benchmark Exp 0 (Base SLM):**
  * Chọn Base SLM (khuyến nghị bắt đầu với `Qwen2.5-3B-Instruct` hoặc `Qwen3-4B`, kèm thử nghiệm nhanh `1.5B` nếu GPU hạn chế).
  * Chạy Zero-shot / 1-shot trên Held-out Evaluation Suite qua `vLLM`.
  * Log toàn bộ chỉ số gốc lên W&B/MLflow và lưu báo cáo `reports/exp0_baseline.md`.

---

## 5. Tiêu chí Nghiệm thu (Definition of Done / Exit Criteria)

Chỉ chuyển sang Phase 2 khi đạt đủ các cổng kiểm duyệt sau:

| Hạng mục | Tiêu chí đạt |
| :--- | :--- |
| **Tính đúng đắn môi trường** | 100% unit tests (`pytest`) pass; chạy 10.000 ván random/bot-vs-bot không crash, bảo toàn tổng chip (zero-sum) tuyệt đối. |
| **Chất lượng 4 Bot** | Thống kê VPIP/PFR/AF khớp cấu hình mục tiêu; `TAG` đạt dương `BB/100` trước 3 bot còn lại trên 5.000 ván. |
| **Hiệu năng Verifier & EV** | `StepVerifier` đạt độ chính xác $\ge 98\%$ trên bộ test tay; `EVCalculator` đạt $< 50\text{ ms}$ / decision point với sai số chuẩn chấp nhận được. |
| **Độ dài Prompt** | 99% trạng thái bàn sau khi format nằm dưới `300 tokens`. |
| **Số liệu gốc Exp 0** | Có đầy đủ bảng số liệu `BB/100` (kèm 95% CI) theo từng bot, `Format Validity Rate`, `Illegal Action Rate`, `Card Hallucination Rate`, `Pot Odds Accuracy`, và log trajectory mẫu để phân tích lỗi sơ đẳng. |

---

## 6. Rủi ro Đặc thù Phase 1 & Cách Xử lý

1. **Nhầm lẫn định nghĩa Pot Odds giữa Prompt và Verifier:**
   * *Rủi ro:* Trong `PokerKit`, tổng `pot` có thể đã cộng hoặc chưa cộng phần cược đang chờ call của đối thủ, dẫn đến tính sai Pot Odds hệ thống.
   * *Xử lý:* Quy ước duy nhất một công thức trong `card_math.py` và hiển thị tường minh trong prompt: `Current Pot (including villain's bet): X BB | To Call: Y BB`, khi đó $\text{Pot Odds} = \frac{Y}{X + Y}$. Viết unit test khóa cứng công thức này.
2. **Đo `BB/100` bị nhiễu nặng trên mẫu nhỏ:**
   * *Rủi ro:* Trong NLHE, độ lệch chuẩn thường tới `70–100 BB/100`, chỉ đánh 200 ván thường có thể sai lệch kết quả rất lớn.
   * *Xử lý:* Bắt buộc dùng **Duplicate Deals** + báo cáo kèm **95% Confidence Interval** (Bootstrap hoặc Standard Error) và dùng thêm chỉ số **All-in Adjusted EV BB/100** song song với thực thắng.
3. **Nút cổ chai tốc độ khi tính EV Rollout:**
   * *Rủi ro:* Chạy `PokerKit` thuần Python 64 lần cho mỗi bước ở RL (Phase 4) sẽ rất chậm.
   * *Xử lý:* Ngay từ Phase 1, thiết kế `ev_calculator.py` hỗ trợ batch/multiprocessing và tách riêng `fast_rollout_state` (nhẹ hơn object đầy đủ của `PokerKit`) hoặc cache kết quả EV theo `(canonical_state_hash, action)`.
