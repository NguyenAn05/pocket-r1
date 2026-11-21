# Kế hoạch Chi tiết — Phase 2: Cold-start SFT, Masked Infilling & Baseline (Exp 1)

> **Thời lượng dự kiến:** Tuần 3  
> **Tài liệu gốc:** [PROJECT_IDEA.md](./PROJECT_IDEA.md)  
> **Giai đoạn trước:** [Phase 1 — Xây Phòng Gym & Đấu trường](./PHASE_1_GYM_AND_ARENA.md)  
> **Giai đoạn tiếp theo:** [Phase 3 — Huấn luyện DeBERTa-v3 Leak Finder](./PHASE_3_DEBERTA_PRM.md)

---

## 1. Mục tiêu & Phạm vi (Goals & Scope)

Base SLM (`Exp 0`) thường mắc nhiều lỗi cơ bản: sai định dạng `<step>`/`<action>`, chọn hành động bất hợp lệ, đọc nhầm chất bài hoặc tính sai Pot Odds. Nếu đưa thẳng vào RL (GRPO), model sẽ rơi vào tình trạng **Zero-Reward** hoặc chỉ tốn bước học để sửa cú pháp thay vì học chiến thuật.

Phase 2 giải quyết triệt để bước khởi động này bằng **Cold-start Supervised Fine-Tuning (SFT)** kết hợp bài tập bổ trợ **Masked Step Infilling (FIM)** để tạo ra checkpoint nền **`Exp 1 (SFT)`** vững chắc.

### Đầu ra bắt buộc của Phase 2
1. Pipeline tạo & làm sạch dữ liệu SFT đa nguồn (kết hợp *PokerBench* + tình huống tự sinh từ Gym Phase 1), chuẩn hóa 100% về định dạng **5-step Structured Poker CoT**.
2. Bộ dữ liệu trộn **Masked Step Infilling (FIM)** (che Bước 2 hoặc Bước 3) và cấu hình bật/tắt để đối chứng.
3. Pipeline huấn luyện SFT (`Unsloth` hoặc HuggingFace `TRL SFTTrainer`) với **Assistant-only Loss Masking** (chỉ tính loss trên phần trả lời của SLM).
4. Checkpoint **`Exp 1 (SFT)`** (và tùy chọn **`Exp 1b (SFT + FIM)`**) được đánh giá trên cùng **Held-out Evaluation Suite** của Phase 1, đạt ngưỡng ổn định sẵn sàng bước vào Phase 3 & 4.

---

## 2. Cấu trúc Module Bổ sung trong Repo

```text
pocket-r1/
├── configs/
│   └── training/
│       ├── sft_leduc.yaml
│       ├── sft_nlhe_base.yaml
│       └── sft_nlhe_fim.yaml
├── src/pocket_r1/
│   ├── data/
│   │   ├── pokerbench_adapter.py   # Chuyển đổi dữ liệu PokerBench về schema Gym
│   │   ├── cot_synthesizer.py      # Sinh CoT 5 bước chuẩn toán học (Template + Solver/LLM polish)
│   │   ├── fim_generator.py        # Tạo mẫu bài tập Masked Step Infilling (<MASK_STEP>)
│   │   └── dataset_builder.py      # Lọc verifier, dedup, tách Train/Val không leak
│   └── training/
│       └── sft_trainer.py          # Script huấn luyện Unsloth / TRL SFTTrainer
├── scripts/
│   ├── build_sft_dataset.py        # Chạy end-to-end tạo tập SFT sạch
│   ├── train_sft.py                # Train Exp 1 (và Exp 1b)
│   └── eval_exp1_sft.py            # Benchmark Exp 1 so với Exp 0
└── data/
    └── sft/
        ├── raw/
        ├── processed/
        │   ├── train_standard.jsonl
        │   ├── train_fim.jsonl
        │   └── val_sft.jsonl
        └── stats_report.json
```

---

## 3. Thiết kế Kỹ thuật Chi tiết

### 3.1. Chiến lược Tạo Dữ liệu Cold-start SFT (`src/pocket_r1/data/`)

Mục tiêu tập dữ liệu khởi động cho HU-NLHE: **10.000 – 25.000 mẫu chất lượng cao** (và **2.000 – 3.000 mẫu** cho prototype Leduc Poker), đảm bảo đa dạng mọi street (`PREFLOP`, `FLOP`, `TURN`, `RIVER`), mọi vị trí (`BTN`, `BB`), mọi loại hành động (`FOLD`, `CHECK_CALL`, `BET_33/75/150`, `ALL_IN`) và mọi kiểu mặt bàn (khô, ướt, đôi, có draw).

#### Nguồn dữ liệu kết hợp (2 nguồn bổ trợ nhau)
1. **Nguồn A — *PokerBench* Adapter (`pokerbench_adapter.py`):**
   * Tận dụng tập tình huống Pre-flop & Post-flop đã có đáp án tối ưu từ solver trong *PokerBench*.
   * Map hành động của *PokerBench* về đúng 6 hành động rời rạc (`ActionAbstraction` từ Phase 1) và chuyển ngữ cảnh sang `StateFormatter` của dự án.
2. **Nguồn B — Tự sinh từ Phòng Gym (`cot_synthesizer.py`):**
   * Lấy mẫu tình huống thực chiến khi cho `TAG Bot` + `EVCalculator` đấu với 4 loại Bot tính cách trong Gym (chọn hành động có `EV` cao nhất theo Monte Carlo + CRN từ Phase 1).
   * Mục đích: giúp SLM làm quen đúng với phân phối trạng thái, SPR và chuỗi lịch sử cược của chính Phòng Gym `pocket-r1` (tránh *distribution mismatch* khi chỉ dùng dữ liệu tĩnh bên ngoài).

#### Quy trình sinh CoT 5 bước "Không ảo giác" (Verified CoT Generation)
Thay vì để LLM lớn tự bịa toàn bộ 5 bước (dễ sai Pot Odds hoặc đếm sai outs), dùng quy trình **Hybrid Grounded Generation**:
1. **Bước 1–3 (Ground-truth bằng Code):** Dùng `card_math.py` (Phase 1) tính chính xác tuyệt đối: vị trí, bài, mặt bàn, sức mạnh tay bài, số outs, Pot Odds, Equity ước lượng.
2. **Bước 4–5 (Giải thích chiến thuật & Quyết định):**
   * *Phương án tiết kiệm (Zero API cost):* Dùng bộ khung câu đa dạng (*diverse templates*) kết hợp lý do dựa trên vị trí, SPR, hành vi đối thủ và mức chênh lệch `Equity - Pot Odds` + hành động tối ưu từ Solver/EV.
   * *Phương án nâng chất lượng diễn đạt (Khuyến nghị cho 3.000–5.000 mẫu lõi):* Đưa sẵn các con số ground-truth của Bước 1–3 + hành động tối ưu vào một LLM mạnh (chạy 1 lần duy nhất) để viết lại thành lời văn tự nhiên, mạch lạc, đa dạng văn phong nhưng **cấm thay đổi con số và hành động**.
3. **Bộ lọc Verifier 100% (`dataset_builder.py`):**
   * Mọi mẫu đưa vào tập SFT **bắt buộc phải pass**:
     * `Parser` hợp lệ 100% (đúng 5 `<step>`, đúng 1 `<action>` trong `legal_actions`).
     * `StepVerifier` (Phase 1) chấm ĐÚNG 100% Bước 1, 2, 3.
     * Tổng độ dài `Prompt + Response` $\le 850$ tokens (chừa khoảng an toàn cho DeBERTa-v3).
   * Cân bằng phân bố nhãn hành động (không để `FOLD` hoặc `CHECK_CALL` chiếm quá 70% gây bias hành vi thụ động).

### 3.2. Thiết kế Bài tập Bổ trợ: Masked Step Infilling (`fim_generator.py`)

Theo mục 3.5 trong [PROJECT_IDEA.md](./PROJECT_IDEA.md), bổ sung nhiệm vụ che-rồi-đoán ngay trên SLM để model học sâu quan hệ nhân quả hai chiều giữa các bước:

* **Cách tạo mẫu FIM:**
  * Lấy các mẫu đã qua bộ lọc chất lượng cao, chọn ngẫu nhiên ẩn **Bước 2** (sức mạnh bài/outs) hoặc **Bước 3** (toán học Pot Odds/Equity) — đây là hai bước cốt lõi dễ sinh ảo giác nhất.
  * Đưa toàn bộ `State Prompt` + các bước còn lại (kèm placeholder `<MASK_STEP>`) và `<action>` vào input; yêu cầu SLM sinh đúng nội dung bước bị che:
    ```text
    [Nhiệm vụ: Điền bước suy luận bị thiếu (<MASK_STEP>) sao cho nhất quán với trạng thái bàn, các bước trước/sau và hành động cuối.]
    <step>1. Tình huống: ...</step>
    <MASK_STEP>
    <step>3. Toán học: ...</step>
    <step>4. Đọc vị: ...</step>
    <step>5. Quyết định: ...</step>
    <action>CHECK_CALL</action>
    ```
  * Target output: `<step>2. Sức mạnh bài: ...</step>`.
* **Tỷ lệ pha trộn khuyến nghị:** `80%` mẫu sinh tuần tự 5 bước + `20%` mẫu Masked Step Infilling.
* **Kiểm chứng khoa học:** Lưu cờ cấu hình `enable_fim: true/false` để so sánh nhanh `Exp 1 (Standard SFT)` vs `Exp 1b (SFT + FIM)` trên tỷ lệ *Card Hallucination* và *Pot Odds Accuracy*.

### 3.3. Chống Rò rỉ Dữ liệu (Data Hygiene)

* **Zero Overlap với Held-out Eval:** Kiểm tra chéo `(hole_cards, board, action_history)` và `seed` giữa tập SFT và `data/eval_suites/hu_nlhe_heldout_v1.jsonl` để đảm bảo trùng lặp = `0`.
* **Tách Train / Val theo ván bài (`hand_id`):** Tất cả các decision point trong cùng một ván bài phải nằm cùng một phía (`train` hoặc `val`), không để Flop ở Train còn Turn của cùng ván đó ở Val.

### 3.4. Cấu hình Huấn luyện SFT (`src/pocket_r1/training/sft_trainer.py`)

* **Mô hình mục tiêu:** `Qwen2.5-3B-Instruct` / `Qwen3-4B` (hoặc `1.5B` / `7B` tùy VRAM sẵn có; giữ cố định 1 Base SLM xuyên suốt toàn bộ bảng Ablation `Exp 0`–`Exp 5`).
* **Kỹ thuật finetune:**
  * Dùng `Unsloth` hoặc `TRL SFTTrainer` + `PEFT LoRA` (hoặc QLoRA 4-bit nếu GPU $\le 16\text{GB}$ VRAM).
  * **LoRA config gợi ý:** `r = 32`, `lora_alpha = 64`, `lora_dropout = 0.05`, target toàn bộ linear layers (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`).
  * **Assistant-only Loss Masking:** Dùng `DataCollatorForCompletionOnlyLM` (hoặc chat template assistant mask) để **chỉ tính Cross-Entropy loss trên phần output của SLM**, không tính trên prompt trạng thái.
  * **Hyperparameters khởi điểm:** `lr = 2e-4` (LoRA) hoặc `2e-5` (Full), `cosine` scheduler, `warmup_ratio = 0.05`, `epochs = 2–3`, effective batch size `32–64`, `bf16 = True`, `max_seq_length = 1024`.
* **Checkpoint Selection:** Lưu checkpoint mỗi 0.5 epoch; chọn checkpoint tốt nhất không chỉ theo `val_loss` mà theo **mini-eval trên Gym** (tránh overfit học thuộc văn phong mà giảm khả năng tổng quát hóa).

---

## 4. Kế hoạch Triển khai Từng Bước (Work Breakdown)

- [ ] **Task 2.1 – Prototype Leduc SFT:** Sinh 2.500 mẫu CoT cho Leduc Poker, chạy thử `sft_trainer.py` từ đầu đến cuối để xác nhận pipeline không lỗi (loss masking, tokenization, lưu/load LoRA adapter vào `vLLM`).
- [ ] **Task 2.2 – Adapter *PokerBench* & Gym Synthesizer:** Implement `pokerbench_adapter.py` và `cot_synthesizer.py` cho HU-NLHE; sinh tập dữ liệu thô (~20.000 mẫu).
- [ ] **Task 2.3 – Lọc Verifier & Cân bằng phân bố:** Chạy `dataset_builder.py` lọc 100% mẫu lỗi, loại trùng lặp, cân bằng theo street & action, xuất báo cáo thống kê `data/sft/stats_report.json`.
- [ ] **Task 2.4 – Tạo tập Masked Step Infilling:** Chạy `fim_generator.py` tạo tập FIM (20% kích thước tập chính) và kiểm tra thủ công 30 mẫu ngẫu nhiên.
- [ ] **Task 2.5 – Huấn luyện Exp 1 (Standard SFT) & Exp 1b (SFT + FIM):** Train song song với cùng seed & budget, log toàn bộ loss curve, gradient norm và token throughput lên W&B.
- [ ] **Task 2.6 – Merge Adapter & Đánh giá toàn diện trên Held-out Suite:**
  * Merge LoRA (hoặc load LoRA trực tiếp trên `vLLM`), chạy `scripts/eval_exp1_sft.py` trên đủ 2.000 ván Duplicate Deals.
  * So sánh trực tiếp **Exp 0 vs Exp 1 vs Exp 1b** trên mọi chỉ số (`BB/100`, Format, Illegal Action, Hallucination, Pot Odds, Logic–Action Consistency).
  * Lưu mẫu rollout của Exp 1 (ở cả `temperature = 0.0` và `temperature = 0.8`) — tập rollout có lỗi thực tế này sẽ là **nguyên liệu vàng** để huấn luyện và test `DeBERTa-v3` ở Phase 3!

---

## 5. Tiêu chí Nghiệm thu (Definition of Done / Exit Criteria)

Phase 2 chỉ hoàn tất khi checkpoint `Exp 1` (chọn bản tốt hơn giữa Standard SFT và SFT+FIM) đạt các ngưỡng sau trên **Held-out Evaluation Suite**:

| Chỉ số | Mục tiêu ngưỡng đạt (Exit Gate) | Ý nghĩa |
| :--- | :--- | :--- |
| **Format Validity Rate** | $\ge 98\%$ | Đảm bảo parser đọc sạch 5 `<step>` + `<action>`, sẵn sàng cho DeBERTa PRM và GRPO. |
| **Illegal Action Rate** | $< 1.0\%$ | Model hiểu rõ `legal_actions` và luật cược. |
| **Card Hallucination Rate** | $< 5.0\%$ (giảm mạnh so với Exp 0) | Bước 1 & 2 bám sát trạng thái bàn thực tế. |
| **Pot Odds Accuracy** | $\ge 90\%$ | Model tính toán đúng phần số học cơ bản ở Bước 3. |
| **Độ dài Output trung bình** | $250 - 450$ tokens (99% dưới 650 tokens) | Không gây cắt cụt context của `DeBERTa-v3` (`max_len = 1024`). |
| **Hiệu suất `BB/100`** | Cải thiện rõ rệt (có ý nghĩa thống kê) so với `Exp 0` trên ít nhất 3/4 Bot | Đảm bảo không còn rơi vào vùng *Zero-Reward* khi vào RL. |

---

## 6. Rủi ro Đặc thù Phase 2 & Cách Xử lý

1. **Overfitting vào văn mẫu (Template Collapse):**
   * *Rủi ro:* Nếu toàn bộ tập SFT viết theo 1–2 mẫu câu cứng nhắc, SLM sẽ sinh câu rập khuôn, làm giảm đa dạng (*entropy*) khi vào GRPO (cả `G` chuỗi rollout giống hệt nhau → advantage bằng 0).
   * *Xử lý:* Đa dạng hóa phrasing ở Bước 4–5 (dùng nhiều template hoặc paraphrase bằng LLM), chỉ train tối đa 2–3 epochs, và kiểm tra **Rollout Diversity** (tỷ lệ hành động/lập luận khác nhau khi sampling $G=8$ ở `temp=0.8`) ngay cuối Phase 2.
2. **Mất khả năng tổng quát hóa / Quên kỹ năng suy luận nền:**
   * *Rủi ro:* Learning rate quá cao hoặc train quá nhiều epoch làm hỏng năng lực toán/ngôn ngữ sẵn có của SLM.
   * *Xử lý:* Dùng LoRA với rank vừa phải (`r=16..32`), theo dõi song song validation loss và độ chính xác Pot Odds trên các mức cược (bet sizes) chưa từng xuất hiện trong tập train.
3. **Mất cân bằng hành động (Fold-heavy hoặc Call-station bias):**
   * *Rủi ro:* Tập dữ liệu có quá nhiều tình huống `FOLD` khiến model sau SFT chơi cực kỳ nhát.
   * *Xử lý:* Stratify sampling theo 6 nhóm hành động khi tạo tập SFT (`dataset_builder.py`) và kiểm tra phân bố hành động dự đoán của `Exp 1` so với `TAG Bot`.
