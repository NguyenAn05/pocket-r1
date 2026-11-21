# Kế hoạch Chi tiết — Phase 3: Huấn luyện Trọng tài DeBERTa-v3 Step-level PRM (Leak Finder)

> **Thời lượng dự kiến:** Tuần 4  
> **Tài liệu gốc:** [PROJECT_IDEA.md](./PROJECT_IDEA.md)  
> **Giai đoạn trước:** [Phase 2 — Cold-start SFT & Baseline](./PHASE_2_COLD_START_SFT.md)  
> **Giai đoạn tiếp theo:** [Phase 4 — Vòng lặp GRPO, Co-evolution & Ablation Study](./PHASE_4_GRPO_COEVOLUTION.md)

---

## 1. Mục tiêu & Phạm vi (Goals & Scope)

Phase 3 xây dựng "trái tim chấm lỗi" của `pocket-r1`: huấn luyện một mô hình **Encoder `DeBERTa-v3`** đóng vai trò **Process Reward Model (PRM)** chấm điểm xác suất đúng $p_k \in [0, 1]$ cho **từng bước suy luận** trong chuỗi 5 bước + hành động của SLM.

Điểm mấu chốt của Phase 3 là:
* **Tự động hóa 100% quy trình tạo dữ liệu gán nhãn theo bước** (không tốn công gán nhãn thủ công).
* **Tận dụng bản năng RTD + Bidirectional Attention** của `DeBERTa-v3` để phát hiện cả lỗi sự thật (Bước 1–3) lẫn lỗi mâu thuẫn logic / lập luận một đằng đánh một nẻo (Bước 4–5 + Action).
* **Chống lệch phân phối (On-policy Realism):** Huấn luyện trên cả dữ liệu *Synthetic Corruption* lẫn **lỗi thật từ rollout của Exp 0 & Exp 1**, rồi kiểm định gắt gao qua cổng chất lượng (`AUC-ROC`, `F1`, `Calibration`) trước khi cho phép tham gia RL ở Phase 4.

### Đầu ra bắt buộc của Phase 3
1. Pipeline tự động tạo tập dữ liệu gán nhãn step-level (`PRM Dataset v1`): ~25.000 – 50.000 chuỗi (cân bằng giữa chuỗi sạch, lỗi tổng hợp kiểu RTD và lỗi thật của SLM).
2. Kiến trúc **`DeBERTaStepPRM`** hỗ trợ token đặc biệt `[STEP]`, `Label Smoothing`, và hiệu chỉnh nhiệt độ (*Temperature Scaling*).
3. Checkpoint **`deberta_prm_v1`** đạt Quality Gate trên tập validation độc lập gồm **lỗi thật của SLM**.
4. Module **`PRMInferenceServer` / Batched Scorer** chạy FP16/BF16 mượt mà (< 2GB VRAM, tốc độ cao) sẵn sàng cắm vào vòng lặp GRPO.

---

## 2. Cấu trúc Module Bổ sung trong Repo

```text
pocket-r1/
├── configs/
│   └── prm/
│       ├── dataset_v1.yaml
│       └── deberta_v3_base.yaml
├── src/pocket_r1/
│   ├── prm/
│   │   ├── data_corruptor.py       # Synthetic Corruption kiểu RTD cho từng loại lỗi
│   │   ├── step_labeler.py         # Gán nhãn tự động: Code Verifier + MC Rollout + Optional LLM
│   │   ├── modeling_deberta.py     # Kiến trúc DeBERTa-v3 với [STEP] pooling head
│   │   ├── calibration.py          # Temperature Scaling & đo Expected Calibration Error (ECE)
│   │   ├── trainer.py              # Vòng train PyTorch / HF Trainer cho PRM
│   │   └── scorer.py               # Fast batched inference cho RL (Phase 4)
├── scripts/
│   ├── collect_onpolicy_rollouts.py # Sinh rollout đa dạng từ Exp 0 & Exp 1 (temp=0.7..1.0)
│   ├── build_prm_dataset.py        # Tạo & cân bằng tập dữ liệu PRM v1
│   ├── train_deberta_prm.py        # Huấn luyện DeBERTa-v3 PRM
│   └── eval_deberta_prm.py         # Chấm AUC/F1 từng bước, ECE, adversarial stress-test
└── data/
    └── prm_v1/
        ├── train.jsonl
        ├── val_synthetic.jsonl
        └── val_onpolicy_real.jsonl  # Tập kiểm thử quan trọng nhất: lỗi thật của SLM
```

---

## 3. Thiết kế Kỹ thuật Chi tiết

### 3.1. Kiến trúc Mô hình `DeBERTaStepPRM` (`modeling_deberta.py`)

* **Backbone:** `microsoft/deberta-v3-base` (~184M tham số, ~1GB VRAM ở FP16) cho vòng lặp nhanh; hỗ trợ nâng lên `microsoft/deberta-v3-large` (~435M) qua config.
* **Định dạng đầu vào & Token `[STEP]`:**
  * Thêm 1 special token `[STEP]` vào tokenizer của DeBERTa (`resize_token_embeddings`).
  * Biến đổi chuỗi `State Prompt + 5 bước + Action` thành chuỗi phẳng duy nhất có 6 mốc chấm điểm (hoặc 5 mốc nếu gộp Bước 5 + Action; khuyến nghị **6 mốc tách biệt** để tách rõ *lập luận quyết định* ở Bước 5 và *hành động cuối* `<action>`):
    ```text
    [CLS] {State Prompt} [SEP]
    {Step 1} [STEP] {Step 2} [STEP] {Step 3} [STEP] {Step 4} [STEP] {Step 5} [STEP] Action: {ACTION} [STEP] [SEP]
    ```
  * **Tại sao tách mốc thứ 6 cho `Action`?** Giúp DeBERTa chỉ rõ khi nào Bước 1–5 đều đúng nhưng `<action>` bị bấm ngược (*Logic–Action Contradiction*).
* **Scoring Head & Hàm Loss:**
  * Trích xuất hidden states $h_k \in \mathbb{R}^d$ tại chính xác vị trí các token `[STEP]` ($k = 1 \dots 6$).
  * Đi qua `LayerNorm -> Dropout(0.1) -> Linear(d, 1)` để ra logit $z_k$, xác suất đúng $p_k = \sigma(z_k / T)$ (với $T$ là nhiệt độ hiệu chỉnh).
  * **Label Smoothing (Chống chấm điểm cực đoan 0.0 / 1.0):**
    * Nhãn cứng $y_k \in \{0, 1\}$ được làm mượt thành $\tilde{y}_k \in \{0.05, 0.95\}$ (hoặc dùng nhãn liên tục $y_k \in [0, 1]$ cho Bước 4–5 nếu chấm bằng MC Rollout).
    * Dùng `BCEWithLogitsLoss` tính trung bình trên các vị trí `[STEP]` hợp lệ (có `step_mask` để hỗ trợ cả chuỗi bị thiếu bước khi đánh giá lỗi).
* **An toàn Context Length (`max_length = 1024`):**
  * Nếu hiếm khi chuỗi vượt 1024 tokens, **chỉ cắt gọn phần lịch sử cược cũ nhất trong `State Prompt`**, tuyệt đối không cắt mất phần 5 bước CoT và các token `[STEP]` ở cuối.

### 3.2. Chiến lược Tạo Dữ liệu Gán nhãn Tự động (3 Tầng Nguồn)

Để DeBERTa-v3 không chỉ giỏi bắt lỗi giả (dễ học vẹt dấu hiệu nhân tạo) mà bắt trúng **lỗi thật của SLM**, tập dữ liệu `PRM v1` kết hợp 3 nguồn:

#### Nguồn 1 — On-policy SLM Rollouts (~40% dữ liệu, quan trọng nhất)
* Dùng `Exp 0` và `Exp 1` sinh hàng nghìn chuỗi suy luận với `temperature = 0.7 – 1.0` trên tập tay bài Train.
* **Tự động gán nhãn từng bước (`step_labeler.py`):**
  * **Step 1, 2, 3:** Gán nhãn trực tiếp bởi `StepVerifier` (Phase 1) — độ chính xác gần như tuyệt đối cho đọc bài, sức mạnh tay bài, outs, Pot Odds, Equity.
  * **Step 4 & Step 5 & Action:**
    * Dùng **EV Regret** từ `EVCalculator` (Phase 1): so sánh $\text{EV}(a_{\text{chosen}})$ với $\max_a \text{EV}(a)$ (hoặc baseline hợp lý). Nếu $\text{EV}_{\text{regret}} > \tau_{\text{bad}}$ (ví dụ mất $> 2.0\text{ BB}$ kỳ vọng) → nhãn Action = `0`, và đánh dấu Step 5 = `0` nếu lập luận dẫn tới hành động đó.
    * Kiểm tra nhất quán lập luận–hành động (ví dụ: Step 3 kết luận `Equity < Pot Odds` và không có lý do bluff/implied odds nhưng Action vẫn `CALL`/`ALL_IN`, hoặc Step 5 khuyên `Fold` nhưng `<action>` là `BET_150`).
    * *(Tùy chọn nâng cao)*: Chưng cất tri thức từ một LLM lớn chấm mẫu 1 lần duy nhất cho **2.000 – 4.000 mẫu khó** ở Bước 4–5 (đọc vị & chiến thuật) để làm giàu tín hiệu mềm.

#### Nguồn 2 — Synthetic Corruption kiểu RTD (~40% dữ liệu)
Từ các chuỗi suy luận sạch (100% nhãn `[1, 1, 1, 1, 1, 1]`), script `data_corruptor.py` chủ động **phá hỏng có kiểm soát 1 hoặc 2 bước ngẫu nhiên** theo 6 nhóm lỗi trọng tâm:

| Nhóm lỗi | Cách tạo lỗi tổng hợp | Mốc bị gắn nhãn `0` |
| :--- | :--- | :--- |
| **1. Ảo giác lá bài / chất bài** | Đổi ngẫu nhiên 1 chất (`♠` → `♥`) hoặc rank bài ở Step 1 hoặc Step 2 | Step 1 (và/hoặc Step 2) |
| **2. Bịa tay bài / Đếm sai Outs** | Đổi `Ace-high` thành `Two Pair`, đổi `9 outs` thành `15 outs`, bịa Flush Draw khi mặt bàn rainbow | Step 2 |
| **3. Sai Pot Odds / So sánh ngược** | Thay đổi công thức hoặc kết quả Pot Odds lệch $\ge 10\%$; đảo dấu `>` ↔ `<` giữa Equity và Pot Odds | Step 3 |
| **4. Mâu thuẫn lịch sử cược** | Thay đổi hành động đối thủ được nhắc ở Step 1/Step 4 (ví dụ đối thủ `BET_75` nhưng viết là "đối thủ check yếu") | Step 1 hoặc Step 4 |
| **5. Lập luận ngược hành động** | Giữ nguyên Step 1–5 chuẩn nhưng tráo `<action>` sang hành động trái ngược (`FOLD` ↔ `ALL_IN`, `CHECK_CALL` ↔ `FOLD`) | Action (mốc 6) |
| **6. Lỗi lan truyền (Cascading Error)** | Phá Step 2 (bịa có thùng) và sửa Step 5/Action đánh All-in theo thông tin bịa đó | Step 2 = `0`, các bước ăn theo dựa trên tiền đề sai cũng được quy ước rõ (ưu tiên gán `0` từ bước sai đầu tiên trở đi — đúng tinh thần PRM *first-error detection*) |

* **Chống học vẹt (Anti-Artifact Rule):** Khi tạo Synthetic Corruption, đảm bảo **giữ nguyên cấu trúc câu, độ dài và dấu câu tự nhiên** để DeBERTa buộc phải đối chiếu ngữ nghĩa với `State Prompt` thay vì nhận diện lỗi đánh máy.

#### Nguồn 3 — Chuỗi Sạch đa dạng (~20% bổ sung để cân bằng tổng thể ~50% chuỗi toàn đúng / ~50% chuỗi có ít nhất 1 bước sai)
* Kết hợp từ tập SFT sạch (Phase 2) và các rollout đạt EV cao + pass toàn bộ verifier của `Exp 1`.

### 3.3. Quy tắc Gán nhãn Chuỗi Lỗi Lan truyền

Cần thống nhất rõ giữa Phase 3 và công thức gộp thưởng $R_{\text{DeBERTa}}$ ở Phase 4:
* Khi bước $k$ mắc lỗi nghiêm trọng (ví dụ Bước 2 bịa có Thùng), dù Bước 5 suy luận "có thùng nên All-in" là hợp lý *theo giả định sai*, toàn bộ lập luận sau bước $k$ đã mất giá trị.
* **Quy ước:**
  * Bước đúng trước lỗi: `1`.
  * Bước sai đầu tiên: `0`.
  * Các bước phía sau nếu phụ thuộc vào lỗi đó: `0` (hoặc giữ độc lập cho các bước không phụ thuộc, ví dụ Bước 3 tính Pot Odds vẫn có thể đúng dù Bước 2 đọc nhầm bài → ưu tiên **gán độc lập cho Bước 1–3**, và **lan truyền về Bước 5 + Action** nếu kết luận bị ảnh hưởng). Điều này giúp DeBERTa đưa tín hiệu phản hồi tinh gọn và chính xác hơn.

### 3.4. Hiệu chỉnh & Tối ưu Inference cho RL (`calibration.py`, `scorer.py`)

* **Calibration (Temperature Scaling):** Sau khi train xong, tối ưu 1 tham số scalar $T > 0$ trên tập `val_onpolicy_real.jsonl` để tối thiểu hóa NLL/ECE, giúp xác suất đầu ra $p_k$ phản ánh đúng xác suất thực tế (cực quan trọng khi kết hợp tuyến tính $R_{\text{EV}} + R_{\text{DeBERTa}}$ trong GRPO).
* **Tối ưu tốc độ (`scorer.py`):**
  * Chạy `bf16` / `fp16`, dynamic padding theo batch (giảm compute tới 2–3x so với pad cứng 1024).
  * Cung cấp hàm tiện ích `score_batch(prompts, responses) -> StepScoreBatch` trả về:
    * `step_probs`: tensor `(B, 6)`
    * `min_score`: $\min_k p_k$
    * `geom_mean`: $\left(\prod_k p_k\right)^{1/6}$
    * `first_error_step`: chỉ số bước có xác suất thấp nhất (phục vụ debug & UI demo).

---

## 4. Kế hoạch Triển khai Từng Bước (Work Breakdown)

- [ ] **Task 3.1 – Thu thập On-policy Rollouts:** Chạy `scripts/collect_onpolicy_rollouts.py` dùng `Exp 0` và `Exp 1` sinh ~15.000 rollout đa dạng (có cả đúng và sai thực tế) trên tập tay bài Train.
- [ ] **Task 3.2 – Xây `DataCorruptor` & `StepLabeler`:** Implement 6 nhóm Synthetic Corruption và pipeline gán nhãn tự động (Code Verifier + EV Regret + tùy chọn LLM Judge cho 2.000 mẫu Bước 4–5).
- [ ] **Task 3.3 – Đóng gói & Kiểm định dữ liệu `PRM v1`:** Tạo `train.jsonl`, `val_synthetic.jsonl`, và đặc biệt là `val_onpolicy_real.jsonl` (tách biệt hoàn toàn theo `hand_id`). Kiểm tra thủ công 50 mẫu ngẫu nhiên để xác nhận độ sạch nhãn tự động $\ge 95\%$.
- [ ] **Task 3.4 – Implement `DeBERTaStepPRM` & Trainer:** Xây kiến trúc pooling `[STEP]`, Label Smoothing (`0.05 / 0.95`), hỗ trợ gradient checkpointing + cosine LR (`lr = 2e-5`, `warmup = 0.1`, `epochs = 3–4`, `weight_decay = 0.01`).
- [ ] **Task 3.5 – Huấn luyện & Ablation nhanh cho PRM:**
  * Train `deberta-v3-base` trên Leduc trước (sanity check 15 phút).
  * Train `deberta-v3-base` trên HU-NLHE `PRM v1`; so sánh hiệu năng nếu thêm/bớt On-policy Rollouts so với chỉ dùng Synthetic Corruption.
- [ ] **Task 3.6 – Calibration & Stress-test (`eval_deberta_prm.py`):**
  * Fit Temperature Scaling, đo `ECE`.
  * Chạy **Adversarial Stress-test** (chống Reward Hacking sớm):
    * Test A: Giữ nguyên nội dung đúng nhưng đổi văn phong/paraphrase → điểm $p_k$ phải vẫn cao (không phụ thuộc từ khóa cứng).
    * Test B: Viết văn phong cực kỳ mượt mà, chuyên nghiệp nhưng cố tình gài 1 lỗi nhỏ (sai 1 chất bài hoặc sai Pot Odds 15%) → $p_k$ tại bước đó phải tụt mạnh (< 0.3).
- [ ] **Task 3.7 – Benchmark Tốc độ `PRM Scorer`:** Đo VRAM peak và throughput (sequences/sec) ở `batch_size = 32, 64` để đảm bảo không làm nghẽn cổ chai bước tính reward của GRPO ở Phase 4.

---

## 5. Tiêu chí Nghiệm thu (Quality Gate — Bắt buộc trước Phase 4)

Theo mục 7 của [PROJECT_IDEA.md](./PROJECT_IDEA.md), **tuyệt đối không đưa `DeBERTa-v3` vào vòng RL nếu chưa qua cổng kiểm duyệt này**:

| Hạng mục Kiểm duyệt | Tập đánh giá | Ngưỡng đạt tối thiểu |
| :--- | :--- | :--- |
| **Overall Step `AUC-ROC`** | `val_onpolicy_real.jsonl` | $\ge 0.88$ |
| **Overall Step `F1-score`** | `val_onpolicy_real.jsonl` | $\ge 0.85$ |
| **Step 1–3 `F1-score` (Lỗi sự thật & toán)** | `val_onpolicy_real.jsonl` | $\ge 0.92$ |
| **Action Consistency `AUC-ROC` (Mốc 6)** | `val_onpolicy_real.jsonl` | $\ge 0.88$ |
| **Calibration (`ECE`)** | `val_onpolicy_real.jsonl` | $\le 0.06$ |
| **Stress-test Paraphrase / Subtle Error** | Bộ 200 cặp đối chứng | Chênh lệch điểm trung bình $\Delta p \ge 0.45$ giữa mẫu đúng paraphrase và mẫu gài lỗi tinh vi |
| **Hiệu năng Inference** | GPU thực tế | VRAM $\le 2.5\text{ GB}$, tốc độ $\ge 100\text{ sequences/giây}$ |

---

## 6. Rủi ro Đặc thù Phase 3 & Cách Xử lý

1. **DeBERTa học vẹt dấu hiệu nhân tạo (Synthetic Shortcut):**
   * *Rủi ro:* Đạt `AUC = 0.99` trên `val_synthetic` nhưng tụt xuống `0.65` trên lỗi thật của SLM vì tập synthetic có mẫu câu hơi khựng.
   * *Xử lý:* Luôn lấy **`val_onpolicy_real.jsonl`** làm thước đo chính (early stopping & chọn checkpoint theo tập này), và bảo đảm tối thiểu 40% dữ liệu train đến từ lỗi thật của SLM.
2. **Mất cân bằng lớp giữa các bước (Step 1 ít sai, Step 3/5 hay sai):**
   * *Rủi ro:* DeBERTa luôn đoán `1` cho Step 1 để ăn loss thấp, bỏ lọt lỗi đọc nhầm bài hiếm gặp.
   * *Xử lý:* Tính `pos_weight` riêng cho từng vị trí bước $k \in \{1..6\}$ trong hàm `BCEWithLogitsLoss`, đồng thời báo cáo `AUC/F1` tách riêng theo từng bước (`Step 1` đến `Action`) thay vì chỉ nhìn số trung bình.
3. **Tokenizer mismatch hoặc mất `[STEP]` token:**
   * *Rủi ro:* Chuỗi quá dài bị cắt đuôi làm mất các token `[STEP]` cuối, gây lỗi shape hoặc train sai vị trí.
   * *Xử lý:* Viết assertion trong `DataLoader` kiểm tra chính xác số lượng `[STEP]` sau khi tokenize khớp với `len(labels)` ở 100% batch; áp dụng cắt chọn lọc phần lịch sử cũ trong `State Prompt` trước khi ghép CoT.
