# Pocket-R1

Phòng Gym tự tiến hóa huấn luyện Small Language Model (SLM) làm chủ tư duy chiến thuật Poker (*High Risk – High Reward*) thông qua vòng lặp **SFT → DeBERTa-v3 Step-level PRM → RL (GRPO)** và **Adversarial Co-evolution**.

## Tài liệu Thiết kế & Kế hoạch
- Ý tưởng tổng thể: [docs/plans/PROJECT_IDEA.md](docs/plans/PROJECT_IDEA.md)
- Phase 1 — Gym, Đấu trường & Baseline (`Exp 0`): [docs/plans/PHASE_1_GYM_AND_ARENA.md](docs/plans/PHASE_1_GYM_AND_ARENA.md)
- Phase 2 — Cold-start SFT & Masked Infilling (`Exp 1`): [docs/plans/PHASE_2_COLD_START_SFT.md](docs/plans/PHASE_2_COLD_START_SFT.md)
- Phase 3 — DeBERTa-v3 Step-level PRM: [docs/plans/PHASE_3_DEBERTA_PRM.md](docs/plans/PHASE_3_DEBERTA_PRM.md)
- Phase 4 — GRPO, Co-evolution & Ablation (`Exp 2`–`Exp 5`): [docs/plans/PHASE_4_GRPO_COEVOLUTION.md](docs/plans/PHASE_4_GRPO_COEVOLUTION.md)

## Cài đặt nhanh
```bash
uv sync --extra dev
uv run pytest
```

## Chạy nhanh các script Phase 1
```bash
# Kiểm chứng 4 bot tính cách (Calling Station, Nit, Maniac, TAG)
uv run python scripts/sanity_check_bots.py --hands 300

# Sinh tập Held-out Evaluation Suite (Duplicate Deals, 2.000 ván)
uv run python scripts/generate_eval_suite.py --pairs-per-opponent 250

# Chạy benchmark Exp 0 (ví dụ với OpenAI-compatible / vLLM endpoint)
uv run python scripts/run_exp0_baseline.py --config configs/eval/exp0_baseline.yaml
```
