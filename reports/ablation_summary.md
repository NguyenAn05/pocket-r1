# Pocket-R1 — Unified Ablation Study Report

| Experiment | Model | Hands | BB/100 (95% CI) | Format | Illegal | Step 2 Acc | Pot Odds | Consistency | Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `exp0_baseline` | `Qwen/Qwen3-4B` | 2000 | `-115.22 ± 66.38` | 99.8% | 0.09% | 41.1% | 88.3% | 99.9% | 155 |
| `exp1b_sft_fim` | `Qwen/Qwen3-4B-exp1b-sft-fim` | 2000 | `-167.97 ± 110.28` | 100.0% | 0.03% | 89.8% | 97.2% | 100.0% | 172 |
| `exp4_grpo_hybrid` | `Qwen/Qwen3-4B-exp4-grpo-hybrid` | 2000 | `-116.05 ± 105.39` | 99.6% | 0.37% | 95.1% | 98.0% | 100.0% | 171 |
| `exp5_coevolution` | `Qwen/Qwen3-4B-exp5-coevolution` | 2000 | `-142.30 ± 46.65` | 99.9% | 0.09% | 93.8% | 74.3% | 100.0% | 172 |
