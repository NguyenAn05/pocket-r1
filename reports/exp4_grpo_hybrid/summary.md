# Benchmark Report — `exp4_grpo_hybrid`

- **Agent / Model:** `Qwen/Qwen3-4B-exp4-grpo-hybrid`
- **Total Hands:** `2000` (4042 Hero decision points)
- **Overall Performance:** `-116.05 ± 105.39 BB/100` (95% CI)

## Per-Opponent Results

| Opponent | Hands | BB/100 (95% CI) | Format Valid | Illegal Act | Card Halluc | Pot Odds Acc | Consistency | Avg Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `calling_station` | 500 | `-174.60 ± 137.97` | 100.0% | 0.00% | 0.0% | 99.5% | 100.0% | 173 |
| `nit` | 500 | `+15.00 ± 88.48` | 99.4% | 0.57% | 0.0% | 98.3% | 100.0% | 165 |
| `maniac` | 500 | `-13.50 ± 326.45` | 99.7% | 0.31% | 0.0% | 96.3% | 100.0% | 170 |
| `tag` | 500 | `-291.10 ± 209.40` | 99.0% | 1.03% | 0.0% | 98.1% | 100.0% | 171 |
| **OVERALL** | **2000** | **`-116.05 ± 105.39`** | **99.6%** | **0.37%** | **0.0%** | **98.0%** | **100.0%** | **171** |
