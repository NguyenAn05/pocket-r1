# Benchmark Report — `exp0_baseline`

- **Agent / Model:** `Qwen/Qwen3-4B`
- **Total Hands:** `2000` (4360 Hero decision points)
- **Overall Performance:** `-115.22 ± 66.38 BB/100` (95% CI)

## Per-Opponent Results

| Opponent | Hands | BB/100 (95% CI) | Format Valid | Illegal Act | Card Halluc | Pot Odds Acc | Consistency | Avg Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `calling_station` | 500 | `-54.10 ± 63.65` | 99.7% | 0.13% | 0.0% | 81.3% | 99.9% | 155 |
| `nit` | 500 | `-136.90 ± 90.72` | 100.0% | 0.00% | 0.0% | 91.5% | 100.0% | 156 |
| `maniac` | 500 | `-172.40 ± 233.23` | 99.8% | 0.10% | 0.0% | 93.6% | 99.9% | 154 |
| `tag` | 500 | `-97.50 ± 62.51` | 99.6% | 0.09% | 0.0% | 90.5% | 99.7% | 156 |
| **OVERALL** | **2000** | **`-115.22 ± 66.38`** | **99.8%** | **0.09%** | **0.0%** | **88.3%** | **99.9%** | **155** |
