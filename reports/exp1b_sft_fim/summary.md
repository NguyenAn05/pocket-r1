# Benchmark Report — `exp1b_sft_fim`

- **Agent / Model:** `Qwen/Qwen3-4B-exp1b-sft-fim`
- **Total Hands:** `2000` (5722 Hero decision points)
- **Overall Performance:** `-167.97 ± 110.28 BB/100` (95% CI)

## Per-Opponent Results

| Opponent | Hands | BB/100 (95% CI) | Format Valid | Illegal Act | Card Halluc | Pot Odds Acc | Consistency | Avg Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `calling_station` | 500 | `-252.30 ± 151.30` | 100.0% | 0.00% | 0.0% | 99.1% | 100.0% | 174 |
| `nit` | 500 | `-262.20 ± 177.27` | 99.9% | 0.11% | 0.0% | 97.4% | 100.0% | 173 |
| `maniac` | 500 | `+269.60 ± 314.53` | 100.0% | 0.00% | 0.0% | 94.7% | 100.0% | 170 |
| `tag` | 500 | `-427.00 ± 194.11` | 99.9% | 0.07% | 0.0% | 97.2% | 100.0% | 173 |
| **OVERALL** | **2000** | **`-167.97 ± 110.28`** | **100.0%** | **0.03%** | **0.0%** | **97.2%** | **100.0%** | **172** |
