# Benchmark Report — `exp5_coevolution`

- **Agent / Model:** `Qwen/Qwen3-4B-exp5-coevolution`
- **Total Hands:** `2000` (4252 Hero decision points)
- **Overall Performance:** `-142.30 ± 46.65 BB/100` (95% CI)

## Per-Opponent Results

| Opponent | Hands | BB/100 (95% CI) | Format Valid | Illegal Act | Card Halluc | Pot Odds Acc | Consistency | Avg Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `calling_station` | 500 | `-26.70 ± 44.42` | 100.0% | 0.00% | 0.0% | 82.9% | 100.0% | 173 |
| `nit` | 500 | `-47.30 ± 66.29` | 100.0% | 0.00% | 0.0% | 55.6% | 100.0% | 173 |
| `maniac` | 500 | `-253.90 ± 108.04` | 99.6% | 0.36% | 0.0% | 71.9% | 100.0% | 170 |
| `tag` | 500 | `-241.30 ± 127.26` | 100.0% | 0.00% | 0.0% | 74.6% | 100.0% | 172 |
| **OVERALL** | **2000** | **`-142.30 ± 46.65`** | **99.9%** | **0.09%** | **0.0%** | **74.3%** | **100.0%** | **172** |
