# C_ExhaustionFade Robustness Grid

## Status

C_ExhaustionFade: ROBUST_CANDIDATE

## Validation

- Replay-specific pytest: 11 passed
- Full pytest: 30 passed
- Anchor verdict: PASS
- Robustness verdict: STRONG_ROBUSTNESS
- Recommendation: proceed_to_paper_sim_design

## Anchor Reproduction

| Metric | Value |
|---|---:|
| bar_size | 750 |
| horizon | 36 |
| cost_bps | 12 |
| trade_count | 221 |
| net_expectancy_bps | 72.212596 |
| win_rate | 0.615385 |
| profit_factor | 2.191757 |
| worst_year | 2025 |

## Cost12 Neighbourhood

| bar_size | horizon | trade_count | net_expectancy_bps | win_rate | profit_factor | worst_year |
|---|---:|---:|---:|---:|---:|---:|
| 500 | 24 | 338 | 18.677216 | 0.556213 | 1.390106 | 2026 |
| 500 | 36 | 329 | 23.123484 | 0.537994 | 1.422463 | 2025 |
| 500 | 48 | 319 | 37.479677 | 0.564263 | 1.612460 | 2025 |
| 750 | 24 | 226 | 48.325184 | 0.557522 | 1.836510 | 2025 |
| 750 | 36 | 221 | 72.212596 | 0.615385 | 2.191757 | 2025 |
| 750 | 48 | 216 | 59.803637 | 0.597222 | 1.751360 | 2026 |
| 1000 | 24 | 139 | 45.039962 | 0.568345 | 1.687257 | 2023 |
| 1000 | 36 | 133 | 30.569653 | 0.571429 | 1.313832 | 2022 |
| 1000 | 48 | 131 | 31.528567 | 0.534351 | 1.271076 | 2026 |

## Interpretation

The reproduced 750/h36 anchor remains the best cost12 row.
Neighbouring horizons at 750 BTC remain positive.
Non-anchor bar sizes 500 and 1000 also remain positive at cost12.
This supports STRONG_ROBUSTNESS rather than a single-island overfit.
This does not mean live-ready.
The next phase is paper-simulation design, not live execution.

## Known Caveats

- Results still depend on historical 500/750/1000 BTC shard quality.
- No live slippage model yet.
- No latency model yet.
- No exchange execution adapter yet.
- 2025 remains the anchor worst year and should be monitored.
- Strategy remains long-only C_ExhaustionFade, not a portfolio.
