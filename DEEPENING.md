# Deeper system (started 2026-09-06)

## Why
Paper sample after Delta fees was ~flat (Rs 43 on Rs 15,000 / 10x).
15m targets were smaller than fees. Labels ignored costs.

## Phase 1 (in code now)
- Cost-aware labels: move < 0.15% becomes HOLD
- 15m is watch-only (no new auto paper trades)
- Signal threshold 52% + 12% edge
- SMA200 regime: BUY only in UP, SELL only in DOWN
- Position size from 1% account risk vs stop (capped at 10x)

## Phase 2 (next)
- Walk-forward report with fees
- Range vs trend vs expansion detector
- Feature importance on the cost-aware labels

## Phase 3 (later)
- Squeeze-then-spike scanner
- News/FX only as a filter, not a predictor
