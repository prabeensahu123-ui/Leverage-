# Leverage - Bitcoin Market Prediction System

A clean, research-oriented Bitcoin trading system built around:

- **Triple Barrier Labeling** (Marcos López de Prado)
- **RandomForest** classifier on carefully selected features
- Real OHLCV data from Delta Exchange
- Paper trading + optional live execution

## Project Structure

```
├── app.py                 # Streamlit dashboard (main interface)
├── paper_trade_step.py    # Automated paper trading runner
├── ml_features.py         # Feature engineering for ML
├── triple_barrier.py      # Labeling method
├── features.py            # ATR, volume, volatility features
├── forecasting.py         # Damped Holt + ensemble forecast
├── signal_engine.py       # Classic technical scoring (secondary)
├── fetch_ohlcv.py         # Data downloader
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Key Improvements Made

- Removed all synthetic / random price generation
- Unified strategy around RandomForest + Triple Barrier
- Clean file structure (removed numbered & spaced filenames)
- Paper mode enabled by default for safety
- Proper confidence thresholds before suggesting trades

## Important Notes

- This is still experimental. Always paper trade first.
- Past performance does not guarantee future results.
- Use at your own risk.
- Never risk money you cannot afford to lose.

## Next Recommended Steps

1. Run `paper_trade_step.py` regularly to collect real paper results
2. Expand feature validation across different market regimes
3. Add funding rate and news sentiment features later
4. Implement proper walk-forward backtesting with transaction costs
