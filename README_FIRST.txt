THE SUPERVISOR V0.2 — iPad-friendly upgrade

WHAT'S NEW
- Upload actual OHLCV candle CSVs.
- MA10/21/50, EMA100/200, RSI14 + RSI average, MACD, Stoch RSI, volume ratio, ATR14.
- Configurable support bounce, pre-breakout and breakout-retest checklists.
- Basic ATR-based illustrative risk sizing.
- Download analysed candle data.

LIMITATIONS
- No live feed, no broker connection, no background alerts, no order execution.
- Synthetic demo prices are fake and only demonstrate the interface.
- Signals are simple rule checks, not a validated predictive model.
- Use at least 200-300 candles for useful EMA200 context.

DEPLOY
1. Replace the existing app.py in your GitHub repository with this app.py.
2. Keep requirements.txt in the repo (replace it with the supplied one).
3. Streamlit Cloud should redeploy automatically.
4. Upload CSV from the sidebar to analyse real historical candles.
