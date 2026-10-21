
import yfinance as yf

symbol = "BELRISE.NS"

print(f"Testing Yahoo Finance for {symbol}...")

try:
    df = yf.Ticker(symbol).history(
        period="1mo",
        interval="1d",
        auto_adjust=False
    )

    if df.empty:
        print("TEST FAILED: No candle data returned.")
    else:
        print(f"TEST PASSED: {len(df)} candles received.")
        print("Latest candle:")
        print(df.tail(1).to_string())
        print("Latest timestamp:", df.index[-1])

except Exception as e:
    print("TEST FAILED:", repr(e))
