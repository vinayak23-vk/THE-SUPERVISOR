import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone

st.set_page_config(page_title="THE SUPERVISOR", page_icon="📈", layout="wide")
st.title("📈 THE SUPERVISOR")
st.caption("V0.3 • Indian equities • Technical analysis only • No order execution")
st.warning("For analysis only, not financial advice. Yahoo Finance data may be delayed, incomplete, or unavailable. Verify important prices on your broker/chart.")

# ---------------- Data and indicators ----------------
YF_INTERVALS = {
    "1D": ("1d", "2y"),
    "1H": ("60m", "730d"),
    "4H": ("60m", "730d"),   # aggregated from hourly candles
    "15m": ("15m", "60d"),
    "5m": ("5m", "60d"),
}

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # Handle uninterrupted rises/falls more sensibly
    out = out.mask((loss == 0) & (gain > 0), 100)
    out = out.mask((gain == 0) & (loss > 0), 0)
    return out

def macd(series, fast=12, slow=26, signal=9):
    line = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig

def add_indicators(df):
    x = df.copy()
    for n in (10, 21, 50):
        x[f"MA{n}"] = x["Close"].rolling(n, min_periods=n).mean()
    for n in (100, 200):
        x[f"EMA{n}"] = x["Close"].ewm(span=n, adjust=False, min_periods=n).mean()
    x["RSI14"] = rsi(x["Close"])
    x["RSI_SMA14"] = x["RSI14"].rolling(14, min_periods=14).mean()
    x["MACD"], x["MACD_Signal"], x["MACD_Hist"] = macd(x["Close"])
    low = x["RSI14"].rolling(14, min_periods=14).min()
    high = x["RSI14"].rolling(14, min_periods=14).max()
    srsi = 100 * (x["RSI14"] - low) / (high - low).replace(0, np.nan)
    x["StochRSI_K"] = srsi.rolling(3, min_periods=3).mean()
    x["StochRSI_D"] = x["StochRSI_K"].rolling(3, min_periods=3).mean()
    x["VolumeAvg20"] = x["Volume"].rolling(20, min_periods=20).mean()
    x["VolumeRatio"] = x["Volume"] / x["VolumeAvg20"].replace(0, np.nan)
    prev_close = x["Close"].shift(1)
    true_range = pd.concat([
        x["High"] - x["Low"],
        (x["High"] - prev_close).abs(),
        (x["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    x["ATR14"] = true_range.rolling(14, min_periods=14).mean()
    return x

def clean_ohlcv(df):
    if df is None or df.empty:
        raise ValueError("The data provider returned no candles.")
    x = df.copy()

    # Flatten Yahoo Finance MultiIndex columns if present.
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [str(col[0] if str(col[0]).lower() in
                         {"open", "high", "low", "close", "adj close", "volume"}
                         else col[-1]) for col in x.columns]
    x.columns = [str(c).strip().replace("_", " ").title() for c in x.columns]
    aliases = {"Adj Close": "Close", "Datetime": "Date", "Timestamp": "Date"}
    x = x.rename(columns=aliases)

    if "Date" not in x.columns:
        x = x.reset_index()
        x.columns = [str(c).strip().replace("_", " ").title() for c in x.columns]
        x = x.rename(columns={"Datetime": "Date", "Index": "Date", "Timestamp": "Date"})
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise ValueError("Missing OHLCV columns: " + ", ".join(missing))
    if "Date" not in x.columns:
        x = x.reset_index().rename(columns={"index": "Date"})
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce", utc=True)
    # Convert timezone-aware timestamps to naive local timestamps for stable Streamlit charting.
    x["Date"] = x["Date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    for c in required:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["Date"] + required)
    x = x.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    if x.empty:
        raise ValueError("No valid OHLCV rows remained after cleaning.")
    # Discard obviously malformed candles.
    x = x[(x["High"] >= x[["Open", "Close", "Low"]].max(axis=1)) &
          (x["Low"] <= x[["Open", "Close", "High"]].min(axis=1)) &
          (x["Volume"] >= 0)]
    if x.empty:
        raise ValueError("All rows failed OHLCV data-quality checks.")
    return x[["Date", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)

def fetch_yahoo(symbol, timeframe):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed in the active environment. Add yfinance to the requirements file used by this Streamlit app.") from exc

    interval, period = YF_INTERVALS[timeframe]
    try:
        # auto_adjust=False keeps Yahoo's OHLC fields as returned by the provider.
        raw = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            prepost=False,
            timeout=20,
        )
    except Exception as exc:
        raise RuntimeError(f"Yahoo Finance request failed: {type(exc).__name__}: {exc}") from exc

    if raw is None or raw.empty:
        raise RuntimeError(
            f"Yahoo Finance returned no {timeframe} candles for {symbol}. "
            "Check the symbol suffix (Indian NSE symbols usually end in .NS), "
            "or try a timeframe with more available history."
        )
    raw = raw.reset_index()
    raw = clean_ohlcv(raw)

    if timeframe == "4H":
        # Yahoo does not offer a native 4-hour interval; aggregate hourly candles.
        raw = raw.set_index("Date").resample("4h", origin="start_day", offset="9h15min").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum"
        }).dropna(subset=["Open", "High", "Low", "Close"]).reset_index()
        raw = clean_ohlcv(raw)
    return raw

def normalize_csv(file):
    df = pd.read_csv(file)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {
        "date": ["date", "datetime", "timestamp", "time", "index"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "adj_close", "ltp", "c"],
        "volume": ["volume", "vol", "v", "qty"],
    }
    mapping = {}
    for target, opts in aliases.items():
        found = next((c for c in opts if c in df.columns), None)
        if found:
            mapping[found] = target
    df = df.rename(columns=mapping)
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing) + ". Required: Date (recommended), Open, High, Low, Close, Volume.")
    if "date" not in df.columns:
        df["date"] = pd.date_range("2000-01-01", periods=len(df), freq="D")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    for c in required:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df = df.dropna(subset=["date"] + required).rename(
        columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    df["Date"] = df["Date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = clean_ohlcv(df)
    if len(df) < 30:
        raise ValueError(f"Only {len(df)} usable candles found. At least 30 are needed; 200+ is preferable for EMA200.")
    return df

def make_demo(n=260):
    rng = np.random.default_rng(17)
    ret = rng.normal(0.0006, 0.018, n)
    close = 150 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1] * (1 + rng.normal(0, 0.006, n - 1))]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.018, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.018, n))
    volume = rng.integers(100000, 900000, n)
    return pd.DataFrame({
        "Date": pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B"),
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume
    })

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Setup")
    symbol = st.text_input("Stock symbol", "BELRISE.NS", help="For NSE stocks, Yahoo symbols commonly use .NS; BSE symbols commonly use .BO.").strip().upper()
    source = st.selectbox("Data source", ["Yahoo Finance", "Upload OHLCV CSV", "Synthetic demo data"])
    timeframe = st.selectbox("Timeframe", ["1D", "1H", "4H", "15m", "5m"], index=0)
    if source == "Yahoo Finance":
        st.caption("Yahoo data is unofficial and may be delayed or intermittently unavailable. 5m/15m history is limited.")
        refresh = st.button("🔄 Fetch / refresh data", use_container_width=True)
    else:
        refresh = False
    st.divider()
    setup = st.selectbox("Setup to evaluate", ["Support bounce", "Pre-breakout", "Breakout retest"])
    support = st.number_input("Support level (₹)", min_value=0.0, value=180.0, step=0.5)
    resistance = st.number_input("Resistance level (₹)", min_value=0.0, value=190.0, step=0.5)
    tolerance = st.slider("Level tolerance (%)", 0.1, 3.0, 0.75, 0.05)
    rsi_min = st.slider("RSI lower bound", 10, 70, 40)
    rsi_max = st.slider("RSI upper bound", 30, 90, 70)
    vol_mult = st.slider("Volume expansion threshold (× 20-bar average)", 1.0, 3.0, 1.2, 0.1)
    atr_mult = st.slider("Stop distance (ATR)", 1.0, 4.0, 2.0, 0.25)
    risk_pct = st.slider("Planned account risk (%)", 0.25, 2.0, 0.5, 0.25)
    capital = st.number_input("Trading capital (₹)", min_value=1000, value=100000, step=5000)

# ---------------- Data selection ----------------
raw = None
data_note = ""
if source == "Yahoo Finance":
    if not symbol:
        st.error("Enter a Yahoo Finance symbol, e.g. BELRISE.NS.")
        st.stop()
    # Cache successful requests briefly; the button can clear the cache to force a fetch.
    if refresh:
        st.cache_data.clear()
    try:
        @st.cache_data(ttl=300, show_spinner=False)
        def cached_yahoo(symbol_arg, timeframe_arg):
            return fetch_yahoo(symbol_arg, timeframe_arg)
        with st.spinner(f"Fetching {symbol} {timeframe} candles from Yahoo Finance…"):
            raw = cached_yahoo(symbol, timeframe)
        data_note = f"Yahoo Finance • {symbol} • {timeframe} • fetched {datetime.now().strftime('%d %b %Y, %H:%M:%S')} app time"
        st.success(f"Loaded {len(raw):,} candles from Yahoo Finance.")
    except Exception as e:
        st.error(f"Could not load Yahoo Finance data: {e}")
        st.info("Try BELRISE.NS for NSE, or switch to CSV/demo data to confirm the dashboard itself works.")
        st.stop()
elif source == "Upload OHLCV CSV":
    uploaded = st.file_uploader("Upload candle data CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV with Date, Open, High, Low, Close and Volume.")
        st.stop()
    try:
        raw = normalize_csv(uploaded)
        data_note = f"CSV upload • {len(raw):,} candles"
    except Exception as e:
        st.error(f"Could not read this CSV: {e}")
        st.stop()
else:
    raw = make_demo()
    data_note = "Synthetic demo data only — not a real stock or market feed."
    if symbol == "BELRISE.NS":
        st.caption("The demo prices are synthetic and do not represent BELRISE.")

if raw is None or raw.empty:
    st.error("No candle data is available. Select another source or timeframe.")
    st.stop()
if len(raw) < 30:
    st.warning(f"Only {len(raw)} candles loaded. Some indicators will be unavailable; 200+ candles are preferable for EMA200.")

if rsi_min >= rsi_max:
    st.error("RSI lower bound must be less than RSI upper bound.")
    st.stop()
if support and resistance and support >= resistance:
    st.warning("Your support is at or above resistance. Check the levels before interpreting the setup.")

df = add_indicators(raw)
last = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else last
close = float(last["Close"])
tol = tolerance / 100
near_support = bool(support > 0 and abs(close - support) / support <= tol)
support_rejection = bool(support > 0 and last["Low"] <= support * (1 + tol) and close > support)
rsi_ok = bool(pd.notna(last["RSI14"]) and rsi_min <= last["RSI14"] <= rsi_max)
vol_ok = bool(pd.notna(last["VolumeRatio"]) and last["VolumeRatio"] >= vol_mult)
above_ema100 = bool(pd.notna(last["EMA100"]) and close > last["EMA100"])
bull_candle = bool(close > float(last["Open"]))
resistance_approach = bool(resistance > 0 and close < resistance and close >= resistance * (1 - 3 * tol))
retest = bool(resistance > 0 and last["Low"] <= resistance * (1 + tol) and close > resistance and bull_candle)

if setup == "Support bounce":
    checks = [
        ("Price near support", near_support or support_rejection, f"Close ₹{close:.2f}; support ₹{support:.2f}"),
        ("Support held / rejection", support_rejection, f"Low ₹{last['Low']:.2f}; close ₹{close:.2f}"),
        ("RSI in configured range", rsi_ok, f"RSI {last['RSI14']:.1f}" if pd.notna(last["RSI14"]) else "RSI unavailable"),
        ("Volume expansion", vol_ok, f"{last['VolumeRatio']:.2f}× 20-bar average" if pd.notna(last["VolumeRatio"]) else "Volume average unavailable"),
        ("Close above EMA100", above_ema100, f"EMA100 ₹{last['EMA100']:.2f}" if pd.notna(last["EMA100"]) else "EMA100 unavailable"),
    ]
elif setup == "Pre-breakout":
    checks = [
        ("Price approaching resistance", resistance_approach, f"Close ₹{close:.2f}; resistance ₹{resistance:.2f}"),
        ("No confirmed breakout yet", bool(resistance > 0 and close <= resistance), f"Close ₹{close:.2f} vs resistance ₹{resistance:.2f}"),
        ("Bullish candle", bull_candle, f"Open ₹{last['Open']:.2f}; close ₹{close:.2f}"),
        ("RSI in configured range", rsi_ok, f"RSI {last['RSI14']:.1f}" if pd.notna(last["RSI14"]) else "RSI unavailable"),
        ("Volume expansion", vol_ok, f"{last['VolumeRatio']:.2f}× 20-bar average" if pd.notna(last["VolumeRatio"]) else "Volume average unavailable"),
    ]
else:
    checks = [
        ("Breakout level reclaimed", retest, f"Low ₹{last['Low']:.2f}; close ₹{close:.2f}; resistance ₹{resistance:.2f}"),
        ("Bullish candle", bull_candle, f"Open ₹{last['Open']:.2f}; close ₹{close:.2f}"),
        ("RSI in configured range", rsi_ok, f"RSI {last['RSI14']:.1f}" if pd.notna(last["RSI14"]) else "RSI unavailable"),
        ("Volume expansion", vol_ok, f"{last['VolumeRatio']:.2f}× 20-bar average" if pd.notna(last["VolumeRatio"]) else "Volume average unavailable"),
        ("Close above EMA100", above_ema100, f"EMA100 ₹{last['EMA100']:.2f}" if pd.notna(last["EMA100"]) else "EMA100 unavailable"),
    ]

score = round(100 * sum(bool(c[1]) for c in checks) / len(checks))
st.subheader(f"{symbol or 'Stock'} — {setup} — {timeframe}")
st.caption(data_note)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest close", f"₹{close:,.2f}")
c2.metric("RSI (14)", f"{last['RSI14']:.1f}" if pd.notna(last["RSI14"]) else "—")
c3.metric("Volume / average", f"{last['VolumeRatio']:.2f}×" if pd.notna(last["VolumeRatio"]) else "—")
c4.metric("Rule checks", f"{sum(bool(c[1]) for c in checks)}/{len(checks)}", f"{score}% passed")

if all(bool(c[1]) for c in checks):
    st.success("All configured checks passed. Candidate for manual review only; not an automatic buy signal.")
elif sum(bool(c[1]) for c in checks) >= 3:
    st.info("Partial confluence. Review failed conditions and price structure before acting.")
else:
    st.warning("Not qualified under the current rules. Do not force a trade.")
st.caption("The checklist is not a validated predictive model. A high pass count does not imply a profitable trade.")

st.subheader("Rule-by-rule breakdown")
breakdown = pd.DataFrame(checks, columns=["Condition", "Pass", "Details"])
breakdown["Result"] = breakdown["Pass"].map({True: "PASS", False: "NOT MET"})
st.dataframe(breakdown.drop(columns="Pass"), hide_index=True, use_container_width=True)

atr = float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan
st.subheader("Illustrative risk planner")
if pd.notna(atr) and atr > 0:
    stop = close - atr_mult * atr
    per_share = close - stop
    risk_cash = capital * risk_pct / 100
    shares = max(0, int(risk_cash / per_share))
    max_value = shares * close
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("ATR (14)", f"₹{atr:.2f}")
    r2.metric("Illustrative stop", f"₹{stop:.2f}")
    r3.metric("Risk budget", f"₹{risk_cash:,.0f}")
    r4.metric("Max shares by risk", f"{shares:,}")
    st.caption(f"Approximate position value at this size: ₹{max_value:,.0f}. Excludes costs, gaps, liquidity and lot-size constraints. This is not a trade recommendation.")
else:
    st.info("Risk planner needs at least 14 usable candles to estimate ATR.")

st.subheader("Price and trend")
chart_cols = ["Close", "MA10", "MA21", "MA50", "EMA100", "EMA200"]
st.line_chart(df.set_index("Date")[chart_cols].tail(250), use_container_width=True)
show_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "MA10", "MA21", "MA50", "EMA100", "EMA200",
             "RSI14", "RSI_SMA14", "MACD", "MACD_Signal", "MACD_Hist", "StochRSI_K", "StochRSI_D", "VolumeRatio", "ATR14"]
st.dataframe(df[[c for c in show_cols if c in df.columns]].tail(50).round(3), hide_index=True, use_container_width=True)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download analysed candles CSV", csv_bytes,
                   file_name=f"{(symbol or 'stock').replace('.', '_').lower()}_{timeframe.lower()}_supervisor.csv",
                   mime="text/csv")

with st.expander("Data notes and limitations"):
    st.markdown("""
- **Daily:** up to about 2 years requested by this app.
- **1-hour:** requested up to about 730 days, subject to Yahoo's actual availability.
- **4-hour:** aggregated from hourly candles; these bars may not align exactly with a charting platform's session boundaries.
- **15-minute and 5-minute:** intraday history is limited (the app requests 60 days); provider availability can change.
- Yahoo Finance is an unofficial source for this use. It is not guaranteed real-time, complete, or suitable as the sole basis for a trade.
- One row represents one candle. Intraday bars may include the current, still-forming candle during market hours.
- Use at least 200 daily candles if you want a fully populated EMA200. Intraday EMA200 is calculated on bars of the selected timeframe.
""")
