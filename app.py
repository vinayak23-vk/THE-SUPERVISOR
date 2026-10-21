import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="THE SUPERVISOR", page_icon="📈", layout="wide")
st.title("📈 THE SUPERVISOR")
st.caption("V0.1 • iPad-friendly web dashboard • Analysis only — no order execution")

def demo_data(n=260, seed=17):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.018, n)
    close = 180 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1] * (1 + rng.normal(0, 0.004, n-1))]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.018, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.018, n))
    volume = rng.integers(80_000, 300_000, n).astype(float)
    idx = rng.choice(np.arange(20, n), size=12, replace=False)
    volume[idx] *= rng.uniform(1.5, 2.6, len(idx))
    return pd.DataFrame({
        "Date": pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n),
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume
    })

def rsi_calc(s, period=14):
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100/(1+rs)

def indicators(df):
    df = df.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)
    for p in (10, 21, 50):
        df[f"MA{p}"] = df["Close"].rolling(p).mean()
    for p in (20, 50, 100, 200):
        df[f"EMA{p}"] = df["Close"].ewm(span=p, adjust=False, min_periods=p).mean()
    df["RSI14"] = rsi_calc(df["Close"], 14)
    e12 = df["Close"].ewm(span=12, adjust=False).mean()
    e26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = e12 - e26
    df["MACDSignal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    r = df["RSI14"]
    lo, hi = r.rolling(14).min(), r.rolling(14).max()
    df["StochRSI"] = (100 * (r-lo)/(hi-lo).replace(0, np.nan)).rolling(3).mean()
    df["VolumeAvg20"] = df["Volume"].rolling(20).mean()
    df["VolumeRatio"] = df["Volume"] / df["VolumeAvg20"].replace(0, np.nan)
    return df

with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.text_input("Stock symbol", "DEMO")
    mode = st.selectbox("Setup type", ["Support bounce", "Pre-breakout"])
    source = st.radio("Data source", ["Synthetic demo data", "Upload OHLCV CSV"])
    support = st.number_input("Support (₹)", min_value=0.0, value=180.0, step=0.5)
    resistance = st.number_input("Resistance (₹)", min_value=0.0, value=190.0, step=0.5)
    tolerance = st.slider("Level tolerance (%)", 0.1, 3.0, 0.75, 0.05)
    rsi_min = st.slider("RSI minimum", 0, 100, 45)
    rsi_max = st.slider("RSI maximum", 0, 100, 70)
    volume_multiple = st.slider("Volume / 20-bar average minimum", 0.5, 3.0, 1.2, 0.1)
    require_trend = st.checkbox("Require close above EMA50", True)

df = None
if source == "Upload OHLCV CSV":
    up = st.file_uploader("CSV columns required: Open, High, Low, Close, Volume. Date is recommended.", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            if not {"Open","High","Low","Close","Volume"}.issubset(df.columns):
                st.error("CSV is missing one or more required columns.")
                df = None
            elif "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.sort_values("Date")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
else:
    df = demo_data()

if df is not None:
    df = indicators(df)
    last = df.iloc[-1]
    close = float(last["Close"])
    rsi = last["RSI14"]
    vr = last["VolumeRatio"]
    ema50 = last["EMA50"]
    tol = tolerance / 100
    checks = []

    def check(name, passed, details):
        checks.append({"Condition": name, "Result": "PASS" if passed else "NOT MET", "Details": details})

    if mode == "Support bounce":
        near = support > 0 and abs(close-support)/support <= tol
        hold = support > 0 and float(last["Low"]) <= support*(1+tol) and close >= support*(1-tol)
        check("Close near support", near, f"Close ₹{close:.2f}; support ₹{support:.2f}")
        check("Support hold/rejection", hold, f"Low ₹{last['Low']:.2f}; close ₹{close:.2f}")
    else:
        near = resistance > 0 and abs(close-resistance)/resistance <= tol
        above = resistance > 0 and close > resistance
        check("Close near resistance", near, f"Close ₹{close:.2f}; resistance ₹{resistance:.2f}")
        check("Close above resistance", above, f"Close ₹{close:.2f}; resistance ₹{resistance:.2f}")

    rsi_ok = pd.notna(rsi) and rsi_min <= float(rsi) <= rsi_max
    check("RSI in configured range", rsi_ok, f"{float(rsi):.1f}" if pd.notna(rsi) else "Not enough history")
    vol_ok = pd.notna(vr) and float(vr) >= volume_multiple
    check("Volume expansion", vol_ok, f"{float(vr):.2f}× 20-bar average" if pd.notna(vr) else "Not enough history")
    trend_ok = pd.notna(ema50) and close > float(ema50)
    if require_trend:
        check("Close above EMA50", trend_ok, f"EMA50 ₹{float(ema50):.2f}" if pd.notna(ema50) else "Not enough history")

    passed = sum(x["Result"] == "PASS" for x in checks)
    total = len(checks)
    if passed == total:
        status, message = "Conditions met", "All configured checks passed. This is a rule match, not an automatic entry."
        st.success(status)
    elif passed >= max(2, total-2):
        status, message = "Watch", f"{passed}/{total} checks passed. Review unmet conditions and confirm price action manually."
        st.warning(status)
    else:
        status, message = "Not qualified", f"{passed}/{total} checks passed. Current data does not meet enough configured rules."
        st.info(status)
    st.subheader(f"{symbol.upper()} — {mode}")
    st.write(message)
    a,b,c,d = st.columns(4)
    a.metric("Latest close", f"₹{close:.2f}")
    b.metric("RSI (14)", f"{float(rsi):.1f}" if pd.notna(rsi) else "—")
    c.metric("Volume / average", f"{float(vr):.2f}×" if pd.notna(vr) else "—")
    d.metric("Checks passed", f"{passed}/{total}")
    st.markdown("#### Rule-by-rule breakdown")
    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)
    st.markdown("#### Recent price trend")
    view = df.tail(100).copy()
    if "Date" in view.columns:
        view = view.set_index("Date")
    st.line_chart(view[["Close","EMA20","EMA50"]], use_container_width=True)
    st.markdown("#### Indicator values")
    show = ["Close","MA10","MA21","MA50","EMA100","EMA200","RSI14","MACD","MACDSignal","StochRSI","VolumeRatio"]
    st.dataframe(df[show].tail(10).round(2), use_container_width=True)
    st.download_button("Download analysed CSV", df.to_csv(index=False), file_name=f"{symbol.lower()}_analysis.csv", mime="text/csv")

st.divider()
st.caption("Prototype only. Synthetic demo prices are not real market data. CSV data is not verified. No broker connection, live feed, trade execution, or investment advice.")
