import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="THE SUPERVISOR", page_icon="📈", layout="wide")
st.title("📈 THE SUPERVISOR")
st.caption("V0.2 • Indian equities • Technical analysis only • No order execution")
st.warning("Analysis tool, not financial advice. Verify every setup on your chart. No signal is guaranteed. Demo data is synthetic.")

# ---------- indicators ----------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    line = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line-sig

def add_indicators(df):
    x=df.copy()
    for n in (10,21,50):
        x[f"MA{n}"]=x["Close"].rolling(n).mean()
    for n in (100,200):
        x[f"EMA{n}"]=x["Close"].ewm(span=n,adjust=False).mean()
    x["RSI14"]=rsi(x["Close"])
    x["RSI_SMA14"]=x["RSI14"].rolling(14).mean()
    x["MACD"],x["MACD_Signal"],x["MACD_Hist"]=macd(x["Close"])
    low=x["RSI14"].rolling(14).min()
    high=x["RSI14"].rolling(14).max()
    srsi=100*(x["RSI14"]-low)/(high-low).replace(0,np.nan)
    x["StochRSI_K"]=srsi.rolling(3).mean()
    x["StochRSI_D"]=x["StochRSI_K"].rolling(3).mean()
    x["VolumeAvg20"]=x["Volume"].rolling(20).mean()
    x["VolumeRatio"]=x["Volume"]/x["VolumeAvg20"].replace(0,np.nan)
    x["ATR14"]=pd.concat([
        x["High"]-x["Low"],
        (x["High"]-x["Close"].shift()).abs(),
        (x["Low"]-x["Close"].shift()).abs()
    ],axis=1).max(axis=1).rolling(14).mean()
    return x

def normalize_csv(file):
    df=pd.read_csv(file)
    df.columns=[str(c).strip().lower().replace(" ","_") for c in df.columns]
    aliases={
        "date":["date","datetime","timestamp","time"],
        "open":["open","o"],
        "high":["high","h"],
        "low":["low","l"],
        "close":["close","adj_close","ltp","c"],
        "volume":["volume","vol","v","qty"]
    }
    mapping={}
    for target, opts in aliases.items():
        found=next((c for c in opts if c in df.columns),None)
        if found: mapping[found]=target
    df=df.rename(columns=mapping)
    needed=["open","high","low","close","volume"]
    missing=[c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: "+", ".join(missing)+". Required: Date (optional), Open, High, Low, Close, Volume.")
    if "date" not in df.columns:
        df["date"]=np.arange(len(df))
    df["date"]=pd.to_datetime(df["date"],errors="coerce")
    for c in needed:
        df[c]=pd.to_numeric(df[c].astype(str).str.replace(",","",regex=False),errors="coerce")
    df=df.dropna(subset=["open","high","low","close","volume"])
    df=df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume","date":"Date"})
    df=df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    if len(df)<30:
        raise ValueError(f"Only {len(df)} usable candles found. Upload at least 30; 200+ is better for EMA200.")
    return df

def make_demo(n=260):
    rng=np.random.default_rng(17)
    ret=rng.normal(0.0006,0.018,n)
    close=150*np.exp(np.cumsum(ret))
    open_=np.r_[close[0],close[:-1]*(1+rng.normal(0,0.006,n-1))]
    high=np.maximum(open_,close)*(1+rng.uniform(0.001,0.018,n))
    low=np.minimum(open_,close)*(1-rng.uniform(0.001,0.018,n))
    volume=rng.integers(100000,900000,n)
    return pd.DataFrame({"Date":pd.date_range(end=pd.Timestamp.today().normalize(),periods=n,freq="B"),
        "Open":open_,"High":high,"Low":low,"Close":close,"Volume":volume})

# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Setup")
    symbol=st.text_input("Stock symbol", "DEMO").strip().upper() or "UNKNOWN"
    mode=st.radio("Price data",["Upload OHLCV CSV","Synthetic demo data"])
    st.caption("CSV must contain Open, High, Low, Close, Volume; Date/Datetime is strongly recommended.")
    st.divider()
    setup=st.selectbox("Setup to evaluate",["Support bounce","Pre-breakout","Breakout retest"])
    support=st.number_input("Support level (₹)",min_value=0.0,value=180.0,step=0.5)
    resistance=st.number_input("Resistance level (₹)",min_value=0.0,value=190.0,step=0.5)
    tolerance=st.slider("Level tolerance (%)",0.1,3.0,0.75,0.05)
    rsi_min=st.slider("RSI lower bound",10,70,40)
    rsi_max=st.slider("RSI upper bound",30,90,70)
    vol_mult=st.slider("Volume expansion threshold (× 20-bar average)",1.0,3.0,1.2,0.1)
    atr_mult=st.slider("Stop distance (ATR)",1.0,4.0,2.0,0.25)
    risk_pct=st.slider("Planned account risk (%)",0.25,2.0,0.5,0.25)
    capital=st.number_input("Trading capital (₹)",min_value=1000,value=100000,step=5000)

if mode=="Upload OHLCV CSV":
    uploaded=st.file_uploader("Upload candle data CSV",type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV exported from your charting/broker platform to analyse actual prices. Until then, you can switch to synthetic demo data.")
        st.stop()
    try:
        raw=normalize_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read this CSV: {e}")
        st.stop()
else:
    raw=make_demo()
    if symbol=="DEMO":
        st.caption("Synthetic sample prices only — not a real stock or market feed.")

if rsi_min>=rsi_max:
    st.error("RSI lower bound must be less than RSI upper bound.")
    st.stop()
if support and resistance and support>=resistance:
    st.warning("Your support is at or above resistance. Check the levels before interpreting the setup.")

df=add_indicators(raw)
last=df.iloc[-1]
prev=df.iloc[-2]
close=float(last["Close"])
tol=tolerance/100
near_support=bool(support>0 and abs(close-support)/support<=tol)
near_resistance=bool(resistance>0 and abs(close-resistance)/resistance<=tol)
support_rejection=bool(support>0 and last["Low"]<=support*(1+tol) and close>support)
rsi_ok=bool(pd.notna(last["RSI14"]) and rsi_min<=last["RSI14"]<=rsi_max)
vol_ok=bool(pd.notna(last["VolumeRatio"]) and last["VolumeRatio"]>=vol_mult)
above_ema100=bool(pd.notna(last["EMA100"]) and close>last["EMA100"])
bull_candle=bool(close>float(last["Open"]))
resistance_approach=bool(resistance>0 and close<resistance and close>=resistance*(1-3*tol))
breakout=bool(resistance>0 and close>resistance and float(prev["Close"])<=resistance and vol_ok)
retest=bool(resistance>0 and last["Low"]<=resistance*(1+tol) and close>resistance and bull_candle)

if setup=="Support bounce":
    checks=[
        ("Price near support",near_support or support_rejection,f"Close ₹{close:.2f}; support ₹{support:.2f}"),
        ("Support held / rejection",support_rejection,f"Low ₹{last['Low']:.2f}; close ₹{close:.2f}"),
        ("RSI in configured range",rsi_ok,f"RSI {last['RSI14']:.1f}" if pd.notna(last["RSI14"]) else "RSI unavailable"),
        ("Volume expansion",vol_ok,f"{last['VolumeRatio']:.2f}× 20-bar average" if pd.notna(last["VolumeRatio"]) else "Volume average unavailable"),
("Close above EMA100", above_ema50, f"EMA100 ₹{last['EMA100']:.2f}" if pd.notna(last["EMA100"]) else "EMA100 unavailable"),]
elif setup=="Pre-breakout":
    checks= [
        ("Price approaching resistance",resistance_approach,f"Close ₹{close:.2f}; resistance ₹{resistance:.2f}"),
        ("No confirmed breakout yet",bool(resistance>0 and close<=resistance),f"Close ₹{close:.2f} vs resistance ₹{resistance:.2f}"),
        ("Bullish candle",bull_candle,f"Open ₹{last['Open']:.2f}; close ₹{close:.2f}"),
        ("RSI in configured range",rsi_ok,f"RSI {last['RSI14']:.1f}" if pd.notna(last["RSI14"]) else "RSI unavailable"),
        ("Volume expansion",vol_ok,f"{last['VolumeRatio']:.2f}× 20-bar average" if pd.notna(last["VolumeRatio"]) else "Volume average unavailable")
    ]
else:
    checks=[
        ("Breakout level reclaimed",retest,f"Low ₹{last['Low']:.2f}; close ₹{close:.2f}; resistance ₹{resistance:.2f}"),
        ("Bullish candle",bull_candle,f"Open ₹{last['Open']:.2f}; close ₹{close:.2f}"),
        ("RSI in configured range",rsi_ok,f"RSI {last['RSI14']:.1f}" if pd.notna(last["RSI14"]) else "RSI unavailable"),
        ("Volume expansion",vol_ok,f"{last['VolumeRatio']:.2f}× 20-bar average" if pd.notna(last["VolumeRatio"]) else "Volume average unavailable"),
        ("Close above EMA100", above_ema50, f"EMA100 ₹{last['EMA100']:.2f}" if pd.notna(last["EMA100"]) else "EMA100 unavailable")

passed=sum(bool(c[1]) for c in checks)
score=round(100*passed/len(checks))
st.subheader(f"{symbol} — {setup}")
c1,c2,c3,c4=st.columns(4)
c1.metric("Latest close",f"₹{close:,.2f}")
c2.metric("RSI (14)",f"{last['RSI14']:.1f}" if pd.notna(last["RSI14"]) else "—")
c3.metric("Volume / average",f"{last['VolumeRatio']:.2f}×" if pd.notna(last["VolumeRatio"]) else "—")
c4.metric("Rule checks",f"{passed}/{len(checks)}",f"{score}% passed")
if passed==len(checks):
    st.success("All configured checks passed. This is a candidate for manual review, not an automatic buy signal.")
elif passed>=3:
    st.info("Partial confluence. Review the failed conditions and price structure before acting.")
else:
    st.warning("Not qualified under the current rules. Do not force a trade.")
st.caption("This simple checklist is not a validated predictive model. A high pass count does not imply a profitable trade.")

st.subheader("Rule-by-rule breakdown")
st.dataframe(pd.DataFrame(checks,columns=["Condition","Pass","Details"]).assign(Result=lambda z:z["Pass"].map({True:"PASS",False:"NOT MET"})).drop(columns="Pass"),hide_index=True,width="stretch")

# risk plan, based on latest close and ATR; not an instruction to trade
atr=float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan
if pd.notna(atr) and atr>0:
    stop=close-atr_mult*atr
    per_share=close-stop
    risk_cash=capital*risk_pct/100
    shares=max(0,int(risk_cash/per_share))
    st.subheader("Illustrative risk planner")
    r1,r2,r3,r4=st.columns(4)
    r1.metric("ATR (14)",f"₹{atr:.2f}")
    r2.metric("Illustrative stop",f"₹{stop:.2f}")
    r3.metric("Risk budget",f"₹{risk_cash:,.0f}")
    r4.metric("Max shares by risk",f"{shares:,}")
    st.caption("Estimate excludes brokerage, taxes, slippage, gaps, liquidity and lot-size constraints. Position size is a ceiling by the chosen risk budget, not a recommendation.")
else:
    st.info("Risk planner needs at least 14 usable candles to estimate ATR.")

st.subheader("Recent price and trend")
show_cols=["Date","Open","High","Low","Close","Volume","MA10","MA21","MA50","EMA100","EMA200","RSI14","MACD","MACD_Signal","StochRSI_K","StochRSI_D","VolumeRatio","ATR14"]
available=[c for c in show_cols if c in df.columns]
st.line_chart(df.set_index("Date")[["Close","MA10","MA21","MA50","EMA100","EMA200"]].tail(150),width="stretch")
st.dataframe(df[available].tail(30).round(2),hide_index=True,width="stretch")
csv=df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download analysed candles CSV",csv,file_name=f"{symbol.lower()}_supervisor_analysis.csv",mime="text/csv")

with st.expander("CSV format and data quality"):
    st.write("Use one row per candle, oldest to newest. Required columns: Open, High, Low, Close, Volume. Include Date or Datetime if available. For EMA200, upload at least 200 candles; 250–300+ is preferable.")
    st.write("Do not upload account statements, API keys, passwords, or personal financial credentials.")
