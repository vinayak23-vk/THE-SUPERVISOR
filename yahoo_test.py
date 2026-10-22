
import streamlit as st
import yfinance as yf

st.title("Yahoo Finance Connection Test")

symbol = st.text_input("Yahoo Finance symbol", "BELRISE.NS")

if st.button("Fetch daily candles"):
    with st.spinner("Requesting market data..."):
        try:
            df = yf.Ticker(symbol.strip()).history(
                period="1mo",
                interval="1d",
                auto_adjust=False
            )

            if df.empty:
                st.error("No candles returned. Check the symbol or Yahoo Finance availability.")
            else:
                st.success(f"Connection returned {len(df)} daily candles.")
                st.write("Latest candle")
                st.dataframe(df.tail(1))
                st.write("Latest timestamp:", df.index[-1])
                st.dataframe(df.tail(10))
        except Exception as e:
            st.error(f"Request failed: {type(e).__name__}: {e}")
