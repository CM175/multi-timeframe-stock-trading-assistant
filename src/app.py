import streamlit as st 
import pandas as pd
import joblib
from src.utils import fetch_data
from src.features import add_technical_indicators, add_lag_and_rolling_features
from src.model import predict
from constants import valid_combinations
import re

st.set_page_config(layout="wide")
st.title("📈 Multi-Timeframe Stock Trading Assistant")

ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
interval = st.sidebar.selectbox("Select Interval", list(valid_combinations.keys()))

# --- Sidebar: Period options update based on interval
period_options = valid_combinations[interval]
period = st.sidebar.selectbox("Select Period", period_options)

st.sidebar.markdown(f"🧠 You selected `{interval}` interval for `{period}` period.")

if st.button("Run Analysis"):
    base_features = [
        'rsi', 'stoch_k', 'stoch_d', 'williams_r',
        'bollinger_m', 'bollinger_h', 'bollinger_l', 'atr',
        'obv', 'mfi',
        'macd', 'macd_signal', 'macd_diff',
        'ema_20', 'sma_20', 'adx'
    ]
    # ✅ Load trained model and matching features
    # --- 1. Fetch and clean data    
    df = fetch_data(ticker, interval=interval, period=period)
    df.dropna(inplace=True)
    
    if re.match('.+m', interval):
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df["Date"] = df["Datetime"].dt.date
        df["Time"] = df["Datetime"].dt.time
        cols = ["Date", "Time"] + [col for col in df.columns if col not in ["Date", "Time"]]
        df = df[cols]
        cols = ["Datetime"] + [col for col in df.columns if col not in ["Datetime"]]
        df = df[cols]

    # --- 2. Feature engineering
    print(df.head())
    df = add_technical_indicators(df, ticker)
    df = add_lag_and_rolling_features(df, base_features)
    model = joblib.load("models/model.pkl")
    

    
    features = df.columns[df.columns.str.startswith(tuple(base_features))].tolist()
    df.dropna(inplace=True)

    # --- 3. Predict
    if df.empty:
        st.warning("⚠️ No data left after feature engineering. Cannot make predictions.")
    else:
        df['Prediction'] = predict(model, df, features)

        # ✅ Show last prediction details
        latest = df.iloc[-1]
        prediction_map = {0: "🔴 Sell", 1: "🟡 Hold", 2: "🟢 Buy"}
        pred = int(latest["Prediction"])
        signal = prediction_map.get(pred, "❓ Unknown")

        st.subheader("📊 Latest Prediction")
        st.markdown(f"**Date:** {latest[0]}")
        st.markdown(f"**Close Price:** ${latest[f'Close_{ticker}']:.2f}")
        st.markdown(f"**Prediction:** {signal}")
        print(pred)
        if pred == 2:  # Only show TP/SL suggestion for Buy signals
            atr = latest["atr"]
            current_price = latest[f"Close_{ticker}"]

            suggested_tp = current_price + 1.5 * atr
            suggested_sl = current_price - 1.0 * atr

            st.subheader("🎯 Suggested Strategy")
            st.markdown(f"**Take Profit:** ${suggested_tp:.2f}  (1.5× ATR)")
            st.markdown(f"**Stop Loss:** ${suggested_sl:.2f}  (1.0× ATR)")

            # --- Estimate bars to TP
            lookahead = 30
            future_prices = df[f"High_{ticker}"].iloc[-lookahead:]
            time_to_hit = None
            for i, price in enumerate(future_prices):
                if price >= suggested_tp:
                    time_to_hit = i + 1
                    break

            if time_to_hit:
                st.markdown(f"📈 Estimated time to hit TP: **{time_to_hit} bars**")
            else:
                st.markdown("⌛ Take Profit not hit in next 30 bars.")
        elif pred == 0:  # Only show TP/SL suggestion for Sell signals
            atr = latest["atr"]
            current_price = latest[f"Close_{ticker}"]

            suggested_tp = current_price - 1.5 * atr
            suggested_sl = current_price + 1.0 * atr

            st.subheader("🎯 Suggested Strategy")
            st.markdown(f"**Take Profit:** ${suggested_tp:.2f}  (1.5× ATR)")
            st.markdown(f"**Stop Loss:** ${suggested_sl:.2f}  (1.0× ATR)")

            # --- Estimate bars to TP
            lookahead = 30
            future_prices = df[f"High_{ticker}"].iloc[-lookahead:]
            time_to_hit = None
            for i, price in enumerate(future_prices):
                if price >= suggested_tp:
                    time_to_hit = i + 1
                    break

            if time_to_hit:
                st.markdown(f"📈 Estimated time to hit TP: **{time_to_hit} bars**")
            else:
                st.markdown("⌛ Take Profit not hit in next 30 bars.")
        


        # --- Optional: Show confidence
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(pd.DataFrame([latest[features]]))[0]
            st.write(f"Confidence: {prob[pred] * 100:.2f}%")
            st.progress(float(prob[pred]))

        # --- Chart
        st.subheader("📉 Close Price Chart")
        st.line_chart(df.set_index(df.columns[0])[[f"Close_{ticker}"]])
