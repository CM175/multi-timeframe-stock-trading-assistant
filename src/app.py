import streamlit as st 
import pandas as pd
import joblib
from src.utils import fetch_data
from src.features import add_technical_indicators, add_lag_and_rolling_features
from src.model import predict
from constants import valid_combinations

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

    # --- 2. Feature engineering
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
        print(df.head(30))

        # ✅ Show last prediction details
        latest = df.iloc[-1]
        prediction_map = {0: "🔴 Sell", 1: "🟡 Hold", 2: "🟢 Buy"}
        pred = int(latest["Prediction"])
        signal = prediction_map.get(pred, "❓ Unknown")

        st.subheader("📊 Latest Prediction")
        st.markdown(f"**Date:** {latest[0].date()}")
        st.markdown(f"**Close Price:** ${latest[f'Close_{ticker}']:.2f}")
        st.markdown(f"**Prediction:** {signal}")

        # --- Optional: Show confidence
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(pd.DataFrame([latest[features]]))[0]
            st.write(f"Confidence: {prob[pred] * 100:.2f}%")
            st.progress(float(prob[pred]))

        # --- Debugging output
        st.subheader("🔍 Prediction Overview")
        st.write("Prediction distribution:")
        st.write(df["Prediction"].value_counts())

        # --- Chart
        st.subheader("📉 Close Price Chart")
        st.line_chart(df.set_index(df.columns[0])[[f"Close_{ticker}"]])
