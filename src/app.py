import streamlit as st
import pandas as pd
import joblib
from src.utils import fetch_data, add_return_label
from src.features import add_technical_indicators
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
    features = ['rsi', 'macd', 'ema_20', 'sma_20', 'volume_sma'] 

    df = fetch_data(ticker, interval=interval, period=period)
    df = add_return_label(df, ticker)
    df = add_technical_indicators(df, ticker)
    model = joblib.load("models/model.pkl")
    if df.empty:
        print("⚠️ No data left after feature engineering. Cannot make predictions.")
    else:
        df['Prediction'] = predict(model, df)


    latest = df.iloc[-1]
    print(latest)   
    prediction_map = {
        -1: "🔴 Sell",
        0: "🟡 Hold",
        1: "🟢 Buy"
    }

    st.subheader("📊 Latest Prediction")
    st.markdown(f"**Date:** {latest[0].date()}")
    st.markdown(f"**Close Price:** ${latest[f'Close_{ticker}']:.2f}")
        
    pred = int(latest["Prediction"])
    signal = prediction_map.get(pred, "❓ Unknown")
        
    st.markdown(f"**Prediction:** {signal}")

    # Optional: show model confidence (if using predict_proba)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba([latest[features]])[0]
        st.write("Confidence:")
        st.progress(prob[pred])  # bar for confidence
        st.json({label: f"{p:.2%}" for label, p in zip(model.classes_, prob)})
       

    st.write(f"Prediction: **{'Buy' if latest['Prediction'] == 1 else 'Sell' if latest['Prediction'] == -1 else 'Hold'}**")

    st.subheader("Chart")
    st.line_chart(df.set_index(df.columns[0])[[f"Close_{ticker}"]])
