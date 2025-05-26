# 📈 Multi-Timeframe Stock Trading Assistant

A smart, interactive Streamlit app that predicts stock price movements across multiple timeframes and generates actionable buy/sell signals using technical indicators and machine learning.

## 🔧 Features

- Pulls real-time and historical stock data (daily, hourly, minute)
- Computes multiple technical indicators
- Predicts short-term stock direction with ML
- Visualizes model confidence and feature importance
- SHAP-based explainability (why the model made a decision)
- Interactive dashboard built with Streamlit

## 🚀 Run Locally

```bash
git clone https://github.com/yourusername/multi-timeframe-stock-trading-assistant.git
cd multi-timeframe-stock-trading-assistant
pip install -r requirements.txt
streamlit run src/app.py
