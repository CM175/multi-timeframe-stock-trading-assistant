# 📈 Multi-Timeframe Stock Trading Assistant

A smart, interactive Streamlit app that predicts stock price movements across multiple timeframes and generates actionable buy/sell signals using technical indicators and machine learning.

## 🔧 Features

- Pulls real-time and historical stock data (daily, hourly, minute)
- Computes multiple technical indicators
- Predicts short-term stock direction with ML
- Visualizes model confidence and feature importance
- Interactive dashboard built with Streamlit

## 🛠 Tech Stack

- **Backend:** Python, yfinance, TA-Lib, scikit-learn
- **Machine Learning:** Random Forest, joblib
- **Visualization:** Streamlit

## 🚀 Run Locally

```bash
git clone https://github.com/CM175/multi-timeframe-stock-trading-assistant.git
cd multi-timeframe-stock-trading-assistant
pip install -r requirements.txt
python -m streamlit run src/app.py
