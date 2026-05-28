import yfinance as yf
from features import add_technical_indicators
from model import train_model, predict, add_lag_and_rolling_features
from utils import fetch_data
import pandas as pd
import re

# --- 1. Download data ---
ticker = "TSLA"  # or NVDA, AMD, COIN, etc.

interval = "60m" 
df = fetch_data(ticker, interval=interval, period="2y")
if re.match(r'\d+m$', interval):
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df["Date"] = df["Datetime"].dt.date
        df["Time"] = df["Datetime"].dt.time
        cols = ["Date", "Time"] + [col for col in df.columns if col not in ["Date", "Time"]]
        df = df[cols]
        cols = ["Datetime"] + [col for col in df.columns if col not in ["Datetime"]]
        df = df[cols]

df.dropna(inplace=True)
# --- 2. Feature engineering ---
df = add_technical_indicators(df, ticker)

# Drop rows with NaN values caused by lag/rolling ops
df.dropna(inplace=True)

# --- 3. Train model and return features used ---
df, model, features = train_model(df, ticker)
# --- 4. Predict using same features ---
predictions = predict(model, df, features)
# --- 5. Add predictions to DataFrame (optional) ---
df["Prediction"] = predictions
print(df.head())
# --- 6. Display most recent predictions ---
print("\nLatest predictions:")
print(df[[f"Close_{ticker}", "Prediction"]].tail(10))
