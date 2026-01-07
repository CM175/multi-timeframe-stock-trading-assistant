import yfinance as yf
from src.features import add_technical_indicators, add_lag_and_rolling_features
from src.model import train_model, predict
from src.utils import fetch_data
import pandas as pd
import re

# --- 1. Download data ---
ticker = "TSLA"  # or NVDA, AMD, COIN, etc.

# Choose interval and period here so downstream code can reuse the `interval` variable
interval = "15m"
df = fetch_data(ticker, interval=interval, period="1mo")
if re.match(r'.+m', interval):
        # Ensure we have a Datetime column and convenient Date/Time columns for minute intervals
        if 'Datetime' in df.columns:
                df["Datetime"] = pd.to_datetime(df["Datetime"]) 
        else:
                # attempt to coerce the first column if it looks like datetime
                first = df.columns[0]
                try:
                        df[first] = pd.to_datetime(df[first])
                        df.rename(columns={first: 'Datetime'}, inplace=True)
                except Exception:
                        pass

        if 'Datetime' in df.columns:
                df["Date"] = df["Datetime"].dt.date
                df["Time"] = df["Datetime"].dt.time
                cols = ["Datetime"] + [col for col in df.columns if col != "Datetime"]
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
