import yfinance as yf
from src.features import add_technical_indicators
from src.model import train_model, predict, add_lag_and_rolling_features
from src.utils import fetch_data
import pandas as pd

# --- 1. Download data ---
ticker = "TSLA"  # or NVDA, AMD, COIN, etc.

df = fetch_data(ticker, interval="5m", period="1mo")

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

# --- 6. Display most recent predictions ---
print("\nLatest predictions:")
print(df[[f"Close_{ticker}", "Prediction"]].tail(10))
