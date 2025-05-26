from src.utils import fetch_data, add_return_label
from src.features import add_technical_indicators
from src.model import train_model
import yfinance as yf

df = yf.download("AAPL", period="3mo", interval="15m")
print(f"Shape: {df.shape}")
print(df.head())

