import pandas as pd
import ta  # pip install ta

def add_technical_indicators(df: pd.DataFrame, ticker) -> pd.DataFrame:
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(close=df[f'Close_{ticker}']).rsi()
    df['macd'] = ta.trend.MACD(close=df[f'Close_{ticker}']).macd()
    df['ema_20'] = ta.trend.EMAIndicator(close=df[f'Close_{ticker}'], window=20).ema_indicator()
    df['sma_20'] = ta.trend.SMAIndicator(close=df[f'Close_{ticker}'], window=20).sma_indicator()
    df['volume_sma'] = df[f'Volume_{ticker}'].rolling(20).mean()
    df.dropna(inplace=True)
    return df
