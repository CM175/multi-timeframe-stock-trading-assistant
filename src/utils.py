import yfinance as yf
import pandas as pd

def fetch_data(ticker, interval="1d", period="1y"):
    df = yf.download(ticker, interval=interval, period=period)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} at interval={interval}, period={period}")
    # Flatten multi-index columns
    df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]

    df.reset_index(inplace=True)
    return df


def add_return_label(df: pd.DataFrame,ticker,  threshold=0.01) -> pd.DataFrame:
    df['Return'] = df[f'Close_{ticker}'].pct_change().shift(-1)
    df['Signal'] = df['Return'].apply(lambda r: 1 if r > threshold else (-1 if r < -threshold else 0))
    return df
