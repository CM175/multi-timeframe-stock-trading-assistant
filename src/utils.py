import yfinance as yf
import pandas as pd

def fetch_data(ticker, interval, period):
    df = yf.download(ticker, interval=interval, period=period)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} at interval={interval}, period={period}")
    # Flatten multi-index columns
    df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]

    df.reset_index(inplace=True)
    return df



