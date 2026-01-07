import yfinance as yf
import pandas as pd

def fetch_data(ticker, interval, period):
    df = yf.download(ticker, interval=interval, period=period)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} at interval={interval}, period={period}")
    # Flatten multi-index columns safely
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            new_cols.append(col[0] if col[1] == '' else f"{col[0]}_{col[1]}")
        else:
            new_cols.append(col)
    df.columns = new_cols

    df.reset_index(inplace=True)
    # Ensure the first column is a consistent datetime column name
    first_col = df.columns[0]
    try:
        if first_col.lower() not in ('datetime', 'date') and pd.api.types.is_datetime64_any_dtype(df[first_col]):
            df.rename(columns={first_col: 'Datetime'}, inplace=True)
    except Exception:
        # If detection fails, leave as-is; downstream code should handle missing Datetime
        pass

    return df



