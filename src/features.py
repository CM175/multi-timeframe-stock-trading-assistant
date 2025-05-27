import pandas as pd
import ta

def add_lag_and_rolling_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_roll5'] = df[col].rolling(window=5).mean()
    df.dropna(inplace=True)
    return df

def add_technical_indicators(df: pd.DataFrame, ticker) -> pd.DataFrame:
    df = df.copy()

    # Momentum
    df['rsi'] = ta.momentum.RSIIndicator(close=df[f'Close_{ticker}']).rsi()
    df['stoch_k'] = ta.momentum.StochasticOscillator(
        high=df[f'High_{ticker}'], low=df[f'Low_{ticker}'], close=df[f'Close_{ticker}']
    ).stoch()
    df['stoch_d'] = ta.momentum.StochasticOscillator(
        high=df[f'High_{ticker}'], low=df[f'Low_{ticker}'], close=df[f'Close_{ticker}']
    ).stoch_signal()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(
        high=df[f'High_{ticker}'], low=df[f'Low_{ticker}'], close=df[f'Close_{ticker}']
    ).williams_r()

    # Volatility
    bb = ta.volatility.BollingerBands(close=df[f'Close_{ticker}'])
    df['bollinger_m'] = bb.bollinger_mavg()
    df['bollinger_h'] = bb.bollinger_hband()
    df['bollinger_l'] = bb.bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(
        high=df[f'High_{ticker}'], low=df[f'Low_{ticker}'], close=df[f'Close_{ticker}']
    ).average_true_range()

    # Volume
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(
        close=df[f'Close_{ticker}'], volume=df[f'Volume_{ticker}']
    ).on_balance_volume()
    df['mfi'] = ta.volume.MFIIndicator(
        high=df[f'High_{ticker}'], low=df[f'Low_{ticker}'], close=df[f'Close_{ticker}'], volume=df[f'Volume_{ticker}']
    ).money_flow_index()

    # Trend
    macd = ta.trend.MACD(close=df[f'Close_{ticker}'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    df['ema_20'] = ta.trend.EMAIndicator(close=df[f'Close_{ticker}'], window=20).ema_indicator()
    df['sma_20'] = ta.trend.SMAIndicator(close=df[f'Close_{ticker}'], window=20).sma_indicator()
    df['adx'] = ta.trend.ADXIndicator(
        high=df[f'High_{ticker}'], low=df[f'Low_{ticker}'], close=df[f'Close_{ticker}']
    ).adx()

    # Drop rows with NaNs from indicator calculations
    df.dropna(inplace=True)
    return df
