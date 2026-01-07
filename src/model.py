import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os
from xgboost import XGBClassifier
from src.features import add_lag_and_rolling_features



def generate_labels(df: pd.DataFrame, ticker: str, horizon: int = 5, stop_loss: float = 0.02, take_profit: float = 0.02) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = 1  # Default to Hold

    for i in range(len(df) - horizon):
        entry = df[f'Close_{ticker}'].iloc[i]
        future_highs = df[f'High_{ticker}'].iloc[i+1:i+1+horizon]
        future_lows = df[f'Low_{ticker}'].iloc[i+1:i+1+horizon]

        tp_level = entry * (1 + take_profit)
        sl_level = entry * (1 - stop_loss)

        for high, low in zip(future_highs, future_lows):
            # If both TP and SL occur in the same future bar we cannot know which occurred first
            # mark it as Hold (ambiguous) to avoid mislabeling from OHLC bars.
            if (high >= tp_level) and (low <= sl_level):
                # ambiguous: leave as Hold (1)
                break
            elif high >= tp_level:
                df.at[df.index[i], 'Signal'] = 2  # Buy
                break
            elif low <= sl_level:
                df.at[df.index[i], 'Signal'] = 0  # Sell
                break
        # If neither, remain Hold

    df.dropna(subset=['Signal'], inplace=True)
    df['Signal'] = df['Signal'].astype(int)

    print("Label distribution (Buy=2, Hold=1, Sell=0):")
    print(df['Signal'].value_counts(normalize=True))

    return df







def train_model(df: pd.DataFrame, ticker):
    # Generate signal
    df = generate_labels(df, ticker)
    if df['Signal'].nunique() < 2:
        raise ValueError("❌ Not enough label diversity — try increasing the horizon or adjusting thresholds.")


    
    base_features = [
        'rsi', 'stoch_k', 'stoch_d', 'williams_r',
        'bollinger_m', 'bollinger_h', 'bollinger_l', 'atr',
        'obv', 'mfi',
        'macd', 'macd_signal', 'macd_diff',
        'ema_20', 'sma_20', 'adx'
    ]
    
    # Add lag and rolling features
    df = add_lag_and_rolling_features(df, base_features)

    # Final feature list
    features = df.columns[df.columns.str.startswith(tuple(base_features))].tolist()
    if 'Time' in df.columns:
        df['Time'] = df['Time'].apply(lambda t: t.hour + t.minute / 60)
        features.append('Time')
    
    X = df[features]
    y = df['Signal']
    # Chronological split (no shuffling) to avoid lookahead/data leakage in time-series
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split].copy(), X.iloc[split:].copy()
    y_train, y_test = y.iloc[:split].copy(), y.iloc[split:].copy()

    
    smote = SMOTE(random_state=42)
    

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Model
    model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test)
    
    

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Sell", "Hold", "Buy"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return df, model, features

def predict(model, df: pd.DataFrame, features: list):
    return model.predict(df[features])
