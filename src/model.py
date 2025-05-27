import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
from xgboost import XGBClassifier
from src.features import add_lag_and_rolling_features



def generate_labels(df: pd.DataFrame,ticker,  horizon=5) -> pd.DataFrame:
    df = df.copy()
    df['future_return'] = df[f'Close_{ticker}'].pct_change(horizon).shift(-horizon)

    low = df['future_return'].quantile(0.33)
    high = df['future_return'].quantile(0.66)

    df['Signal'] = pd.cut(
        df['future_return'],
        bins=[-float('inf'), low, high, float('inf')],
        labels=[0, 1, 2]
    )

    df.dropna(subset=['Signal'], inplace=True)
    df['Signal'] = df['Signal'].astype(int)
    print("Label distribution:")
    print(df['Signal'].value_counts(normalize=True))

    return df



def train_model(df: pd.DataFrame, ticker):
    # Generate signal
    df = generate_labels(df, ticker)
    

    
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
    
    X = df[features]
    y = df['Signal']
    # Split and balance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True)

    
    smote = SMOTE()
    

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Model
    model = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test)
    
    

    joblib.dump(model, 'models/model.pkl')
    return df, model, features

def predict(model, df: pd.DataFrame, features: list):
    return model.predict(df[features])
