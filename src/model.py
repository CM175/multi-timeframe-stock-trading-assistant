import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_model(df: pd.DataFrame, ticker):
    features = ['rsi', 'macd', 'ema_20', 'sma_20', 'volume_sma']
    X = df[features]
    y = df['Signal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print(classification_report(y_test, clf.predict(X_test)))
    joblib.dump(clf, 'models/model.pkl')
    return clf

def predict(model, df: pd.DataFrame):
    features = ['rsi', 'macd', 'ema_20', 'sma_20', 'volume_sma']
    return model.predict(df[features])
