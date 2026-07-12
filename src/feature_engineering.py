import numpy as np
import pandas as pd

def calculate_mean_return(df):
    returns = df["Close"].pct_change().mean()
    return returns

def calculate_volatility(df):
    returns = df["Close"].pct_change().std()
    return returns

def calculate_momentum(df, period=20):
    momentum = df["Close"].pct_change(period)
    return momentum.tail(period).mean()

def calculate_avg_vol(df):
    return df["Volume"].mean()

def calculate_volume_growth(df, window = 20):
    recent = df["Volume"].tail(window).mean()
    previous = df["Volume"].iloc[-2*window:-window].mean()
    return recent / (previous + 1e-9)

def calculate_distance_sma20(df):
    sma20 = df["Close"].rolling(20).mean()
    return((df["Close"].iloc[-1] - sma20.iloc[-1]) / sma20.iloc[-1]) 

def calculate_distance_sma50(df):
    sma50 = df["Close"].rolling(50).mean()
    return((df["Close"].iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1])

def calculate_sharpe_ratio(df):
    returns = df["Close"].pct_change().dropna()
    return returns.mean() / (returns.std() + 1e-9)

def calculate_rsi(df, period = 14):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_atr(df, period = 14):
    today_range = df["High"] - df["Low"]
    gap_up = abs(df["High"] - df["Close"].shift())
    gap_down = abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([today_range, gap_up, gap_down], axis = 1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.iloc[-1]

FEATURES = {
    "Mean_Return": calculate_mean_return,
    "Volatility": calculate_volatility,
    "Momentum": calculate_momentum,
    "RSI": calculate_rsi,
    "ATR": calculate_atr,
    "Average_Volume": calculate_avg_vol,
    "Volume_Growth": calculate_volume_growth,
    "Distance_SMA20": calculate_distance_sma20,
    "Distance_SMA50": calculate_distance_sma50,
    "Sharpe_Ratio": calculate_sharpe_ratio,
}

def compute_features(df):
    features = {}
    for name, function in FEATURES.items():
        features[name] = function(df)

    return features
