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

def calculate_range_position_52w(df, window=252):
    """
    Where the latest close sits within its trailing ~52-week (252 trading
    day) high-low range, scaled to 0 (sitting right at the 52-week low)
    through 1 (sitting right at the 52-week high), with 0.5 meaning dead
    center. Used for within-basket allocation strategies that weight
    stocks by their position in their own recent trading range.

    Falls back to however much history is actually available (via
    .tail(), not a hard requirement of exactly 252 rows) so it behaves
    consistently with the other rolling-window features here rather than
    returning NaN for a stock with slightly under a year of history.
    """
    window_data = df["Close"].tail(window)
    low_52w = window_data.min()
    high_52w = window_data.max()
    if high_52w - low_52w < 1e-12:
        return 0.5
    return (df["Close"].iloc[-1] - low_52w) / (high_52w - low_52w)

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
    "Range_Position_52W": calculate_range_position_52w,
}

def compute_features(df):
    features = {}
    for name, function in FEATURES.items():
        features[name] = function(df)

    return features
