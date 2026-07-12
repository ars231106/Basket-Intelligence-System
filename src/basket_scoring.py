import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

WEIGHTS = {
    "Average_Sharpe_Ratio": 0.25,
    "Average_Return": 0.20,
    "Basket_Cohesion": 0.15,
    "Average_Stock_Score": 0.15,
    "Average_Momentum": 0.10,
    "Average_Volume_Growth": 0.05,
    "Average_Volume": 0.05,
    "Average_RSI_Score": 0.02,
    "Average_SMA20_Score": 0.015,
    "Average_SMA50_Score": 0.015,
    "Average_Volatility": -0.10,
    "Average_ATR": -0.05
}

def score_baskets(basket_metrics_file, output_file):

    df = pd.read_csv(basket_metrics_file)

    positive_features = [
        "Average_Return",
        "Average_Momentum",
        "Average_Volume",
        "Average_Volume_Growth",
        "Average_Sharpe_Ratio",
        "Average_Stock_Score",
        "Basket_Cohesion",
        "Average_RSI_Score",
        "Average_SMA20_Score",
        "Average_SMA50_Score"
    ]

    negative_features = [
        "Average_Volatility",
        "Average_ATR"
    ]

    scaled_df = df.copy()

    positive_scaler = MinMaxScaler()
    negative_scaler = MinMaxScaler()

    scaled_df[positive_features] = positive_scaler.fit_transform(scaled_df[positive_features])
    scaled_df[negative_features] = negative_scaler.fit_transform(scaled_df[negative_features])

    df["Basket_Score"] = 0

    for feature, weight in WEIGHTS.items():
        df["Basket_Score"] += weight * scaled_df[feature]

    
    df = df.sort_values("Basket_Score", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    return df
