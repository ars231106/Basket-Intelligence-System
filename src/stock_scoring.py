import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

WEIGHTS = {
    "Sharpe_Ratio": 0.25,
    "Mean_Return": 0.25,
    "Momentum": 0.20,
    "Volume_Growth": 0.15,
    "Average_Volume": 0.10,
    "Volatility": -0.15,
    "ATR": -0.10,
    "RSI_Score" : 0.02,
    "SMA20_Score" : 0.01,
    "SMA50_Score" : 0.01 
}

def score_stocks(behaviour_dataset, communities_file, output_file):
    features = pd.read_csv(behaviour_dataset)
    communities = pd.read_csv(communities_file)

    df = communities.merge(features, on="Symbol")

    df["RSI_Score"] = 1 - abs(df["RSI"] - 50) / 50
    df["SMA20_Score"] = 1 - abs(df["Distance_SMA20"])
    df["SMA50_Score"] = 1 - abs(df["Distance_SMA50"])

    df["RSI_Score"] = df["RSI_Score"].clip(0,1)
    df["SMA20_Score"] = df["SMA20_Score"].clip(0,1)
    df["SMA50_Score"] = df["SMA50_Score"].clip(0,1)
 
    positive_features = ["Mean_Return", "Momentum", "Average_Volume", "Volume_Growth", "Sharpe_Ratio"]
    negative_features = ["Volatility", "ATR"]

    scored_rows = []

    for community, group in df.groupby("Community"):
        group = group.copy()

        if len(group) < 2:
            # A single-stock community has no peers to scale relative to.
            # sklearn's MinMaxScaler already avoids a divide-by-zero here
            # (a zero-range feature maps to 0), but that would silently
            # give every feature a scaled value of 0 regardless of the
            # stock's actual performance -- assign a neutral score instead
            # of letting that happen implicitly.
            group["Stock_Score"] = 0.5
            scored_rows.append(group)
            print(f"Processed Community {community} ({len(group)} stock) -- singleton, neutral score assigned")
            continue

        positive_scaler = MinMaxScaler()
        negative_scaler = MinMaxScaler()

        group[positive_features] = positive_scaler.fit_transform(group[positive_features])
        group[negative_features] = negative_scaler.fit_transform(group[negative_features])

        group["Stock_Score"] = 0

        for feature, weight in WEIGHTS.items():
            group["Stock_Score"] += weight * group[feature]

        scored_rows.append(group)

        print(f"Processed Community {community} ({len(group)} stocks)")

    scored_df = pd.concat(scored_rows, ignore_index=True)
    scored_df = scored_df.sort_values(["Community", "Stock_Score"], ascending=[True, False])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    scored_df.to_csv(output_file, index=False)
    return scored_df
