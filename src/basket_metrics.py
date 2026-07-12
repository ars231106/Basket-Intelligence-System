import os
import pandas as pd
import numpy as np

def generate_basket_metrics(basket_file, similarity_matrix_file, behaviour_dataset, output_file):
    baskets = pd.read_csv(basket_file)
    similarity_matrix = pd.read_csv(similarity_matrix_file, index_col=0)
    behaviour = pd.read_csv(behaviour_dataset)

    df = baskets.merge(behaviour, on="Symbol")

    df["RSI_Score"] = 1 - abs(df["RSI"] - 50) / 50
    df["SMA20_Score"] = 1 - abs(df["Distance_SMA20"])
    df["SMA50_Score"] = 1 - abs(df["Distance_SMA50"])

    df["RSI_Score"] = df["RSI_Score"].clip(0,1)
    df["SMA20_Score"] = df["SMA20_Score"].clip(0,1)
    df["SMA50_Score"] = df["SMA50_Score"].clip(0,1)

    rows = []
    for basket_id, group in df.groupby("Basket_ID"):
        symbols = group["Symbol"].tolist()
        members = len(group)

        avg_return = group["Mean_Return"].mean()
        avg_volatility = group["Volatility"].mean()
        avg_momentum = group["Momentum"].mean()
        avg_rsi = group["RSI_Score"].mean()
        avg_atr = group["ATR"].mean()
        avg_volume = group["Average_Volume"].mean()
        avg_volume_growth = group["Volume_Growth"].mean()
        avg_sma20 = group["SMA20_Score"].mean()
        avg_sma50 = group["SMA50_Score"].mean()
        avg_sharpe = group["Sharpe_Ratio"].mean()
        avg_stock_score = group["Stock_Score"].mean()

        pairwise_similarities = []

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                stock1 = symbols[i]
                stock2 = symbols[j]

                similarity_score = similarity_matrix.loc[stock1, stock2]
                pairwise_similarities.append(similarity_score)

        if pairwise_similarities:
            basket_cohesion = np.mean(pairwise_similarities)
        else:
            basket_cohesion = 1.0

        row = {
            "Basket_ID" : basket_id,
            "Community" : group["Community"].iloc[0],
            "Members" : members,

            "Average_Return": avg_return,
            "Average_Volatility": avg_volatility,
            "Average_Momentum": avg_momentum,
            "Average_RSI_Score": avg_rsi,
            "Average_ATR": avg_atr,
            "Average_Volume": avg_volume,
            "Average_Volume_Growth": avg_volume_growth,
            "Average_SMA20_Score": avg_sma20,
            "Average_SMA50_Score": avg_sma50,
            "Average_Sharpe_Ratio": avg_sharpe,

            "Average_Stock_Score": avg_stock_score,
            "Basket_Cohesion": basket_cohesion
        }

        rows.append(row)

    basket_metrics = pd.DataFrame(rows)
    basket_metrics = basket_metrics.sort_values("Average_Stock_Score", ascending=False)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    basket_metrics.to_csv(output_file, index=False)
    return basket_metrics
