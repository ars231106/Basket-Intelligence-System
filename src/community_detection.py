import community as community_louvain
from collections import defaultdict 
import pandas as pd
import os

def detect_communities(graph, random_state=42):
    # Louvain's local optimisation has a random tie-breaking component;
    # without a fixed random_state, re-running the same universe on the
    # same data can produce different community partitions (and therefore
    # different baskets) from one run to the next. Pinning it makes runs
    # reproducible.
    communities = community_louvain.best_partition(graph, weight="weight", random_state=random_state)
    return communities

def group_communities(communities):
    baskets = defaultdict(list)

    for stock, community in communities.items():
        baskets[community].append(stock)

    return dict(baskets)

def generate_community_statistics(dataset_file, baskets, output_file):
    df = pd.read_csv(dataset_file)

    rows =[]

    for community, stocks in baskets.items():
        community_df = df[df["Symbol"].isin(stocks)]

        row = {
            "Community" : community,
            "Members" : len(stocks),
            "Average Returns" : community_df["Mean_Return"].mean(),
            "Average Volatility" : community_df["Volatility"].mean(),
            "Average_Momentum": community_df["Momentum"].mean(),
            "Average_RSI": community_df["RSI"].mean(),
            "Average_ATR": community_df["ATR"].mean(),
            "Average_Volume": community_df["Average_Volume"].mean(),
            "Average_Volume_Growth": community_df["Volume_Growth"].mean(),
            "Average_Distance_SMA20": community_df["Distance_SMA20"].mean(),
            "Average_Distance_SMA50": community_df["Distance_SMA50"].mean(),
            "Average_Sharpe_Ratio": community_df["Sharpe_Ratio"].mean() 
        }

        rows.append(row)

    statistics = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    statistics.to_csv(output_file, index=False)
    return statistics