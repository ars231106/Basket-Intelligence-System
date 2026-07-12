import os
import pandas as pd

def load_selected_baskets(selected_baskets_file):
    return pd.read_csv(selected_baskets_file)

def equal_weight(df, capital):
    allocation = df.copy()
    n = len(allocation)

    allocation["Weight"] = 1/n
    allocation["Capital_Distribution"] = allocation["Weight"] * capital

    return allocation

def score_weighted(df, capital):
    allocation = df.copy()
    total_score = allocation["Basket_Score"].sum()

    allocation["Weight"] = allocation["Basket_Score"] / total_score
    allocation["Capital_Distribution"] = allocation["Weight"] * capital

    return allocation

def inverse_volatility(df, capital):
    allocation = df.copy()
    # +1e-9 guards against a basket with ~zero volatility producing a
    # division-by-zero that collapses every other basket's weight to 0
    # (confirmed happening in a saved nasdaq100 allocation output).
    inverse_vol = 1 / (allocation["Average_Volatility"] + 1e-9)

    allocation["Weight"] = inverse_vol / inverse_vol.sum()
    allocation["Capital_Distribution"] = allocation["Weight"] * capital

    return allocation

def allocate_portfolio(df, strategy, capital):
    strategy = strategy.lower()

    if strategy == "equal_weighted":
        return equal_weight(df, capital)

    elif strategy == "score_weighted":
        return score_weighted(df, capital)

    elif strategy == "inverse_volatility":
        return inverse_volatility(df, capital)

    else:
        raise ValueError("Invalid Heuristic (Rule-Based) Allocation Strategy.\n Choose an Optimisation-Based Allocation Strategy")

def save_allocations(allocation_df, universe_name, save_dir = "./portfolio/allocations"):
    os.makedirs(save_dir,exist_ok=True)
    output_file = os.path.join(save_dir, f"{universe_name}_basket_allocation.csv")

    allocation_df.to_csv(output_file, index=False)

    print(f"\nPortfolio allocation saved to:\n{output_file}")
