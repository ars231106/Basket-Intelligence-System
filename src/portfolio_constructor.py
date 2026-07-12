import os
import pandas as pd

def load_top_baskets(basket_scoring_file):
    return pd.read_csv(basket_scoring_file)

def select_top_baskets(rankings_df, k):
    return rankings_df.sort_values("Rank").head(k).reset_index(drop=True)

def save_portfolio_baskets(portfolio_df, universe_name, save_dir="./portfolio/selected"):
    os.makedirs(save_dir, exist_ok=True)

    output_file = os.path.join(save_dir, f"{universe_name}_selected_baskets.csv")
    portfolio_df.to_csv(output_file, index=False)

    print(f"Selected baskets saved to {output_file}")