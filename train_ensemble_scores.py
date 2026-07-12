import os
import sys
import pandas as pd

from src.backtest_dataset import build_walk_forward_dataset
from src.ml_scoring import train_all_models, score_stocks_ensemble

print("===================================")
print("   ENSEMBLE STOCK SCORING (ML)")
print("===================================\n")

universe_name = input("Universe (must already have been run through main.py): ").strip().lower()

data_folder = f"data/{universe_name}"
behaviour_dataset = f"features/{universe_name}_behaviour_dataset.csv"
communities_file = f"communities/{universe_name}_communities.csv"

if not os.path.exists(data_folder):
    print(f"{data_folder} not found -- run main.py for this universe first.")
    sys.exit(1)

if not os.path.exists(behaviour_dataset) or not os.path.exists(communities_file):
    print("Missing behaviour dataset or communities file -- run main.py for this universe first.")
    sys.exit(1)

# use the symbol map if download_data.py wrote one, so sanitised tickers (CON.DE, AV/.L, etc) resolve correctly
symbol_map_path = os.path.join(data_folder, "_symbol_filename_map.csv")
if os.path.exists(symbol_map_path):
    symbols = pd.read_csv(symbol_map_path)["Symbol"].tolist()
else:
    symbols = [f.replace(".csv", "") for f in os.listdir(data_folder) if f.endswith(".csv")]

start_date = input("Walk-forward training start date (YYYY-MM-DD): ").strip()
end_date = input("Walk-forward training end date (YYYY-MM-DD): ").strip()

print("\nBuilding walk-forward training dataset (this recomputes features at each past rebalance date)...")
training_data = build_walk_forward_dataset(symbols, data_folder, start_date, end_date)
print(f"Built {len(training_data)} (feature, forward-return) training rows.\n")

if training_data.empty:
    print("No training rows produced -- date range is likely too short for the feature lookback windows.")
    sys.exit(1)

print("Training Decision Tree, Random Forest, Gradient Boosting...")
models = train_all_models(training_data)
print("Done.\n")

output_file = f"baskets/stock_scores/{universe_name}_stock_scores_ensemble.csv"
print("Scoring stocks: blending hardcoded weights + all three models...")
scored_df = score_stocks_ensemble(behaviour_dataset, communities_file, models, output_file)

print(f"\nSaved -> {output_file}")
print("\nTop-scored stocks per community:")
print(scored_df[["Community", "Symbol", "Company", "Stock_Score"]].groupby("Community").head(3).to_string(index=False))
