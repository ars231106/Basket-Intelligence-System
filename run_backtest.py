import os
import sys
import pandas as pd

from src.backtest_dataset import build_walk_forward_dataset
from src.ml_scoring import train_all_models
from src.backtest_engine import run_backtest, run_benchmark, compute_summary_stats

print("===================================")
print("   WALK-FORWARD BACKTEST")
print("===================================\n")

universe_name = input("Universe (must already have been run through main.py): ").strip().lower()

data_folder = f"data/{universe_name}"
communities_file = f"communities/{universe_name}_communities.csv"

if not os.path.exists(data_folder) or not os.path.exists(communities_file):
    print("Missing data or communities file -- run main.py for this universe first.")
    sys.exit(1)

start_date = input("Backtest start date (YYYY-MM-DD): ").strip()
end_date = input("Backtest end date (YYYY-MM-DD): ").strip()
k_baskets = int(input("Number of top baskets to hold each rebalance: ").strip())

print("\nScoring Method: ")
print("1. Hardcoded Weights only")
print("2. ML Ensemble only")
print("3. Compare both")
scoring_choice = input("Enter a choice: ").strip()

print("\nAllocation Strategy: ")
print("1. Equal Weight")
print("2. Score Weighted")
print("3. Inverse Volatility")
print("4. Markowitz Optimisation")
print("5. Risk Parity")
print("6. Black Litterman")
print("7. All strategies")
strategy_choice = input("Enter a choice: ").strip()

strategy_map = {"1": "equal_weighted", "2": "score_weighted", "3": "inverse_volatility",
               "4": "Markowitz_Optimisation", "5": "Risk_Parity", "6": "Black_Litterman"}

strategies = list(strategy_map.values()) if strategy_choice == "7" else [strategy_map[strategy_choice]]
scoring_methods = ["hardcoded", "ensemble"] if scoring_choice == "3" else (
    ["ensemble"] if scoring_choice == "2" else ["hardcoded"])

models = None
if "ensemble" in scoring_methods:
    symbol_map_path = os.path.join(data_folder, "_symbol_filename_map.csv")
    if os.path.exists(symbol_map_path):
        ml_symbols = pd.read_csv(symbol_map_path)["Symbol"].tolist()
    else:
        ml_symbols = pd.read_csv(communities_file)["Symbol"].unique().tolist()

    train_start = input("\nML training window start date (should end before backtest start): ").strip()
    train_end = input("ML training window end date (recommend <= backtest start date): ").strip()

    print("\nBuilding walk-forward training dataset...")
    training_data = build_walk_forward_dataset(ml_symbols, data_folder, train_start, train_end)
    print(f"Built {len(training_data)} training rows.")
    print("Training Decision Tree, Random Forest, Gradient Boosting...")
    models = train_all_models(training_data)

print("\nRunning equal-weight universe benchmark...")
benchmark = run_benchmark(universe_name, start_date, end_date)
benchmark_stats = compute_summary_stats(benchmark)

os.makedirs("backtests", exist_ok=True)
benchmark.to_csv(f"backtests/{universe_name}_benchmark_equity_curve.csv", index=False)

results = []
for scoring_method in scoring_methods:
    for strategy in strategies:
        label = f"{scoring_method}_{strategy}"
        print(f"\nRunning backtest: scoring={scoring_method}, allocation={strategy}...")
        equity_curve = run_backtest(universe_name, start_date, end_date, scoring_method, strategy,
                                    k_baskets=k_baskets, models=models)
        stats = compute_summary_stats(equity_curve)
        stats["Scoring_Method"] = scoring_method
        stats["Allocation_Strategy"] = strategy
        results.append(stats)

        equity_curve.to_csv(f"backtests/{universe_name}_{label}_equity_curve.csv", index=False)

summary_df = pd.DataFrame(results)
benchmark_stats["Scoring_Method"] = "benchmark"
benchmark_stats["Allocation_Strategy"] = "equal_weight_universe"
summary_df = pd.concat([summary_df, pd.DataFrame([benchmark_stats])], ignore_index=True)

summary_file = f"backtests/{universe_name}_backtest_summary.csv"
summary_df.to_csv(summary_file, index=False)

print(f"\nSaved summary -> {summary_file}")
print("\n" + summary_df.to_string(index=False))
