import os
import shutil
import pandas as pd
import numpy as np

from src.download_data import download_stock_data
from src.universe_loader import load_universe
from src.build_dataset import build_dataset
from src.feature_scaling import scale_behaviour_dataset
from src.similarity_engine import compute_similarity_matrix
from src.graph_builder import build_similarity_graph
from src.community_detection import (detect_communities,group_communities, generate_community_statistics)
from src.visualisation import (plot_similarity_graph, plot_communities_graph)
from src.save_communities import save_communities
from src.stock_scoring import score_stocks
from src.backtest_dataset import build_walk_forward_dataset
from src.ml_scoring import train_all_models, score_stocks_ensemble, evaluate_model
from src.backtest_dataset import split_train_test
from src.basket_generator import generate_candidate_baskets
from src.basket_metrics import generate_basket_metrics
from src.basket_scoring import score_baskets
from src.portfolio_constructor import (load_top_baskets, select_top_baskets, save_portfolio_baskets)
from src.portfolio_allocator import (load_selected_baskets, allocate_portfolio,save_allocations)
from src.basket_returns import  (load_candidate_baskets,compute_basket_daily_returns, 
                                 compute_covariance_matrix,save_outputs, compute_expected_returns, 
                                 save_expected_returns)
from src.markowitz_optimiser import (negetive_sharpe_ratio_scipy_minimisation, optimise_portfolio, save_markovitz)
from src.risk_parity_optimiser import (optimize_risk_parity, save_risk_parity)
from src.backtest_engine import run_backtest, run_benchmark, compute_summary_stats
from src.intra_basket_allocator import basket_weights_to_stock_weights_v2
from src.capital_sizing import load_latest_prices, compute_minimum_viable_capital, check_affordability
from src.black_litterman import (compute_market_weights, compute_equilibrium_returns,
                                 generate_picking_matrix, generate_view_returns, compute_confidence_matrix, 
                                 compute_black_litterman_returns, save_black_litterman)
print("===================================")
print("         ATLASQUANT")
print("Basket Intelligence System")
print("===================================\n")

print("Available Universes:")
print("1. NIFTY50")
print("2. NIFTY100")
print("3. S&P500")
print("4. NASDAQ100")
print("5. DOWJONES")
print("6. DAX")
print("7. FTSE100")
print("8. HSI")
print("9. NIFTYMID150")
print("10. NIKKEI225")
print("11. KOSPI200")
print("12. NIFTYSMALL250")

choice = input("\nEnter Choice of Index (1-12): ")

if choice == "1":
    universe_name = "nifty50"

elif choice == "2":
    universe_name = "nifty100"

elif choice == "3":
    universe_name = "sp500"

elif choice == "4":
    universe_name = "nasdaq100"

elif choice == "5":
    universe_name = "dowjones"

elif choice == "6":
    universe_name = "dax"

elif choice == "7":
    universe_name = "ftse100"

elif choice == "8":
    universe_name = "hsi"

elif choice == "9":
    universe_name = "niftymid150"

elif choice == "10":
    universe_name = "nikkei225"

elif choice == "11":
    universe_name = "kospi200"

elif choice == "12":
    universe_name = "niftysmall250" 

else:
    print("Universe Currently Unavailable.")
    exit()

stocks = load_universe(universe_name)

print(f"\nLoaded {len(stocks)} stocks.")

start_date = input("\nStart Date (YYYY-MM-DD): ")
end_date = input("End Date (YYYY-MM-DD): ")

print("\n===================================")
print("DOWNLOADING MARKET DATA")
print("===================================\n")

save_dir = f"data/{universe_name}"

os.makedirs(save_dir, exist_ok=True)

for file in os.listdir(save_dir):
    if file.endswith(".csv"):
        os.remove(os.path.join(save_dir, file))

download_stock_data(stocks, start_date, end_date, save_dir=save_dir)

print("\n===================================")
print("BUILDING BEHAVIOUR DATASET")
print("===================================\n")

dataset = build_dataset(data_folder=save_dir, output_file=f"features/{universe_name}_behaviour_dataset.csv")

symbols, scaled_features = scale_behaviour_dataset(f"features/{universe_name}_behaviour_dataset.csv")

print("Features Scaled Successfully....")

similarity_matrix = compute_similarity_matrix(
    symbols,
    scaled_features
)

os.makedirs("graphs", exist_ok=True)

similarity_matrix.to_csv(f"graphs/{universe_name}_similarity_matrix.csv")
print(similarity_matrix.head())

graph = build_similarity_graph(symbols, similarity_matrix, k=None)
print("\nGRAPH SUMMARY")
print(f"Nodes : {graph.number_of_nodes()}")
print(f"Edges : {graph.number_of_edges()}")

similarity_plot = plot_similarity_graph(graph, output_path=f"graphs/{universe_name}_similarity_graph.png")
print("Similarity Graph Saved....")

communities = detect_communities(graph)
baskets = group_communities(communities)
companies = pd.read_csv(f"universes/{universe_name}.csv")
name_lookup = dict(zip(companies["Symbol"], companies["Name"]))

community_df = save_communities(baskets, name_lookup, f"communities/{universe_name}_communities.csv")

print("\nCOMUNITIES")
for community, stocks in baskets.items():
    print(f"COMMUNITY {community} ({len(stocks)} Stocks)")
    
    for stock in stocks:
        print(f"{stock:<15} {name_lookup[stock]}")

    print()

community_plot = plot_communities_graph(graph,communities, output_path=f"graphs/{universe_name}_community_graph.png")
print("Community Graph Saved....")

community_statistics = generate_community_statistics(dataset_file=f"features/{universe_name}_behaviour_dataset.csv", baskets=baskets,output_file=
                                                     f"reports/community_stats/{universe_name}_community_statistics.csv")
print("\nCOMMUNITY STATISTICS")
print(community_statistics)

print("\nStock Scoring Method: ")
print("1. Hardcoded Weights (default)")
print("2. ML Ensemble (Decision Tree + Random Forest + Gradient Boosting, blended with Hardcoded Weights)")
scoring_choice = input("Enter a choice: ")

stock_scores_file = f"baskets/stock_scores/{universe_name}_stock_scores.csv"

if scoring_choice == "2":
    symbol_map_path = os.path.join(save_dir, "_symbol_filename_map.csv")
    if os.path.exists(symbol_map_path):
        ml_symbols = pd.read_csv(symbol_map_path)["Symbol"].tolist()
    else:
        ml_symbols = companies["Symbol"].tolist()

    ml_start_date = input("Walk-forward training start date (YYYY-MM-DD): ")
    ml_end_date = input("Walk-forward training end date (YYYY-MM-DD): ")
    

    print("\nBuilding walk-forward training dataset...")
    training_data = build_walk_forward_dataset(ml_symbols, save_dir, ml_start_date, ml_end_date)
    print(f"Built {len(training_data)} training rows.")

    # held-out sanity check: last 20% of the date range by time, never seen during eval training
    cutoff = pd.to_datetime(ml_start_date) + 0.8 * (pd.to_datetime(ml_end_date) - pd.to_datetime(ml_start_date))
    eval_train, eval_test = split_train_test(training_data, cutoff)

    if len(eval_train) > 0 and len(eval_test) > 0:
        print(f"\nHeld-out evaluation (train <= {cutoff.date()}, test after):")
        eval_models = train_all_models(eval_train)
        for name, model in eval_models.items():
            metrics = evaluate_model(model, eval_test)
            print(f"{name}: pearson={metrics['pearson_correlation']:.3f}, rank_corr={metrics['rank_correlation']:.3f}, "
                  f"top_minus_bottom_spread={metrics['top_minus_bottom_quantile_spread']:.4f}")
    else:
        print("\nNot enough data on both sides of the 80% cutoff -- skipping held-out evaluation.")

    print("\nTraining final models on the full date range for live scoring...")
    models = train_all_models(training_data)

    stock_scores_file = f"baskets/stock_scores/{universe_name}_stock_scores_ensemble.csv"
    stock_scores = score_stocks_ensemble(behaviour_dataset=f"features/{universe_name}_behaviour_dataset.csv",
                                         communities_file=f"communities/{universe_name}_communities.csv",
                                         models=models,
                                         output_file=stock_scores_file)
else:
    stock_scores = score_stocks(behaviour_dataset=f"features/{universe_name}_behaviour_dataset.csv",
                                communities_file= f"communities/{universe_name}_communities.csv",
                                output_file=stock_scores_file)

print("\nStock Scores: ")
print(stock_scores[["Community", "Symbol", "Company", "Stock_Score"]])

stock_baskets = generate_candidate_baskets(stock_scores_files=stock_scores_file,
                                           similarity_matrix_file=f"graphs/{universe_name}_similarity_matrix.csv",
                                           output_file=f"baskets/candidates/{universe_name}_candidate_basket.csv",
                                           min_size=5, max_size=10)

print("\nBaskets: ")
print(stock_baskets[["Basket_ID", "Community", "Symbol", "Company", "Stock_Score"]])


basket_metrics = generate_basket_metrics(basket_file=f"baskets/candidates/{universe_name}_candidate_basket.csv",
                                         similarity_matrix_file=f"graphs/{universe_name}_similarity_matrix.csv",
                                         behaviour_dataset=f"features/{universe_name}_behaviour_dataset.csv",
                                         output_file=f"reports/basket_metrics/{universe_name}_basket_metrics.csv")
print("\nBaskets Metrics: ")
print(basket_metrics)

basket_scoring = score_baskets(basket_metrics_file=f"reports/basket_metrics/{universe_name}_basket_metrics.csv",
                               output_file=f"baskets/ranked/{universe_name}_basket_rankings.csv")

print("\nBasket Scores: ")
print(basket_scoring[["Rank", "Basket_ID", "Community", "Basket_Score"]])

rankings = load_top_baskets(basket_scoring_file=f"baskets/ranked/{universe_name}_basket_rankings.csv")

while True:
    try:
        k = int(input("\nEnter number of top baskets to be selected: "))
        if 1 <= k <= len(rankings):
            break
        else:
            print(f"Enter a value between 1 and {len(rankings)}")

    except:
        print("Enter a valid Integer value")

selected_baskets = select_top_baskets(rankings, k)
save_portfolio_baskets(selected_baskets, universe_name)

print("\nBasket Returns Calculations: ")
candidate_baskets = load_candidate_baskets(f"baskets/candidates/{universe_name}_candidate_basket.csv")
basket_returns = compute_basket_daily_returns(candidate_baskets, f"data/{universe_name}")
covariance_matrix = compute_covariance_matrix(basket_returns)
save_outputs(basket_returns, covariance_matrix, universe_name)

print("\nExpected Returns for Markowitz: ")
expected_returns = compute_expected_returns(basket_returns)
save_expected_returns(expected_returns, universe_name)
print(expected_returns)

basket_selection = load_selected_baskets(f"portfolio/selected/{universe_name}_selected_baskets.csv")
print("\nAverage Volatility:")
print(selected_baskets[["Basket_ID", "Average_Volatility"]])
print("=================\n")

capital = float(input("\nEnter total investment capital: "))

print("\nHeuristic (Rule-Based) Allocation Strategies: ")
print("1. Equal Weight")
print("2. Score Weighted")
print("3. Inverse Volatility")

print("\nOptimisation Based Allocation Strategies: ")
print("4. Markowitz Optimisation")
print("5. Risk Parity")
print("6. Black Litterman")

choice = input("Enter a choice: ")
strategy_map = {"1" : "equal_weighted", "2" : "score_weighted", "3" : "inverse_volatility", 
                "4" : "Markowitz_Optimisation", "5" : "Risk_Parity", 
                "6" : "Black_Litterman"}
strategy = strategy_map.get(choice)

if strategy is None:
    raise ValueError("Enter a valid heuristic based allocation strategy from the given options")

if choice in ["1", "2", "3"]:

    allocation = allocate_portfolio(basket_selection, strategy, capital)

    save_allocations(allocation, universe_name)

    print("\nPortfolio Allocation")
    print(allocation[["Rank", "Basket_ID", "Weight", "Capital_Distribution"]])

elif choice == "4":
    selected_ids = basket_selection["Basket_ID"].tolist()
    expected_returns = expected_returns.set_index("Basket_ID").loc[selected_ids].reset_index()
    covariance_matrix = covariance_matrix.loc[selected_ids, selected_ids]

    result = optimise_portfolio(expected_returns["Expected_Return"].values, covariance_matrix.values)
    allocation = save_markovitz(result, expected_returns, capital, universe_name)

    print("\nOptimised Portfolio")
    print(allocation)

elif choice == "5":
    selected_ids = basket_selection["Basket_ID"].tolist()
    expected_returns = expected_returns.set_index("Basket_ID").loc[selected_ids].reset_index()
    covariance_matrix = covariance_matrix.loc[selected_ids, selected_ids]

    result = optimize_risk_parity(covariance_matrix.values)
    allocation = save_risk_parity(result, expected_returns, capital, universe_name)

    print("\nRisk Parity Portfolio")
    print(allocation)

elif choice == "6":
    selected_ids = basket_selection["Basket_ID"].tolist()
    expected_returns = expected_returns.set_index("Basket_ID").loc[selected_ids].reset_index()
    covariance_matrix = covariance_matrix.loc[selected_ids, selected_ids]

    selected_baskets = basket_selection.copy()

    tau = 0.025
    market_weights = compute_market_weights(selected_baskets)
    equilibrium_returns = compute_equilibrium_returns(covariance_matrix.values, market_weights)
    picking_matrix = generate_picking_matrix(selected_baskets)
    view_returns = generate_view_returns(selected_baskets, equilibrium_returns)
    confidence_matrix = compute_confidence_matrix(selected_baskets, covariance_matrix.values, tau=tau)

    posterior_returns = compute_black_litterman_returns(covariance_matrix.values, equilibrium_returns, picking_matrix, confidence_matrix, view_returns, tau=tau)
    save_black_litterman(posterior_returns, expected_returns, capital, universe_name)

    result = optimise_portfolio(posterior_returns, covariance_matrix.values)
    allocation = save_markovitz(result, expected_returns, capital, universe_name, filename_suffix="_black_litterman")
    print("\n Black-Litterman Portfolio")
    print(allocation)

print("\n===================================")
print("WITHIN-BASKET ALLOCATION")
print("===================================\n")
print("This decides how each basket's own capital is split among its member stocks --")
print("a separate choice from the basket-level strategy above.\n")
print("1. Equal Weight (default -- same behaviour as before this feature existed)")
print("2. Range Position (Mid-Weighted) -- heaviest weight to stocks mid-range in their 52-week high/low")
print("3. 52-Week High Momentum -- heaviest weight to stocks near their 52-week high with strong momentum")
print("4. Inverse Volatility (Intra-Basket) -- calmer stocks in the basket get more capital")
print("5. Volume-Conditioned Momentum -- momentum winners, discounted if recent volume spiked")

intra_choice = input("Enter a choice: ")
intra_strategy_map = {
    "1": "equal_weighted", "2": "range_position", "3": "high_momentum",
    "4": "inverse_volatility_intra", "5": "volume_conditioned_momentum",
}
intra_basket_strategy = intra_strategy_map.get(intra_choice, "equal_weighted")

# stock_baskets only carries Basket_ID/Community/Symbol/Company/Stock_Score -- the
# within-basket strategies need the actual behaviour features (Volatility, Momentum,
# Volume_Growth, Range_Position_52W), which live in the behaviour dataset built earlier.
behaviour_dataset = pd.read_csv(f"features/{universe_name}_behaviour_dataset.csv")
baskets_with_features = stock_baskets.merge(behaviour_dataset, on="Symbol", how="left")

stock_capital = basket_weights_to_stock_weights_v2(
    baskets_with_features, allocation["Basket_ID"].tolist(), allocation["Capital_Distribution"].tolist(),
    strategy=intra_basket_strategy,
)

stock_allocation = pd.DataFrame(
    [{"Symbol": symbol, "Capital_Distribution": amount} for symbol, amount in stock_capital.items()]
).merge(stock_baskets[["Basket_ID", "Symbol", "Company"]], on="Symbol", how="left")
stock_allocation = stock_allocation.sort_values("Capital_Distribution", ascending=False).reset_index(drop=True)

stock_allocation_dir = "./portfolio/stock_allocations"
os.makedirs(stock_allocation_dir, exist_ok=True)
stock_allocation_file = os.path.join(stock_allocation_dir, f"{universe_name}_stock_allocation.csv")
stock_allocation.to_csv(stock_allocation_file, index=False)

print(f"\nPer-Stock Capital Allocation ({intra_basket_strategy}):")
print(stock_allocation[["Basket_ID", "Symbol", "Company", "Capital_Distribution"]].to_string(index=False))
print(f"\nSaved to: {stock_allocation_file}")

print("\n===================================")
print("CAPITAL SIZING CHECK")
print("===================================\n")
print("NSE/BSE (and most non-US exchanges) require whole-share purchases -- this checks")
print("whether your entered capital can actually buy every selected stock as at least")
print("1 whole share close to its target weight, or if it's below the real-world floor.\n")

latest_prices = load_latest_prices(stock_allocation["Symbol"].tolist(), save_dir)
sizing = compute_minimum_viable_capital(stock_allocation, capital, latest_prices)

if sizing.empty:
    print("Could not compute a capital sizing check -- no price data available for the selected stocks.")
else:
    floor_row = sizing.iloc[0]
    min_viable_capital = floor_row["Required_Capital_For_1_Share"]

    print(f"Minimum viable capital for this exact selection: {min_viable_capital:,.2f}")
    print(f"(Set by {floor_row['Symbol']} -- {floor_row['Company']}, priced at {floor_row['Price']:,.2f}, "
          f"{floor_row['Target_Weight']*100:.2f}% target weight)")

    if capital < min_viable_capital:
        affordability = check_affordability(stock_allocation, capital, latest_prices)
        unaffordable = affordability[~affordability["Affordable"]]
        print(f"\nYour entered capital ({capital:,.2f}) is below this floor.")
        print(f"{len(unaffordable)} of {len(affordability)} selected stocks would get 0 whole shares "
              f"and sit as unspent cash at this capital level:")
        print(unaffordable[["Basket_ID", "Symbol", "Company", "Price", "Allocated_Capital"]].to_string(index=False))
    else:
        print(f"\nYour entered capital ({capital:,.2f}) comfortably covers this -- every selected stock "
              f"can be bought as at least 1 whole share close to its target weight.")

    sizing_dir = "./portfolio/capital_sizing"
    os.makedirs(sizing_dir, exist_ok=True)
    sizing_file = os.path.join(sizing_dir, f"{universe_name}_capital_sizing.csv")
    sizing.to_csv(sizing_file, index=False)
    print(f"\nSaved to: {sizing_file}")

print("\n===================================")
print("WALK-FORWARD BACKTEST")
print("===================================\n")

backtest_choice = input("Backtest this configuration over history? (y/n): ").strip().lower()

if backtest_choice == "y":
    bt_start_date = input("Backtest start date (YYYY-MM-DD): ")
    bt_end_date = input("Backtest end date (YYYY-MM-DD): ")

    bt_models = None
    bt_scoring_method = "ensemble" if scoring_choice == "2" else "hardcoded"

    if scoring_choice == "2":
        bt_train_start = input("ML training window start date (must end before backtest start, to avoid lookahead): ")
        bt_train_end = input("ML training window end date: ")

        print("\nBuilding walk-forward training dataset for the backtest...")
        bt_training_data = build_walk_forward_dataset(ml_symbols, save_dir, bt_train_start, bt_train_end)
        print(f"Built {len(bt_training_data)} training rows.")
        bt_models = train_all_models(bt_training_data)

    print("\nRunning equal-weight universe benchmark...")
    benchmark = run_benchmark(universe_name, bt_start_date, bt_end_date)
    benchmark["Capital_Value"] = benchmark["Equity"] * capital
    benchmark_stats = compute_summary_stats(benchmark)
    benchmark_stats["Scoring_Method"] = "benchmark"
    benchmark_stats["Allocation_Strategy"] = "equal_weight_universe"
    benchmark_stats["Within_Basket_Strategy"] = "n/a"

    print(f"Running backtest: scoring={bt_scoring_method}, allocation={strategy}, "
          f"within-basket={intra_basket_strategy}...")
    equity_curve = run_backtest(universe_name, bt_start_date, bt_end_date, bt_scoring_method, strategy,
                               k_baskets=k, models=bt_models, intra_basket_strategy=intra_basket_strategy)
    equity_curve["Capital_Value"] = equity_curve["Equity"] * capital
    stats = compute_summary_stats(equity_curve)
    stats["Scoring_Method"] = bt_scoring_method
    stats["Allocation_Strategy"] = strategy
    stats["Within_Basket_Strategy"] = intra_basket_strategy

    os.makedirs("backtests", exist_ok=True)
    equity_curve.to_csv(f"backtests/{universe_name}_{bt_scoring_method}_{strategy}_equity_curve.csv", index=False)
    benchmark.to_csv(f"backtests/{universe_name}_benchmark_equity_curve.csv", index=False)

    summary_df = pd.DataFrame([stats, benchmark_stats])
    summary_file = f"backtests/{universe_name}_backtest_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\nSaved summary -> {summary_file}")
    print("\n" + summary_df.to_string(index=False))
