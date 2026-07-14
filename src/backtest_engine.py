import os
import shutil
import pandas as pd
import numpy as np

from src.feature_engineering import compute_features
from src.backtest_dataset import load_price_series, generate_rebalance_dates
from src.stock_scoring import score_stocks
from src.ml_scoring import score_stocks_ensemble
from src.basket_generator import generate_candidate_baskets
from src.basket_metrics import generate_basket_metrics
from src.basket_scoring import score_baskets
from src.portfolio_constructor import load_top_baskets, select_top_baskets
from src.portfolio_allocator import allocate_portfolio
from src.markowitz_optimiser import optimise_portfolio
from src.risk_parity_optimiser import optimize_risk_parity
from src.black_litterman import (compute_market_weights, compute_equilibrium_returns,
                                 generate_picking_matrix, generate_view_returns,
                                 compute_confidence_matrix, compute_black_litterman_returns)
from src.intra_basket_allocator import basket_weights_to_stock_weights_v2

HEURISTIC_STRATEGIES = {"equal_weighted", "score_weighted", "inverse_volatility"}


def load_all_price_data(symbols, data_folder):
    price_data = {}
    for symbol in symbols:
        try:
            price_data[symbol] = load_price_series(symbol, data_folder)
        except Exception as e:
            print(f"Skipping {symbol}: {e}")
    return price_data


def build_trailing_behaviour_dataset(symbols, price_data, as_of_date, min_history=260):
    rows = []
    for symbol in symbols:
        df = price_data.get(symbol)
        if df is None:
            continue
        trailing = df[df["Date"] <= as_of_date]
        if len(trailing) < min_history:
            continue
        features = compute_features(trailing)
        if any(pd.isna(v) for v in features.values()):
            continue
        rows.append({"Symbol": symbol, **features})
    return pd.DataFrame(rows)


def compute_trailing_basket_returns(basket_df, price_data, as_of_date):
    series_by_basket = {}
    for basket_id, group in basket_df.groupby("Basket_ID"):
        member_returns = []
        for symbol in group["Symbol"]:
            df = price_data.get(symbol)
            if df is None:
                continue
            trailing = df[df["Date"] <= as_of_date].set_index("Date")["Close"]
            member_returns.append(trailing.pct_change().rename(symbol))
        if not member_returns:
            continue
        merged = pd.concat(member_returns, axis=1, join="inner")
        series_by_basket[basket_id] = merged.mean(axis=1)

    basket_returns = pd.DataFrame(series_by_basket).dropna()
    basket_returns = basket_returns.reset_index().rename(columns={"index": "Date"})
    return basket_returns


def compute_forward_returns(symbols, price_data, t, t_next):
    forward_returns = {}
    for symbol in symbols:
        df = price_data.get(symbol)
        if df is None:
            continue
        trailing = df[df["Date"] <= t]
        future = df[df["Date"] >= t_next]
        if trailing.empty or future.empty:
            continue
        entry_price = trailing["Close"].iloc[-1]
        exit_price = future["Close"].iloc[0]
        forward_returns[symbol] = (exit_price / entry_price) - 1
    return forward_returns


def basket_weights_to_stock_weights(basket_df, basket_ids, weights):
    stock_weights = {}
    for basket_id, weight in zip(basket_ids, weights):
        members = basket_df.loc[basket_df["Basket_ID"] == basket_id, "Symbol"].tolist()
        if not members:
            continue
        per_stock = weight / len(members)
        for symbol in members:
            stock_weights[symbol] = stock_weights.get(symbol, 0) + per_stock
    return stock_weights


def compute_turnover(prev_weights, new_weights):
    symbols = set(prev_weights) | set(new_weights)
    diff = sum(abs(new_weights.get(s, 0) - prev_weights.get(s, 0)) for s in symbols)
    return diff / 2


def reindex_expected_returns(expected_returns, selected_ids):
    return expected_returns.set_index("Basket_ID").loc[selected_ids].reset_index()


def allocate(strategy, selected_baskets, basket_returns, tmp_dir):
    selected_ids = selected_baskets["Basket_ID"].tolist()
    weights = None

    if strategy in HEURISTIC_STRATEGIES:
        allocation = allocate_portfolio(selected_baskets, strategy, 1.0)
        weights = allocation["Weight"].to_numpy()

    else:
        returns_only = basket_returns[selected_ids]
        cov = returns_only.cov().loc[selected_ids, selected_ids].values
        avg_var = np.trace(cov) / len(cov)
        cov = 0.8 * cov + 0.2 * np.eye(len(cov)) * avg_var
        mu = returns_only.mean().reindex(selected_ids).values

        if strategy == "Markowitz_Optimisation":
            result = optimise_portfolio(mu, cov)
            weights = result.x

        elif strategy == "Risk_Parity":
            result = optimize_risk_parity(cov)
            weights = result.x

        elif strategy == "Black_Litterman":
            tau = 0.025
            market_weights = compute_market_weights(selected_baskets)
            equilibrium_returns = compute_equilibrium_returns(cov, market_weights)
            picking_matrix = generate_picking_matrix(selected_baskets)
            view_returns = generate_view_returns(selected_baskets, equilibrium_returns)
            confidence_matrix = compute_confidence_matrix(selected_baskets, cov, tau=tau)
            posterior_returns = compute_black_litterman_returns(cov, equilibrium_returns, picking_matrix,
                                                                 confidence_matrix, view_returns, tau=tau)
            result = optimise_portfolio(posterior_returns, cov)
            weights = result.x

        else:
            raise ValueError(f"Unknown allocation strategy: {strategy}")

    return dict(zip(selected_ids, weights))


def run_backtest(universe_name, start_date, end_date, scoring_method, allocation_strategy,
                 k_baskets=9, freq="MS", txn_cost_bps=10, min_history=260, models=None,
                 tmp_dir="backtest_tmp", intra_basket_strategy="equal_weighted"):
    os.makedirs(tmp_dir, exist_ok=True)

    communities_file = f"communities/{universe_name}_communities.csv"
    similarity_matrix_file = f"graphs/{universe_name}_similarity_matrix.csv"
    data_folder = f"data/{universe_name}"

    communities = pd.read_csv(communities_file)
    symbols = communities["Symbol"].unique().tolist()
    price_data = load_all_price_data(symbols, data_folder)

    rebalance_dates = generate_rebalance_dates(start_date, end_date, freq)

    records = []
    prev_stock_weights = {}

    for i in range(len(rebalance_dates) - 1):
        t = rebalance_dates[i]
        t_next = rebalance_dates[i + 1]

        trailing_behaviour = build_trailing_behaviour_dataset(symbols, price_data, t, min_history)
        if trailing_behaviour.empty:
            continue

        behaviour_file = os.path.join(tmp_dir, "behaviour.csv")
        trailing_behaviour.to_csv(behaviour_file, index=False)

        scores_file = os.path.join(tmp_dir, "scores.csv")
        if scoring_method == "ensemble":
            score_stocks_ensemble(behaviour_file, communities_file, models, scores_file)
        else:
            score_stocks(behaviour_file, communities_file, scores_file)

        baskets_file = os.path.join(tmp_dir, "baskets.csv")
        candidate_baskets = generate_candidate_baskets(scores_file, similarity_matrix_file, baskets_file,
                                                       min_size=5, max_size=10)

        metrics_file = os.path.join(tmp_dir, "metrics.csv")
        generate_basket_metrics(baskets_file, similarity_matrix_file, behaviour_file, metrics_file)

        rankings_file = os.path.join(tmp_dir, "rankings.csv")
        score_baskets(metrics_file, rankings_file)

        rankings = load_top_baskets(rankings_file)
        k = min(k_baskets, len(rankings))
        selected_baskets = select_top_baskets(rankings, k)

        basket_returns = None
        if allocation_strategy not in HEURISTIC_STRATEGIES:
            basket_returns = compute_trailing_basket_returns(candidate_baskets, price_data, t)
            available = [b for b in selected_baskets["Basket_ID"] if b in basket_returns.columns]
            selected_baskets = selected_baskets[selected_baskets["Basket_ID"].isin(available)].reset_index(drop=True)
            if len(selected_baskets) < 2:
                continue

        basket_weights = allocate(allocation_strategy, selected_baskets, basket_returns, tmp_dir)

        # candidate_baskets only carries Basket_ID/Community/Symbol/Company/Stock_Score --
        # the within-basket strategies need the actual behaviour features (Volatility,
        # Momentum, Volume_Growth, Range_Position_52W), which live in trailing_behaviour.
        # Every symbol in candidate_baskets was scored from trailing_behaviour in the
        # first place, so this merge is always complete -- no missing-feature rows.
        candidate_baskets_with_features = candidate_baskets.merge(trailing_behaviour, on="Symbol", how="left")
        stock_weights = basket_weights_to_stock_weights_v2(candidate_baskets_with_features, basket_weights.keys(),
                                                            basket_weights.values(), strategy=intra_basket_strategy)

        forward_returns = compute_forward_returns(stock_weights.keys(), price_data, t, t_next)
        gross_return = sum(w * forward_returns.get(s, 0) for s, w in stock_weights.items())

        turnover = compute_turnover(prev_stock_weights, stock_weights)
        cost = turnover * (txn_cost_bps / 10000)
        net_return = gross_return - cost

        records.append({
            "Date": t, "Next_Date": t_next, "Baskets_Selected": len(selected_baskets),
            "Gross_Return": gross_return, "Turnover": turnover, "Cost": cost, "Net_Return": net_return,
        })

        prev_stock_weights = stock_weights

    equity_curve = pd.DataFrame(records)
    if not equity_curve.empty:
        equity_curve["Equity"] = (1 + equity_curve["Net_Return"]).cumprod()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return equity_curve


def run_benchmark(universe_name, start_date, end_date, freq="MS", min_history=260):
    communities_file = f"communities/{universe_name}_communities.csv"
    data_folder = f"data/{universe_name}"

    communities = pd.read_csv(communities_file)
    symbols = communities["Symbol"].unique().tolist()
    price_data = load_all_price_data(symbols, data_folder)

    rebalance_dates = generate_rebalance_dates(start_date, end_date, freq)
    records = []

    for i in range(len(rebalance_dates) - 1):
        t = rebalance_dates[i]
        t_next = rebalance_dates[i + 1]

        eligible = [s for s in symbols if price_data.get(s) is not None
                   and len(price_data[s][price_data[s]["Date"] <= t]) >= min_history]
        if not eligible:
            continue

        forward_returns = compute_forward_returns(eligible, price_data, t, t_next)
        if not forward_returns:
            continue

        equal_weight = 1 / len(forward_returns)
        period_return = sum(equal_weight * r for r in forward_returns.values())
        records.append({"Date": t, "Next_Date": t_next, "Net_Return": period_return})

    equity_curve = pd.DataFrame(records)
    if not equity_curve.empty:
        equity_curve["Equity"] = (1 + equity_curve["Net_Return"]).cumprod()
    return equity_curve


def compute_summary_stats(equity_curve, periods_per_year=12):
    if equity_curve.empty or len(equity_curve) < 2:
        return {"error": "Not enough rebalance periods to compute stats"}

    returns = equity_curve["Net_Return"]
    total_return = equity_curve["Equity"].iloc[-1] - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year

    cagr = (equity_curve["Equity"].iloc[-1]) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (returns.mean() * periods_per_year) / ann_vol if ann_vol > 0 else np.nan

    running_max = equity_curve["Equity"].cummax()
    drawdown = (equity_curve["Equity"] - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "Total_Return": total_return, "CAGR": cagr, "Annualised_Volatility": ann_vol,
        "Sharpe_Ratio": sharpe, "Max_Drawdown": max_drawdown,
        "Avg_Turnover": equity_curve.get("Turnover", pd.Series(dtype=float)).mean(),
        "N_Rebalances": n_periods,
    }
