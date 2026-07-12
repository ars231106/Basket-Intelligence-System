import os
import pandas as pd
import numpy as np

from src.download_data import sanitize_filename

def load_candidate_baskets(candidate_baskets_file):
    df = pd.read_csv(candidate_baskets_file)
    return df

def load_stock_returns(symbol, data_folder):
    # download_data.py may have sanitised this symbol on disk (a '/' in the
    # ticker, or a Windows-reserved device name) -- reconstruct the same
    # sanitised filename here so the lookup actually finds the file.
    file = os.path.join(data_folder, f"{sanitize_filename(symbol)}.csv")
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Daily_Returns"] = df["Close"].pct_change()
    return df[["Date", "Daily_Returns"]]

def compute_basket_daily_returns(candidate_baskets_file, data_folder):
    basket_returns = pd.DataFrame()
    basket_ids = candidate_baskets_file["Basket_ID"].unique()

    for basket in basket_ids:
        symbols = candidate_baskets_file.loc[candidate_baskets_file["Basket_ID"] == basket, "Symbol"]
        stock_returns = []

        for sym in symbols:
            returns = load_stock_returns(sym, data_folder)
            returns = returns.rename(columns={"Daily_Returns" : sym})
            stock_returns.append(returns)

        merged = stock_returns[0]

        for returns in stock_returns[1:]:
            merged = merged.merge(returns, on="Date", how="inner")
        
        merged[basket] = merged.drop(columns="Date").mean(axis=1)
        basket_series = merged[["Date", basket]]

        if basket_returns.empty:
            basket_returns = basket_series

        else:
            basket_returns = basket_returns.merge(basket_series, on="Date", how="inner")

    basket_returns = basket_returns.dropna()
    return basket_returns

def compute_covariance_matrix(basket_returns, shrinkage_intensity=0.2):
    """
    Sample covariance shrunk toward a scaled-identity target (simplified
    Ledoit-Wolf style shrinkage). The raw sample covariance is noisy,
    especially with few return observations relative to the number of
    baskets; shrinking the off-diagonal structure toward a diagonal target
    stabilises the matrix that feeds the optimiser.

    shrinkage_intensity: fixed value in [0, 1]. 0 = raw sample covariance
    (original behaviour), 1 = pure diagonal target. 0.2 is a common,
    conservative default; this is not the fully-estimated optimal Ledoit-Wolf
    intensity (that requires the pairwise asymptotic variance terms), just a
    fixed, documented compromise.
    """
    sample_cov = basket_returns.drop(columns="Date").cov()

    if shrinkage_intensity <= 0:
        return sample_cov

    avg_var = np.trace(sample_cov.values) / len(sample_cov)
    target = np.eye(len(sample_cov)) * avg_var
    shrunk = (1 - shrinkage_intensity) * sample_cov.values + shrinkage_intensity * target

    return pd.DataFrame(shrunk, index=sample_cov.index, columns=sample_cov.columns)

def save_outputs(basket_returns, covariance_matrix, universe_name):
    os.makedirs("portfolio/returns", exist_ok=True)
    os.makedirs("portfolio/covariance", exist_ok=True)

    basket_returns.to_csv(f"portfolio/returns/{universe_name}_basket_returns.csv", index=False)
    covariance_matrix.to_csv(f"portfolio/covariance/{universe_name}_basket_covariance.csv")
    
    print("\nBasket returns saved.")
    print("Basket covariance matrix saved.")

def compute_expected_returns(basket_returns, shrinkage=True):
    """
    Expected returns, optionally shrunk toward the cross-basket grand mean
    using the Jorion (1986) Bayes-Stein estimator. Raw historical sample
    means are noisy estimates of true expected return, and mean-variance
    optimisation is extremely sensitive to that noise -- it tends to bet
    heavily on whichever basket happened to have the best historical
    average, which is exactly the corner-solution behaviour this is meant
    to dampen. Shrinkage intensity is estimated from the data (not a fixed
    constant): it shrinks harder when the spread across basket means is
    small relative to their estimation error, and barely at all when the
    baskets are genuinely, confidently different.
    """
    returns_only = basket_returns.drop(columns="Date")
    mu = returns_only.mean()

    if shrinkage and len(mu) >= 3:
        T = len(returns_only)
        k = len(mu)
        grand_mean = mu.mean()

        sample_var = returns_only.var()
        se2 = (sample_var / T).mean()

        dispersion = ((mu - grand_mean) ** 2).sum()

        if dispersion > 1e-12:
            lam = ((k - 2) * se2) / dispersion
            lam = min(max(lam, 0.0), 1.0)
        else:
            lam = 1.0

        mu = grand_mean + (1 - lam) * (mu - grand_mean)

    expected_returns = mu.reset_index()
    expected_returns.columns = ["Basket_ID", "Expected_Return"]
    return expected_returns

def save_expected_returns(expected_returns, universe_name, save_dir="./portfolio/returns"):
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, f"{universe_name}_expected_returns.csv")

    expected_returns.to_csv(output_file, index=False)

    print(f"\nExpected returns saved to:\n{output_file}")