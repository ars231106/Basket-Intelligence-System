import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_basket_rankings(selected_baskets_file):
    df = pd.read_csv(selected_baskets_file)
    return df

def compute_market_weights(selected_baskets_df):
    """
    Proxy for each basket's relative "market" weight, used as the prior the
    Black-Litterman equilibrium return is built from. This pipeline has no
    true market-capitalisation data (download_data.py only pulls OHLCV
    history), so an equal weight was used previously -- but that means the
    BL equilibrium anchor is not "the market", it is just "equal weight by
    construction", which is why the posterior kept reverting to equal
    weight regardless of the views fed in.

    As a practical substitute, this uses each basket's average traded
    dollar volume (Average_Volume) as an observable proxy for relative
    economic significance. It's not a substitute for real market cap /
    AUM data -- if that becomes available (e.g. via shares outstanding),
    it should replace this.
    """
    if "Average_Volume" in selected_baskets_df.columns:
        weights = selected_baskets_df["Average_Volume"].to_numpy(dtype=float)
        if np.isfinite(weights).all() and weights.sum() > 0:
            return weights / weights.sum()

    n_assets = len(selected_baskets_df)
    return np.ones(n_assets) / n_assets

def compute_equilibrium_returns(covaraiance_matrix, market_weights, risk_aversion=2.5):
    equilibrium_returns = risk_aversion * np.dot(covaraiance_matrix, market_weights)
    return equilibrium_returns

def generate_picking_matrix(selected_baskets_df):
    n_assets = len(selected_baskets_df)
    picking_matirx = np.eye(n_assets)
    return picking_matirx

def generate_view_returns(selected_baskets_df, equilibrium_returns, sensitivity=0.5):
    """
    View returns for each basket, expressed relative to its own
    equilibrium return rather than as a share of total Basket_Score.

    The previous version (scores / sum(scores)) produced a vector that
    sums to 1 across baskets -- effectively a weight, not a return -- and
    for a handful of already-similar, top-ranked baskets that vector
    naturally clusters near 1/n. Feeding a near-flat "view" into the BL
    blend meant the posterior had nothing to pull away from equilibrium
    with, however differentiated the equilibrium itself was.

    Here each basket's view is its equilibrium return nudged up or down by
    its standardised Basket_Score (z-score across the selected baskets),
    scaled by `sensitivity`. A basket scoring one standard deviation above
    the mean gets a view `sensitivity` (default 50%) above its equilibrium
    return; one std below gets a view that much lower. This lets genuinely
    high-conviction baskets actually diverge from equilibrium, while a
    basket with an average score reverts to (near) equilibrium, which is
    the behaviour Black-Litterman is meant to have.
    """
    scores = selected_baskets_df["Basket_Score"].to_numpy(dtype=float)
    std = scores.std()

    if std < 1e-12:
        z = np.zeros_like(scores)
    else:
        z = (scores - scores.mean()) / std

    view_returns = equilibrium_returns * (1 + sensitivity * z)
    return view_returns

def compute_confidence_matrix(selected_baskets_df, covariance_matrix, tau=0.025):
    scalar = MinMaxScaler()
    confidence_features = selected_baskets_df[["Basket_Score", "Basket_Cohesion", "Average_Sharpe_Ratio"]]
    scaled_features = scalar.fit_transform(confidence_features)
    confidence_scores = (0.40 * scaled_features[:, 0] + 0.40 * scaled_features[:, 1] + 0.20 * scaled_features[:, 2])
    confidence_scores = np.clip(confidence_scores, 0.05, 1.00)
    # Omega scaled to tau*Sigma's own units (variance), not an arbitrary 1-20 range, so views actually compete with equilibrium
    variances = np.diag(covariance_matrix)
    omega_diag = (tau * variances) / confidence_scores
    confidence_matrix = np.diag(omega_diag)
    return confidence_matrix

def compute_black_litterman_returns(covariance_matrix, equilibrium_returns, picking_matrix, confidence_matrix, view_returns, tau=0.025):
    tau_sigma = tau * covariance_matrix

    left = np.linalg.inv(np.linalg.inv(tau_sigma) + picking_matrix.T @ np.linalg.inv(confidence_matrix) @ picking_matrix)
    right = (np.linalg.inv(tau_sigma) @ equilibrium_returns + picking_matrix.T @ np.linalg.inv(confidence_matrix) @ view_returns)

    posterior_returns = left @ right
    return posterior_returns

def save_black_litterman(posterior_returns, expected_returns_df, capital, universe_name,
                         save_dir="./portfolio/optimisations"):
    os.makedirs(save_dir, exist_ok=True)

    allocation = expected_returns_df.copy()
    allocation["Posterior_Return"] = posterior_returns

    output_file = os.path.join(save_dir, f"{universe_name}_black_litterman_returns.csv")
    allocation.to_csv(output_file, index=False)
    return allocation
