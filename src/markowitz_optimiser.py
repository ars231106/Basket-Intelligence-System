import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def load_expected_returns(expected_returns_file):
    df = pd.read_csv(expected_returns_file)
    return df

def load_covariance_matrix(covariance_matrix_file):
    df = pd.read_csv(covariance_matrix_file)
    return df

def portfolio_return(weights, expected_returns):
    potfolio_returns = np.dot(weights, expected_returns)
    return potfolio_returns

def portfolio_volatility(weights, covarience_matrix):
    varience = np.dot(weights.T, np.dot(weights, covarience_matrix))
    volatility = np.sqrt(varience)
    return volatility

def negetive_sharpe_ratio_scipy_minimisation(weights, expected_returns, covarience_matrix, risk_free_rate=0):
    port_return = portfolio_return(weights, expected_returns)
    port_volatility = portfolio_volatility(weights, covarience_matrix)

    sharpe = (port_return - risk_free_rate) / port_volatility

    return -sharpe

def optimise_portfolio(expected_returns, covarience_matrix):
    n_assets = len(expected_returns)
    intial_weights = np.ones(n_assets) / n_assets
    bounds = tuple((0,0.4) for _ in range(n_assets))
    constraints = ({"type" : "eq" , 
                    "fun" : lambda x: np.sum(x) - 1})
    
    covarience_matrix = covarience_matrix + np.eye(len(covarience_matrix)) * 1e-6
    
    result = minimize(fun=negetive_sharpe_ratio_scipy_minimisation, x0=intial_weights,
                      args=(expected_returns, covarience_matrix, 0), method="SLSQP", bounds=bounds, 
                      constraints=constraints, options={"ftol": 1e-12, "maxiter": 1000})

    return result

def save_markovitz(result, expected_returns_df, capital, universe_name, save_dir="./portfolio/optimisations", filename_suffix=""):
    os.makedirs(save_dir, exist_ok=True)

    allocation = pd.DataFrame({"Basket_ID" : expected_returns_df["Basket_ID"], "Weight" : result.x})
    allocation["Capital_Distribution"] = (allocation["Weight"] * capital)

    allocation.loc[np.abs(allocation["Weight"]) < 1e-8, "Weight"] = 0
    allocation.loc[np.abs(allocation["Capital_Distribution"]) < 1e-4,
                   "Capital_Distribution"] = 0

    # filename_suffix lets callers (e.g. the Black-Litterman branch, which also
    # runs its posterior returns through this same Markowitz optimiser) save to
    # a distinct file instead of silently overwriting the plain Markowitz output.
    output_file = os.path.join(save_dir, f"{universe_name}{filename_suffix}_markowitz_weights.csv")
    allocation.to_csv(output_file, index=False)
    return allocation
