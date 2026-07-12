import os;
import numpy as np;
import pandas as pd;
from scipy.optimize import minimize

def portfolio_volatility(weights, covariance_matrix):
    variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    volatility = np.sqrt(variance)
    return volatility

def portfolio_risk_contribution(weights, covariance_matrix):
    portfolio_vol = portfolio_volatility(weights, covariance_matrix)
    marginal_risk_contribution = np.dot(covariance_matrix, weights) / portfolio_vol
    risk_contribution = weights * marginal_risk_contribution
    return risk_contribution

def risk_parity_objective(weights, covariance_matrix):
    portfolio_vol = portfolio_volatility(weights, covariance_matrix)
    risk_contribution = portfolio_risk_contribution(weights, covariance_matrix)
    percentage_risk = risk_contribution / portfolio_vol
    target = np.ones(len(weights)) / len(weights)
    return np.sum((percentage_risk - target) ** 2)

def optimize_risk_parity(covariance_matrix):
    n_assets = len(covariance_matrix)
    intial_weights = np.ones(n_assets) / n_assets

    bounds = tuple((0, 0.4) for _ in range(n_assets))
    constraints = ({"type" : "eq", "fun" : lambda x: np.sum(x) - 1},)
    
    result = minimize(fun=risk_parity_objective, x0 = intial_weights, 
                      args = (covariance_matrix,), method="SLSQP", bounds=bounds, constraints=constraints, 
                      options={ "ftol": 1e-12, "maxiter": 1000})
    
    return result

def save_risk_parity(result, expected_returns_df, capital, universe_name, save_dir = "./portfolio/optimisations"):
    os.makedirs(save_dir, exist_ok=True)
    allocation = pd.DataFrame({"Basket_ID" : expected_returns_df["Basket_ID"], "Weight" : result.x})
    allocation["Capital_Distribution"] = (allocation["Weight"] * capital)
    allocation.loc[np.abs(allocation["Weight"]) < 1e-8, "Weight"] = 0
    allocation.loc[np.abs(allocation["Capital_Distribution"]) < 1e-4, "Capital_Distribution"] = 0
    output_file = os.path.join(save_dir, f"{universe_name}_risk_parity.csv")
    allocation.to_csv(output_file, index=False)
    return allocation    