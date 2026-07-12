import numpy as np
import pandas as pd
from src.black_litterman import compute_confidence_matrix, compute_black_litterman_returns


def make_baskets(scores, cohesion, sharpe):
    return pd.DataFrame({
        "Basket_Score": scores,
        "Basket_Cohesion": cohesion,
        "Average_Sharpe_Ratio": sharpe,
    })


def test_omega_scales_with_tau():

    baskets = make_baskets([0.2, 0.8], [0.2, 0.8], [0.2, 0.8])
    covariance_matrix = np.array([[0.04, 0.0], [0.0, 0.09]])

    omega_small_tau = compute_confidence_matrix(baskets, covariance_matrix, tau=0.01)
    omega_big_tau = compute_confidence_matrix(baskets, covariance_matrix, tau=0.05)

    ratio = np.diag(omega_big_tau) / np.diag(omega_small_tau)
    assert np.allclose(ratio, 5.0, rtol=0.05)


def test_higher_confidence_basket_gets_tighter_omega():
    baskets = make_baskets([0.1, 0.9], [0.1, 0.9], [0.1, 0.9])
    covariance_matrix = np.array([[0.04, 0.0], [0.0, 0.04]])

    omega_diag = np.diag(compute_confidence_matrix(baskets, covariance_matrix, tau=0.025))
    assert omega_diag[1] < omega_diag[0]


def test_bl_posterior_matches_equilibrium_when_view_echoes_it():
    covariance_matrix = np.array([[0.04, 0.0], [0.0, 0.09]])
    equilibrium_returns = np.array([0.05, 0.03])
    picking_matrix = np.eye(2)
    view_returns = equilibrium_returns.copy()
    confidence_matrix = np.diag([0.001, 0.001])

    posterior = compute_black_litterman_returns(
        covariance_matrix, equilibrium_returns, picking_matrix, confidence_matrix, view_returns, tau=0.025
    )
    assert np.allclose(posterior, equilibrium_returns, atol=1e-6)


def test_bl_posterior_moves_toward_a_confident_view():
    covariance_matrix = np.array([[0.04, 0.0], [0.0, 0.09]])
    equilibrium_returns = np.array([0.05, 0.03])
    picking_matrix = np.eye(2)
    view_returns = np.array([0.20, 0.03])
    confidence_matrix = np.diag([0.0001, 100.0])

    posterior = compute_black_litterman_returns(
        covariance_matrix, equilibrium_returns, picking_matrix, confidence_matrix, view_returns, tau=0.025
    )
    assert posterior[0] > equilibrium_returns[0] + 0.05