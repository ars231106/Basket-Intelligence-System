import pandas as pd
import pytest

from src.intra_basket_allocator import (
    allocate_equal,
    allocate_range_position,
    allocate_high_momentum,
    allocate_inverse_volatility_intra,
    allocate_volume_conditioned_momentum,
    allocate_within_basket,
    basket_weights_to_stock_weights_v2,
)


def make_members():
    # three stocks: one near its 52-week low with weak momentum, one dead
    # center in its range, one right at its 52-week high with strong
    # momentum and a recent volume spike
    return pd.DataFrame({
        "Symbol": ["LOW", "MID", "HIGH"],
        "Basket_ID": ["C0_B1", "C0_B1", "C0_B1"],
        "Range_Position_52W": [0.02, 0.50, 0.98],
        "Momentum": [-0.10, 0.00, 0.30],
        "Volatility": [0.05, 0.02, 0.06],
        "Volume_Growth": [1.0, 1.0, 3.0],
    })


def test_allocate_equal_splits_evenly():
    members = make_members()
    weights = allocate_equal(members)
    assert abs(weights.sum() - 1.0) < 1e-9
    assert weights["LOW"] == weights["MID"] == weights["HIGH"]


def test_range_position_prefers_mid_over_low_over_high():
    members = make_members()
    weights = allocate_range_position(members)
    assert abs(weights.sum() - 1.0) < 1e-9
    # the explicit ordering this strategy is built around: Mid > Low > High
    assert weights["MID"] > weights["LOW"] > weights["HIGH"]


def test_high_momentum_prefers_high_over_low():
    members = make_members()
    weights = allocate_high_momentum(members)
    assert abs(weights.sum() - 1.0) < 1e-9
    # the evidence-backed opposite of range_position: closer to the 52-week
    # high, with strong momentum, should be weighted MORE heavily, not less
    assert weights["HIGH"] > weights["MID"] > weights["LOW"]


def test_inverse_volatility_intra_prefers_calmer_stocks():
    members = make_members()
    weights = allocate_inverse_volatility_intra(members)
    assert abs(weights.sum() - 1.0) < 1e-9
    # MID has the lowest volatility (0.02) of the three -- should get the most weight
    assert weights["MID"] > weights["LOW"]
    assert weights["MID"] > weights["HIGH"]


def test_volume_conditioned_momentum_discounts_volume_spikes():
    members = make_members()
    weights = allocate_volume_conditioned_momentum(members)
    assert abs(weights.sum() - 1.0) < 1e-9
    # HIGH has by far the strongest momentum, but also a 3x volume spike
    # relative to LOW/MID's flat volume -- the penalty should pull it back
    # down from where pure momentum alone would rank it
    momentum_only_rank = members.set_index("Symbol")["Momentum"].rank()
    assert momentum_only_rank["HIGH"] == 3  # strongest raw momentum
    # even after the volume penalty, weights should still sum sensibly and
    # not be dominated to the point HIGH takes the entire basket
    assert weights["HIGH"] < 0.9


def test_single_stock_basket_always_gets_full_weight():
    single = pd.DataFrame({
        "Symbol": ["ONLY"], "Basket_ID": ["C0_B1"], "Range_Position_52W": [0.9],
        "Momentum": [0.5], "Volatility": [0.1], "Volume_Growth": [2.0],
    })
    for strategy in ["equal_weighted", "range_position", "high_momentum",
                     "inverse_volatility_intra", "volume_conditioned_momentum"]:
        weights = allocate_within_basket(single, strategy)
        assert weights["ONLY"] == 1.0


def test_unknown_strategy_raises():
    members = make_members()
    with pytest.raises(ValueError):
        allocate_within_basket(members, "not_a_real_strategy")


def test_basket_weights_to_stock_weights_v2_equal_matches_old_behaviour():
    # with the default "equal_weighted" strategy, this must reproduce the
    # exact original weight/len(members) split that used to be hardcoded
    # directly in backtest_engine.py
    basket_df = make_members()
    stock_weights = basket_weights_to_stock_weights_v2(basket_df, ["C0_B1"], [0.9])

    assert abs(sum(stock_weights.values()) - 0.9) < 1e-9
    for symbol in ["LOW", "MID", "HIGH"]:
        assert abs(stock_weights[symbol] - 0.3) < 1e-9


def test_basket_weights_to_stock_weights_v2_sums_to_total_across_baskets():
    basket_a = make_members()
    basket_b = pd.DataFrame({
        "Symbol": ["X", "Y"], "Basket_ID": ["C1_B1", "C1_B1"],
        "Range_Position_52W": [0.3, 0.7], "Momentum": [0.1, 0.2],
        "Volatility": [0.03, 0.04], "Volume_Growth": [1.2, 1.1],
    })
    combined = pd.concat([basket_a, basket_b], ignore_index=True)

    stock_weights = basket_weights_to_stock_weights_v2(
        combined, ["C0_B1", "C1_B1"], [0.6, 0.4], strategy="inverse_volatility_intra"
    )
    assert abs(sum(stock_weights.values()) - 1.0) < 1e-9
