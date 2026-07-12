import pandas as pd
from src.portfolio_allocator import allocate_portfolio

def make_baskets():
    return pd.DataFrame({"Basket_ID" : ["C0_B1", "C1_B1", "C2_B1"], "Basket_Score" : [0.5, 0.3, 0.2], 
                         "Average_Volatility" : [0.02, 0.04, 0.01],})

def test_equal_weighted_matches_main_py_strategy_string():
    baskets = make_baskets()
    result = allocate_portfolio(baskets, "equal_weighted", capital=1000000)
    assert abs(result["Weight"].sum() - 1.0) < 1e-9

def test_score_weighted_weights_sum_to_one():
    baskets = make_baskets()
    result = allocate_portfolio(baskets, "score_weighted", capital=1000000)
    assert abs(result["Weight"].sum() - 1.0) < 1e-9

def test_inverse_volatility_weights_sum_to_one():
    baskets = make_baskets()
    result = allocate_portfolio(baskets, "inverse_volatility", capital=1000000)
    assert abs(result["Weight"].sum() - 1.0) < 1e-9

def test_unknown_strategy_raises():
    baskets = make_baskets()
    try:
        allocate_portfolio(baskets, "not_a_real_strategy", capital = 1000000)
        assert False, "expected a ValueError for an unrecognised strategy"
    except ValueError:
        pass