import pandas as pd

from src.capital_sizing import compute_minimum_viable_capital, check_affordability


def make_allocation():
    # capital = 30000, so weights work out to: EXPENSIVE 8%, MID 20%, CHEAP 12%
    return pd.DataFrame({
        "Symbol": ["EXPENSIVE", "MID", "CHEAP"],
        "Company": ["Expensive Co.", "Mid Co.", "Cheap Co."],
        "Basket_ID": ["C0_B1", "C0_B1", "C1_B1"],
        "Capital_Distribution": [2400.0, 6000.0, 3600.0],
    })


def test_minimum_viable_capital_uses_price_over_weight_not_price_alone():
    allocation = make_allocation()
    capital = 30000.0
    prices = {"EXPENSIVE": 17690.0, "MID": 500.0, "CHEAP": 50.0}

    sizing = compute_minimum_viable_capital(allocation, capital, prices)

    # EXPENSIVE: weight = 2400/30000 = 0.08 -> required = 17690 / 0.08 = 221125
    # MID:       weight = 6000/30000 = 0.20 -> required = 500 / 0.20 = 2500
    # CHEAP:     weight = 3600/30000 = 0.12 -> required = 50 / 0.12 = ~416.67
    expensive_row = sizing[sizing["Symbol"] == "EXPENSIVE"].iloc[0]
    assert abs(expensive_row["Required_Capital_For_1_Share"] - 221125.0) < 1.0

    # the priciest stock also has the tightest weight here, so it should
    # set the floor for the whole portfolio -- first row after sorting
    assert sizing.iloc[0]["Symbol"] == "EXPENSIVE"


def test_minimum_viable_capital_can_be_set_by_a_cheap_tightly_weighted_stock():
    # a CHEAP stock with a tiny enough weight can need more capital to
    # clear than an EXPENSIVE stock with a generous weight -- this is the
    # whole reason the formula is price/weight, not price alone
    allocation = pd.DataFrame({
        "Symbol": ["EXPENSIVE", "CHEAP_TIGHT"],
        "Company": ["Expensive Co.", "Cheap But Tight Co."],
        "Basket_ID": ["C0_B1", "C0_B1"],
        "Capital_Distribution": [9000.0, 100.0],
    })
    capital = 10000.0
    prices = {"EXPENSIVE": 5000.0, "CHEAP_TIGHT": 200.0}

    sizing = compute_minimum_viable_capital(allocation, capital, prices)
    # EXPENSIVE: weight 0.9 -> required = 5000/0.9 = ~5555.6
    # CHEAP_TIGHT: weight 0.01 -> required = 200/0.01 = 20000
    assert sizing.iloc[0]["Symbol"] == "CHEAP_TIGHT"
    assert sizing.iloc[0]["Required_Capital_For_1_Share"] > sizing.iloc[1]["Required_Capital_For_1_Share"]


def test_missing_price_is_skipped_not_crashed_on():
    allocation = make_allocation()
    prices = {"EXPENSIVE": 17690.0, "MID": 500.0}  # CHEAP has no price on file
    sizing = compute_minimum_viable_capital(allocation, 30000.0, prices)
    assert "CHEAP" not in sizing["Symbol"].tolist()
    assert len(sizing) == 2


def test_check_affordability_flags_zero_share_positions():
    allocation = make_allocation()
    prices = {"EXPENSIVE": 17690.0, "MID": 500.0, "CHEAP": 50.0}

    result = check_affordability(allocation, 30000.0, prices)

    expensive_row = result[result["Symbol"] == "EXPENSIVE"].iloc[0]
    assert expensive_row["Whole_Shares_Affordable"] == 0  # 2400 // 17690 = 0
    assert expensive_row["Affordable"] == False

    mid_row = result[result["Symbol"] == "MID"].iloc[0]
    assert mid_row["Whole_Shares_Affordable"] == 12  # 6000 // 500 = 12
    assert mid_row["Affordable"] == True


def test_check_affordability_missing_price_counts_as_unaffordable():
    allocation = make_allocation()
    prices = {"EXPENSIVE": 17690.0, "MID": 500.0}  # CHEAP has no price
    result = check_affordability(allocation, 30000.0, prices)
    cheap_row = result[result["Symbol"] == "CHEAP"].iloc[0]
    assert cheap_row["Affordable"] == False
    assert cheap_row["Whole_Shares_Affordable"] == 0
