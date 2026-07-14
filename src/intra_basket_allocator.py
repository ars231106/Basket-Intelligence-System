import pandas as pd

# A basket's own allocated capital is split among ITS member stocks by one
# of these strategies -- a separate, independent choice from which of the
# six strategies (Equal Weight ... Black-Litterman) splits capital ACROSS
# baskets. Every function here takes a DataFrame of one basket's member
# rows (must include a "Symbol" column, plus whatever feature columns that
# particular strategy needs) and returns a pandas Series of weights,
# indexed by Symbol, that sum to 1.0 -- the caller then multiplies by that
# basket's own share of total capital.

# Small floor added before normalising in every strategy below so that no
# single stock in a basket ever gets driven all the way to a literal zero
# weight, which would otherwise be possible for whichever stock scores
# lowest on a given signal in a small basket. A zero weight is a silent
# "this stock is never actually held", which is a stronger claim than any
# of these heuristics are meant to make.
_FLOOR = 0.10


def _minmax(series):
    """
    Rescale a feature to 0..1 within the basket, so unlike-scaled signals
    (a 0..1 range position vs. a small-percentage momentum figure) can be
    combined without one dominating purely because of its raw magnitude.
    Returns a flat 0.5 for every member if the basket has no spread on
    this feature at all (e.g. a 1-stock basket, or a genuine tie).
    """
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _normalise(raw_scores, symbols):
    weights = raw_scores + _FLOOR
    weights = weights.clip(lower=0)
    total = weights.sum()
    if total <= 0:
        # degenerate case (shouldn't happen given the floor above, but
        # fall back to equal weight rather than divide by zero)
        return pd.Series(1.0 / len(symbols), index=symbols)
    return weights / total


def allocate_equal(members_df):
    """The original behaviour: every member of the basket gets the same share."""
    symbols = members_df["Symbol"]
    n = len(symbols)
    return pd.Series(1.0 / n, index=symbols)


def allocate_range_position(members_df):
    """
    Heaviest weight to stocks sitting near the middle of their own 52-week
    trading range, tapering off toward both edges -- but tapering off
    *faster* toward the 52-week high than toward the 52-week low, so the
    resulting preference order is Mid > Low > High, matching the original
    idea this strategy is built from. Worth knowing going in: this runs
    against the George & Hwang (2004) 52-week-high momentum finding, which
    found nearness to the 52-week high historically predicts continued
    outperformance rather than reversal -- see allocate_high_momentum
    below for the evidence-backed alternative.
    """
    symbols = members_df["Symbol"]
    position = members_df["Range_Position_52W"]

    dist_to_high = (position - 0.5).clip(lower=0) / 0.5   # 0 at the mid, 1 at the high
    dist_to_low = (0.5 - position).clip(lower=0) / 0.5    # 0 at the mid, 1 at the low

    HIGH_PENALTY = 0.8   # steeper taper toward the high
    LOW_PENALTY = 0.5    # gentler taper toward the low

    raw = 1.0 - HIGH_PENALTY * dist_to_high - LOW_PENALTY * dist_to_low
    raw = pd.Series(raw.to_numpy(), index=symbols)
    return _normalise(raw, symbols)


def allocate_high_momentum(members_df):
    """
    The evidence-backed counterpart to allocate_range_position: heaviest
    weight to stocks close to their 52-week high AND showing strong
    trailing momentum, combining both signals (George & Hwang 2004 found
    nearness-to-52-week-high improves on momentum alone, rather than the
    two being redundant).
    """
    symbols = members_df["Symbol"]
    range_score = _minmax(members_df["Range_Position_52W"])
    momentum_score = _minmax(members_df["Momentum"])

    raw = 0.5 * range_score + 0.5 * momentum_score
    raw = pd.Series(raw.to_numpy(), index=symbols)
    return _normalise(raw, symbols)


def allocate_inverse_volatility_intra(members_df):
    """
    Same logic as the existing basket-level Inverse Volatility strategy in
    portfolio_allocator.py, applied one level down: calmer stocks within a
    basket get more of that basket's capital, volatile ones get less.
    Grounded in the low-volatility anomaly (Ang, Hodrick, Xing & Zhang,
    2006), which found low-volatility stocks have historically delivered
    better risk-adjusted returns than a simple risk/return tradeoff would
    predict.
    """
    symbols = members_df["Symbol"]
    inverse_vol = 1 / (members_df["Volatility"] + 1e-9)
    weights = pd.Series(inverse_vol.to_numpy(), index=symbols)
    return weights / weights.sum()


def allocate_volume_conditioned_momentum(members_df):
    """
    Weights toward momentum winners, but discounts winners that have also
    seen an unusually large recent jump in trading volume -- Lee &
    Swaminathan (1998) found low-volume momentum winners keep
    outperforming for longer, while high-volume winners tend to reverse
    faster, on the theory that a lot of the move is already "used up" /
    priced in by the time volume has spiked.
    """
    symbols = members_df["Symbol"]
    momentum_score = _minmax(members_df["Momentum"])
    volume_penalty = _minmax(members_df["Volume_Growth"])

    raw = momentum_score - 0.5 * volume_penalty
    raw = raw - raw.min()  # shift so the worst-scoring member is at 0, not negative
    raw = pd.Series(raw.to_numpy(), index=symbols)
    return _normalise(raw, symbols)


INTRA_BASKET_STRATEGIES = {
    "equal_weighted": allocate_equal,
    "range_position": allocate_range_position,
    "high_momentum": allocate_high_momentum,
    "inverse_volatility_intra": allocate_inverse_volatility_intra,
    "volume_conditioned_momentum": allocate_volume_conditioned_momentum,
}


def allocate_within_basket(members_df, strategy):
    """
    Dispatch to one of the strategies above for a single basket's member
    rows. A 1-stock basket always gets 100% regardless of strategy -- none
    of these signals mean anything with only one stock to compare against.
    """
    symbols = members_df["Symbol"]
    if len(symbols) == 1:
        return pd.Series([1.0], index=symbols)

    func = INTRA_BASKET_STRATEGIES.get(strategy)
    if func is None:
        raise ValueError(
            f"Unknown within-basket allocation strategy: {strategy!r}. "
            f"Choose one of: {sorted(INTRA_BASKET_STRATEGIES)}"
        )
    return func(members_df)


def basket_weights_to_stock_weights_v2(basket_df, basket_ids, weights, strategy="equal_weighted"):
    """
    Turn basket-level weights into stock-level weights using the chosen
    within-basket strategy. basket_df must contain "Basket_ID" and
    "Symbol" for every strategy, plus "Range_Position_52W" / "Momentum" /
    "Volatility" / "Volume_Growth" as needed by whichever strategy is
    selected (all produced automatically by feature_engineering.py's
    FEATURES dict, so any dataset built via build_dataset /
    build_trailing_behaviour_dataset already has them).

    This supersedes the old flat weight/len(members) split that used to
    live directly in backtest_engine.py -- passing strategy="equal_weighted"
    (the default) reproduces that exact original behaviour.
    """
    stock_weights = {}
    for basket_id, basket_weight in zip(basket_ids, weights):
        members_df = basket_df.loc[basket_df["Basket_ID"] == basket_id].reset_index(drop=True)
        if members_df.empty:
            continue

        within_basket_weights = allocate_within_basket(members_df, strategy)

        for symbol, w in within_basket_weights.items():
            stock_weights[symbol] = stock_weights.get(symbol, 0) + basket_weight * w

    return stock_weights
