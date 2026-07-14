import pandas as pd

from src.backtest_dataset import load_price_series

# NSE/BSE (unlike a lot of US brokers) don't support fractional shares --
# every position has to be bought as a whole number of shares. A stock's
# target Capital_Distribution from the rest of this pipeline is a
# continuous dollar figure with no such constraint, so a small enough
# total capital can produce target allocations smaller than the price of
# a single share of the stock they're meant to buy. This module answers
# two practical questions the rest of the pipeline doesn't: "how much
# capital would this specific selection actually need to be executable?"
# and "given the capital I actually have, which of these positions can't
# be bought at all?"


def load_latest_prices(symbols, data_folder):
    """
    Latest available closing price for each symbol, read from the same
    downloaded price files (data/{universe}/*.csv) the rest of the
    pipeline already uses -- reuses load_price_series so filename
    sanitisation (Windows-reserved names, '/' in LSE tickers, etc.) stays
    consistent with everywhere else it's handled. A symbol with no price
    file on disk is simply left out of the result rather than raising.
    """
    prices = {}
    for symbol in symbols:
        try:
            df = load_price_series(symbol, data_folder)
            if not df.empty:
                prices[symbol] = df["Close"].iloc[-1]
        except Exception:
            continue
    return prices


def compute_minimum_viable_capital(stock_allocation_df, capital, price_lookup):
    """
    For every stock in stock_allocation_df, the minimum TOTAL capital that
    would let it be bought as at least one whole share at (approximately)
    its own target weight. This is price / weight, not price alone --
    a cheap stock with a tiny target weight can genuinely need MORE total
    capital to clear this bar than an expensive stock with a large weight,
    since the target weight has to stretch far enough to cover one share.

    Returns a DataFrame sorted by that requirement, descending, so the
    single stock that sets the floor for the whole portfolio is always
    the first row -- that row's value is the minimum viable capital for
    this exact selection and these exact target weights.
    """
    rows = []
    for _, row in stock_allocation_df.iterrows():
        symbol = row["Symbol"]
        price = price_lookup.get(symbol)
        if price is None or price <= 0:
            continue

        weight = row["Capital_Distribution"] / capital if capital > 0 else 0
        if weight <= 0:
            continue

        required_capital = price / weight
        rows.append({
            "Symbol": symbol,
            "Company": row.get("Company", ""),
            "Basket_ID": row.get("Basket_ID", ""),
            "Price": price,
            "Target_Weight": weight,
            "Required_Capital_For_1_Share": required_capital,
        })

    columns = ["Symbol", "Company", "Basket_ID", "Price", "Target_Weight", "Required_Capital_For_1_Share"]
    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows)[columns]
    return result.sort_values("Required_Capital_For_1_Share", ascending=False).reset_index(drop=True)


def check_affordability(stock_allocation_df, capital, price_lookup):
    """
    Given an actual, real capital amount, checks each selected stock's OWN
    target dollar allocation against its share price and reports how many
    whole shares that specific slice of capital can actually buy.
    Anything at 0 shares is a stock that would sit as unspent cash at this
    capital level -- the real-world "cash drag" this whole module exists
    to surface before a backtest or a live decision, not after.
    """
    rows = []
    for _, row in stock_allocation_df.iterrows():
        symbol = row["Symbol"]
        price = price_lookup.get(symbol)
        allocated = row["Capital_Distribution"]

        if price is None or price <= 0:
            shares = 0
        else:
            shares = int(allocated // price)

        rows.append({
            "Symbol": symbol,
            "Company": row.get("Company", ""),
            "Basket_ID": row.get("Basket_ID", ""),
            "Price": price,
            "Allocated_Capital": allocated,
            "Whole_Shares_Affordable": shares,
            "Affordable": shares >= 1,
        })

    columns = ["Symbol", "Company", "Basket_ID", "Price", "Allocated_Capital",
               "Whole_Shares_Affordable", "Affordable"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]
