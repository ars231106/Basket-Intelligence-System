import os
import pandas as pd

from src.feature_engineering import compute_features
from src.download_data import sanitize_filename


def load_price_series(symbol, data_folder):
    path = os.path.join(data_folder, f"{sanitize_filename(symbol)}.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def generate_rebalance_dates(start_date, end_date, freq="MS"):
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def build_walk_forward_dataset(symbols, data_folder, start_date, end_date, freq="MS", min_history=260):
    rebalance_dates = generate_rebalance_dates(start_date, end_date, freq)
    rows = []

    for symbol in symbols:
        try:
            price_df = load_price_series(symbol, data_folder)
        except Exception as e:
            print(f"Skipping {symbol}: {e}")
            continue

        for i in range(len(rebalance_dates) - 1):
            t = rebalance_dates[i]
            t_next = rebalance_dates[i + 1]

            # only data up to t is visible -- this is what keeps the dataset point-in-time
            trailing = price_df[price_df["Date"] <= t]
            if len(trailing) < min_history:
                continue

            future = price_df[price_df["Date"] >= t_next]
            if future.empty:
                continue

            entry_price = trailing["Close"].iloc[-1]
            exit_price = future["Close"].iloc[0]
            forward_return = (exit_price / entry_price) - 1

            features = compute_features(trailing)
            if any(pd.isna(v) for v in features.values()):
                continue

            rows.append({"Symbol": symbol, "Date": t, **features, "Forward_Return": forward_return})

    return pd.DataFrame(rows)


def split_train_test(df, cutoff_date):
    # split by date, not randomly -- test rows must all be after every train row
    cutoff_date = pd.to_datetime(cutoff_date)
    train = df[df["Date"] < cutoff_date].reset_index(drop=True)
    test = df[df["Date"] >= cutoff_date].reset_index(drop=True)
    return train, test


def save_dataset(df, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved walk-forward dataset -> {output_file} ({len(df)} rows)")
