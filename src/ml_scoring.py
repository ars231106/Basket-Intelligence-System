import os
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.stock_scoring import score_stocks

FEATURE_COLUMNS = [
    "Mean_Return", "Volatility", "Momentum", "RSI", "ATR",
    "Average_Volume", "Volume_Growth", "Distance_SMA20", "Distance_SMA50", "Sharpe_Ratio",
]

# factories, not instances -- so train_all_models gets a fresh, unfitted model each time
MODEL_FACTORIES = {
    "DecisionTree": lambda: DecisionTreeRegressor(max_depth=4, min_samples_leaf=15, random_state=42),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=15, random_state=42),
    "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
}


def train_score_model(training_data, model_type="GradientBoosting", feature_columns=FEATURE_COLUMNS, label_column="Forward_Return"):
    X = training_data[feature_columns].values
    y = training_data[label_column].values
    model = MODEL_FACTORIES[model_type]()
    model.fit(X, y)
    return model


def train_all_models(training_data, feature_columns=FEATURE_COLUMNS, label_column="Forward_Return"):
    return {name: train_score_model(training_data, name, feature_columns, label_column) for name in MODEL_FACTORIES}


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def evaluate_model(model, test_data, feature_columns=FEATURE_COLUMNS, label_column="Forward_Return", n_quantiles=5):
    X = test_data[feature_columns].values
    y = test_data[label_column].values
    preds = model.predict(X)

    pearson = np.corrcoef(preds, y)[0, 1]
    spearman = pd.Series(preds).corr(pd.Series(y), method="spearman")

    # bucket by predicted score, check if top-predicted stocks actually returned more than bottom-predicted
    quantiles = pd.qcut(preds, n_quantiles, labels=False, duplicates="drop")
    quantile_returns = pd.Series(y).groupby(quantiles).mean()

    return {
        "n_test_rows": len(test_data),
        "pearson_correlation": pearson,
        "rank_correlation": spearman,
        "quantile_avg_forward_return": quantile_returns.to_dict(),
        "top_minus_bottom_quantile_spread": quantile_returns.iloc[-1] - quantile_returns.iloc[0],
    }


def score_stocks_ml(behaviour_dataset, communities_file, model, output_file, feature_columns=FEATURE_COLUMNS):
    features = pd.read_csv(behaviour_dataset)
    communities = pd.read_csv(communities_file)
    df = communities.merge(features, on="Symbol")

    # model output is a raw predicted forward return, not yet comparable across communities
    df["Stock_Score"] = model.predict(df[feature_columns].values)

    scored_rows = []
    for community, group in df.groupby("Community"):
        group = group.copy()
        lo, hi = group["Stock_Score"].min(), group["Stock_Score"].max()
        # same 0-1 rescaling convention as the hand-weighted scorer, so downstream code needs no changes
        group["Stock_Score"] = 0.5 if hi - lo < 1e-12 else (group["Stock_Score"] - lo) / (hi - lo)
        scored_rows.append(group)
        print(f"Processed Community {community} ({len(group)} stocks)")

    scored_df = pd.concat(scored_rows, ignore_index=True)
    scored_df = scored_df.sort_values(["Community", "Stock_Score"], ascending=[True, False])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    scored_df.to_csv(output_file, index=False)
    return scored_df


def score_stocks_ensemble(behaviour_dataset, communities_file, models, output_file, feature_columns=FEATURE_COLUMNS):
    # blends hardcoded weights + every model in `models` (e.g. train_all_models() output) into one averaged score
    hardcoded_file = output_file.replace(".csv", "_hardcoded_component.csv")
    hardcoded_df = score_stocks(behaviour_dataset, communities_file, hardcoded_file)
    hardcoded_df = hardcoded_df.rename(columns={"Stock_Score": "Score_Hardcoded"})

    features = pd.read_csv(behaviour_dataset)
    df = hardcoded_df[["Community", "Symbol", "Company", "Score_Hardcoded"]].merge(features, on="Symbol")

    for name, model in models.items():
        df[f"Score_{name}"] = model.predict(df[feature_columns].values)

    score_cols = ["Score_Hardcoded"] + [f"Score_{name}" for name in models]

    scored_rows = []
    for community, group in df.groupby("Community"):
        group = group.copy()
        for col in score_cols:
            lo, hi = group[col].min(), group[col].max()
            # rescale every component to 0-1 within the community before averaging, so no
            # single component dominates just because its raw numbers happen to be bigger
            group[col] = 0.5 if hi - lo < 1e-12 else (group[col] - lo) / (hi - lo)
        group["Stock_Score"] = group[score_cols].mean(axis=1)
        scored_rows.append(group)
        print(f"Processed Community {community} ({len(group)} stocks)")

    scored_df = pd.concat(scored_rows, ignore_index=True)
    scored_df = scored_df.sort_values(["Community", "Stock_Score"], ascending=[True, False])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    scored_df.to_csv(output_file, index=False)
    return scored_df
