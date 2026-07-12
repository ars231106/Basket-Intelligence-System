from src.feature_engineering import compute_features

def build_behaviour_vector(df, symbol):
    features = {"Symbol": symbol}
    features.update(compute_features(df))
    return features