import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_behaviour_dataset(input_file):
    df = pd.read_csv(input_file)
    symbols = df["Symbol"]
    features = df.drop(columns="Symbol")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    return symbols, scaled_df

