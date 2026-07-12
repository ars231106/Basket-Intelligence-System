import os
import pandas as pd

def load_universe(name):
    path = f"universes/{name.lower()}.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    
    df = pd.read_csv(path)
    
    stocks = df["Symbol"].astype(str).str.strip().to_list()
    return stocks