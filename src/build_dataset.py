import os
import pandas as pd

from src.behaviour_vector import build_behaviour_vector


def build_dataset(data_folder="data", output_file="features/behaviour_dataset.csv"):
    rows = []

    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"{data_folder} not found.")

    # download_data.py sanitises filesystem-unsafe symbols (a '/' in the
    # ticker, or a Windows-reserved device name) when saving; this map
    # recovers the true original symbol so it still matches the universe
    # CSV's Symbol column downstream (community/basket name lookups, etc).
    symbol_map = {}
    map_path = os.path.join(data_folder, "_symbol_filename_map.csv")
    if os.path.exists(map_path):
        map_df = pd.read_csv(map_path)
        symbol_map = dict(zip(map_df["Filename"], map_df["Symbol"]))

    for file in os.listdir(data_folder):

        if not file.endswith(".csv") or file == "_symbol_filename_map.csv":
            continue

        file_path = os.path.join(data_folder, file)

        try:
            df = pd.read_csv(file_path)
            filename_stem = file.replace(".csv", "")
            symbol = symbol_map.get(filename_stem, filename_stem)
            row = build_behaviour_vector(df, symbol)
            rows.append(row)
            print(f"Processed -> {symbol}")

        except Exception as e:
            print(f"Error Proccessing {file}: {e}")

    dataset = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dataset.to_csv(output_file, index=False)

    print("\n===================================")
    print("DATASET SUMMARY")
    print("===================================")
    print(f"Stocks Processed : {len(dataset)}")
    print(f"Saved To         : {output_file}")

    return dataset
