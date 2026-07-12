import pandas as pd
import os

def save_communities(baskets, name_lookup, output_file):
    rows = []

    for community, stocks in baskets.items():
        for stock in stocks:
            rows.append({"Community" : community, "Symbol" : stock, "Company" : name_lookup.get(stock, "Unknown")})

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    return df        

