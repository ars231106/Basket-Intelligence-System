import os
import pandas as pd
import math

def generate_candidate_baskets(stock_scores_files, similarity_matrix_file, output_file, min_size , max_size):
    df = pd.read_csv(stock_scores_files)
    baskets = []

    for community, group in df.groupby("Community"):
        group = group.sort_values("Stock_Score", ascending=False).reset_index(drop=True)
        total_stocks = len(group)

        if total_stocks < min_size:
            basket = group.copy()
            basket["Basket_ID"] = f"C{community}_B1"

            baskets.append(basket)
            continue

        num_baskets = math.ceil(total_stocks / max_size)

        while total_stocks / num_baskets < min_size:
            num_baskets -= 1

        base_size = total_stocks // num_baskets
        remainder = total_stocks % num_baskets

        start = 0

        for basket_no in range(num_baskets):
            basket_size = base_size
            if basket_no < remainder:
                basket_size += 1
            basket = group.iloc[start:start + basket_size].copy()
            basket["Basket_ID"] = f"C{community}_B{basket_no + 1}"
            baskets.append(basket)
            start = start + basket_size
        
    basket_df = pd.concat(baskets, ignore_index=True)
    
    basket_df = basket_df[["Basket_ID", "Community", "Symbol", "Company", "Stock_Score"]]
    basket_df = basket_df.sort_values(["Community", "Basket_ID", "Stock_Score"], ascending=[True, True, False])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    basket_df.to_csv(output_file, index=False)
    return basket_df