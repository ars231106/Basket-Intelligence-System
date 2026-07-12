import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_matrix(symbols, scaled_features):
    similarity = cosine_similarity(scaled_features)
    similarity_df = pd.DataFrame(similarity, index=symbols, columns=symbols)
    return similarity_df

