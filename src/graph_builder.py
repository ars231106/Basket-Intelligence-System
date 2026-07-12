import networkx as nx
import math

def build_similarity_graph(symbols, similarity_matrix, k=None):
    if k is None:
        k = max(5, round(math.sqrt(len(symbols))))

    graph = nx.Graph()

    for symbol in symbols:
        graph.add_node(symbol)

    n = len(symbols)

    for i in range(n):
        similarities = similarity_matrix.iloc[i].copy()
        similarities.iloc[i] = -1
        closest = similarities.nlargest(k)

        for neighbour_index, similarity in closest.items():
            graph.add_edge(symbols.iloc[i], neighbour_index, weight = similarity)

    return graph