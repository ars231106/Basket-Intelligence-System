import os
import math
import networkx as nx
import matplotlib.pyplot as plt

def plot_similarity_graph(graph, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(28,28))

    pos = nx.spring_layout(graph, seed=42, k=2/math.sqrt(graph.number_of_nodes()), iterations=200)

    nx.draw_networkx_nodes(graph, pos, node_size=250, node_color="dodgerblue")

    nx.draw_networkx_edges(graph, pos, alpha=0.5, edge_color="gray", width = 0.8)

    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold")

    plt.title("Similarity Graph (k-nearest neighbours)", fontsize = 18)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches = "tight")
    plt.close()

def plot_communities_graph(graph, communities, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(28,28), facecolor="white")

    pos = nx.spring_layout(graph, seed=42, k=2/math.sqrt(graph.number_of_nodes()), iterations=200)

    unique_communities = sorted(set(communities.values()))

    cmap = plt.colormaps["tab10"]

    node_colors = [cmap(unique_communities.index(communities[node]) / max(1, len(unique_communities) - 1)) for node in graph.nodes()]

    nx.draw_networkx_nodes(graph, pos, node_size=250, node_color=node_colors)

    nx.draw_networkx_edges(graph, pos, alpha=0.5, edge_color="gray", width = 0.8)

    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold")

    plt.title("Behaviour Communities (Louvain)", fontsize = 18)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(output_path, dpi = 300, bbox_inches = "tight")
    plt.close()