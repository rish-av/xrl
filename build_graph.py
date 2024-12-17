import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def build_cluster_graph(codebook_embeddings, transitions, num_clusters=10):
    """
    Clusters codebook embeddings into `num_clusters` and then builds a cluster-level
    directed graph based on transitions.

    Parameters:
    - codebook_embeddings: torch.Tensor or np.ndarray of shape [num_embeddings, embedding_dim]
      The embeddings from the VQ codebook.
    - transitions: list of tuples (i, j) indicating a transition from code index i to code index j.
                   For example, if you have encoded sequences of states into code indices,
                   you can produce such transitions from consecutive steps: (code_i -> code_j).
    - num_clusters: int, number of clusters for K-Means.

    Returns:
    - G: A networkx.DiGraph representing transitions between clusters.
    - cluster_labels: An array of shape [num_embeddings] with the cluster assignment for each embedding.
    """

    # Convert codebook to numpy if needed
    if isinstance(codebook_embeddings, torch.Tensor):
        codebook_embeddings = codebook_embeddings.detach().cpu().numpy()

    # Step 1: Cluster the embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(codebook_embeddings)

    # Step 2: Build a cluster graph
    G = nx.DiGraph()

    # Add nodes for each cluster
    for c in range(num_clusters):
        G.add_node(c, label=f'Cluster {c}')

    # Step 3: For each transition (code_i -> code_j), find the clusters (c_i -> c_j)
    # If c_i != c_j (transition between different clusters), add an edge
    for (code_i, code_j) in transitions:
        c_i = cluster_labels[code_i]
        c_j = cluster_labels[code_j]
        if c_i != c_j:
            if G.has_edge(c_i, c_j):
                G[c_i][c_j]['weight'] += 1
            else:
                G.add_edge(c_i, c_j, weight=1)

    return G, cluster_labels

def draw_cluster_graph(G):
    """
    Draw the cluster graph created by build_cluster_graph.
    Node size proportional to out-degree, edge width proportional to edge weight.
    """

    # Compute node sizes based on out-degree
    out_degrees = dict(G.out_degree(weight='weight'))
    max_out_degree = max(out_degrees.values()) if out_degrees else 1

    # Positions for nodes in a circular layout
    pos = nx.circular_layout(G)

    # Draw nodes, size scaled by out-degree
    node_sizes = [3000 * (out_degrees[n]/max_out_degree) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue')

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels={n:G.nodes[n]['label'] for n in G.nodes()}, font_size=10)

    # Draw edges with width proportional to weight
    edges = G.edges(data='weight', default=1)
    max_weight = max((d for (_,_,d) in edges), default=1)
    edges = G.edges(data='weight', default=1)  # Re-iterate
    edge_widths = [2.0 * (w / max_weight) for (_,_,w) in edges]

    nx.draw_networkx_edges(G, pos, width=edge_widths, arrowstyle='->', arrowsize=15, edge_color='gray')

    plt.axis('off')
    plt.title("Cluster Transition Graph")
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Suppose we have codebook embeddings and transitions
    # codebook_embeddings: [num_embeddings, embedding_dim]
    num_embeddings = 50
    embedding_dim = 16
    codebook_embeddings = torch.randn(num_embeddings, embedding_dim)

    # Create some dummy transitions. For instance, a sequence of length L can produce L-1 transitions.
    # Let's say we have sequences of code indices: e.g. [0, 5, 10, 3, 5 ...]
    # Here we'll just create random transitions between code indices:
    transitions = [(i, (i+1) % num_embeddings) for i in range(num_embeddings)]

    G, cluster_labels = build_cluster_graph(codebook_embeddings, transitions, num_clusters=5)
    draw_cluster_graph(G)
