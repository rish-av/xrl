import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


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
    plt.savefig('cluster_graph.png')


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



import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt



#parition based on threshold
class AdaptiveGraphPartitioner:
    def __init__(self, distance_threshold=None, transition_threshold=None):
        self.distance_threshold = distance_threshold
        self.transition_threshold = transition_threshold
        
    def fit_transform(self, node_vectors, transition_matrix):
        distances = euclidean_distances(node_vectors)
        
        if self.distance_threshold is None:
            self.distance_threshold = np.percentile(distances[distances > 0], 25)
            
        if self.transition_threshold is None:
            nonzero_transitions = transition_matrix[transition_matrix > 0]
            self.transition_threshold = np.percentile(nonzero_transitions, 75)
        
        n_nodes = len(node_vectors)
        adjacency = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if (distances[i, j] <= self.distance_threshold and 
                    (transition_matrix[i, j] >= self.transition_threshold or 
                     transition_matrix[j, i] >= self.transition_threshold)):
                    adjacency[i, j] = adjacency[j, i] = 1
                    
        sparse_adj = csr_matrix(adjacency)
        n_components, labels = connected_components(sparse_adj, directed=False)
        return labels
    
    def get_cluster_metrics(self, node_vectors, transition_matrix, labels):
        unique_labels = np.unique(labels)
        metrics = {
            'n_clusters': len(unique_labels),
            'cluster_sizes': [],
            'avg_intra_cluster_distance': [],
            'avg_inter_cluster_transitions': []
        }
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)
            metrics['cluster_sizes'].append(cluster_size)
            
            cluster_vectors = node_vectors[cluster_mask]
            if cluster_size > 1:
                intra_distances = euclidean_distances(cluster_vectors)
                metrics['avg_intra_cluster_distance'].append(
                    np.mean(intra_distances[np.triu_indices(cluster_size, k=1)])
                )
            
            cluster_transitions = transition_matrix[cluster_mask][:, cluster_mask]
            metrics['avg_inter_cluster_transitions'].append(
                np.mean(cluster_transitions[cluster_transitions > 0])
                if np.any(cluster_transitions > 0) else 0
            )
            
        return metrics

    def render_graph(self, node_vectors, transition_matrix, labels, figsize=(12, 8)):
        G = nx.Graph()
        n_nodes = len(node_vectors)
        
        for i in range(n_nodes):
            G.add_node(i, pos=node_vectors[i][:2] if node_vectors[i].shape[0] > 2 else node_vectors[i])
            
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if (transition_matrix[i, j] >= self.transition_threshold or 
                    transition_matrix[j, i] >= self.transition_threshold):
                    weight = max(transition_matrix[i, j], transition_matrix[j, i])
                    G.add_edge(i, j, weight=weight)

        plt.figure(figsize=figsize)
        pos = nx.get_node_attributes(G, 'pos')
        
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights)
        edge_weights_normalized = [w / max_weight * 3 for w in edge_weights]
        
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: color for label, color in zip(unique_labels, colors)}
        node_colors = [color_map[label] for label in labels]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200)
        nx.draw_networkx_edges(G, pos, width=edge_weights_normalized, alpha=0.7)
        nx.draw_networkx_labels(G, pos)
        
        plt.title(f'Graph Partitioning Result\nClusters: {len(unique_labels)}')
        plt.axis('equal')
        plt.show()
        plt.savefig('graph_partitioning.png')

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

class SpectralGraphPartitioner:
    def __init__(self, n_clusters=None, scaling='standard', 
                 similarity_mode='combined', alpha=0.5):
        self.n_clusters = n_clusters
        self.scaling = scaling
        self.similarity_mode = similarity_mode
        self.alpha = alpha
        self.fitted_scalers = {}
        
    def _scale_matrix(self, matrix, name):
        if self.scaling == 'none':
            return matrix
            
        flat_matrix = matrix.flatten().reshape(-1, 1)
        if self.scaling == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
            
        self.fitted_scalers[name] = scaler
        scaled_flat = scaler.fit_transform(flat_matrix)
        return scaled_flat.reshape(matrix.shape)
    
    def _compute_distance_similarity(self, node_vectors):
        distances = euclidean_distances(node_vectors)
        sigma = np.mean(distances)
        if sigma <= 0:
            sigma = 1e-10
        similarity = np.exp(-distances / (2 * sigma ** 2))
        return self._scale_matrix(similarity, 'distance')

    def _compute_transition_similarity(self, transition_matrix):
        max_count = transition_matrix.max()
        if max_count == 0:
            similarity = np.zeros_like(transition_matrix)
        else:
            similarity = np.maximum(
                (transition_matrix + transition_matrix.T) / (2 * max_count),
                0
            )
        return self._scale_matrix(similarity, 'transition')

    def _compute_combined_similarity(self, distance_sim, transition_sim):
        similarity = self.alpha * distance_sim + (1 - self.alpha) * transition_sim
        return np.maximum(similarity, 0)


    def _similarity_threshold(self, similarity):
        flat_similarity = similarity[np.triu_indices_from(similarity, k=1)]
        flat_similarity = flat_similarity[flat_similarity > 0]
        threshold = np.percentile(flat_similarity, 90)  # Adjust percentile as needed
        
        print(f"threshold determined: {threshold}")
        return threshold

    def _estimate_n_clusters(self, similarity_matrix):
        eigenvals = np.linalg.eigvals(similarity_matrix)
        sorted_eigenvals = np.sort(np.abs(eigenvals))[::-1]
        gaps = np.diff(sorted_eigenvals)
        normalized_gaps = gaps / sorted_eigenvals[:-1]
        return np.argmax(normalized_gaps[1:])
    
    def fit_transform(self, node_vectors, transition_matrix):
        distance_sim = self._compute_distance_similarity(node_vectors)
        transition_sim = self._compute_transition_similarity(transition_matrix)
        similarity = self._compute_combined_similarity(distance_sim, transition_sim)

        if not np.allclose(similarity, similarity.T):
            similarity = (similarity + similarity.T) / 2

        if np.min(similarity) < 0:
            raise ValueError("Similarity matrix contains negative values.")

        if np.any(np.sum(similarity, axis=1) == 0):
            raise ValueError("Rows with all zeros in the similarity matrix.")

        similarity /= np.max(similarity)

        # Regularization: Add a small value to the diagonal of the Laplacian
        laplacian = np.diag(np.sum(similarity, axis=1)) - similarity
        regularization_term = 1e-5  # Small value for regularization
        laplacian += np.eye(laplacian.shape[0]) * regularization_term

        if self.n_clusters is None:
            self.n_clusters = self._estimate_n_clusters(laplacian)

        print(f"Estimated number of clusters: {self.n_clusters}")

        spectral = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            random_state=42
        )
        return spectral.fit_predict(similarity)
    
    def visualize_similarity(self, node_vectors, transition_matrix):
        distance_sim = self._compute_distance_similarity(node_vectors)
        transition_sim = self._compute_transition_similarity(transition_matrix)
        combined_sim = self._compute_combined_similarity(distance_sim, transition_sim)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        imgs = [
            axes[0].imshow(distance_sim, origin='lower', aspect='auto', cmap='viridis'),
            axes[1].imshow(transition_sim, origin='lower', aspect='auto', cmap='viridis'),
            axes[2].imshow(combined_sim, origin='lower', aspect='auto', cmap='viridis')
        ]
        
        titles = ['Distance Similarity', 'Transition Similarity', 'Combined Similarity']
        
        for ax, img, title in zip(axes, imgs, titles):
            ax.set_title(title)
            ax.set_xlabel("Indices")
            ax.set_ylabel("Indices")
            plt.colorbar(img, ax=ax)
        
        plt.tight_layout()
        plt.show()
        plt.savefig('similarity_matrices.png')

        
    def get_cluster_metrics(self, node_vectors, transition_matrix, labels):
        metrics = {}
        unique_labels = np.unique(labels)
        
        distance_sim = self._compute_distance_similarity(node_vectors)
        transition_sim = self._compute_transition_similarity(transition_matrix)
        
        metrics['n_clusters'] = len(unique_labels)
        metrics['cluster_sizes'] = []
        metrics['spatial_cohesion'] = []
        metrics['temporal_cohesion'] = []
        
        for label in unique_labels:
            mask = labels == label
            size = np.sum(mask)
            metrics['cluster_sizes'].append(size)
            
            if size > 1:
                cluster_distance = distance_sim[np.ix_(mask, mask)]
                cluster_transitions = transition_sim[np.ix_(mask, mask)]
                
                metrics['spatial_cohesion'].append(
                    np.mean(cluster_distance[np.triu_indices(size, k=1)])
                )
                metrics['temporal_cohesion'].append(
                    np.mean(cluster_transitions[np.triu_indices(size, k=1)])
                )
        
        return metrics

    def build_transition_matrix(self, matrix):
        unique_elements = sorted(set(element for row in matrix for element in row))
        element_index = {element: i for i, element in enumerate(unique_elements)}
        size = len(unique_elements)
        transition_matrix = np.zeros((size, size), dtype=int)
        for row in matrix:
            for i in range(len(row) - 1):
                from_idx = element_index[row[i]]
                to_idx = element_index[row[i + 1]]
                transition_matrix[from_idx][to_idx] += 1
        return transition_matrix, unique_elements


    def _merge_small_close_clusters(self, node_vectors, labels, similarity_threshold=0.8, size_threshold=10, min_size=2):
        unique_labels = np.unique(labels)
        cluster_centroids = {}
        cluster_sizes = {}

        # Calculate centroids and sizes for each cluster
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_centroids[label] = node_vectors[cluster_indices].mean(axis=0)
            cluster_sizes[label] = len(cluster_indices)

        centroids = np.array(list(cluster_centroids.values()))
        similarities = cosine_similarity(centroids)

        # Merge small clusters based on similarity
        for i, label_i in enumerate(unique_labels):
            for j, label_j in enumerate(unique_labels):
                if i >= j:  # Avoid self-comparison
                    continue

                # Merge if highly similar and both clusters are small
                if similarities[i, j] > similarity_threshold and cluster_sizes[label_i] < size_threshold and cluster_sizes[label_j] < size_threshold:
                    labels[labels == label_j] = label_i
                    cluster_sizes[label_i] += cluster_sizes[label_j]
                    cluster_sizes[label_j] = 0  # Mark as merged

        # **New Step: Merge Very Small Clusters**
        for label in unique_labels:
            if cluster_sizes[label] < min_size and cluster_sizes[label] > 0:  # If it's a small but valid cluster
                # Find the most similar larger cluster
                best_match = None
                max_sim = -1

                for other_label in unique_labels:
                    if cluster_sizes[other_label] >= min_size:  # Merge only with large enough clusters
                        sim = cosine_similarity(
                            [cluster_centroids[label]], [cluster_centroids[other_label]]
                        )[0, 0]
                        if sim > max_sim:
                            max_sim = sim
                            best_match = other_label

                if best_match is not None:
                    labels[labels == label] = best_match
                    cluster_sizes[best_match] += cluster_sizes[label]
                    cluster_sizes[label] = 0  # Mark as merged

        # Relabel clusters to be sequential
        unique_labels = np.unique(labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        labels = np.vectorize(label_mapping.get)(labels)

        return labels



    def render_partitioned_graph(self, node_vectors, transition_matrix, labels, 
                             figsize=(12, 8), edge_threshold=0.2, 
                             node_size=200, with_labels=True, 
                             combine_nodes=False, show_cluster_nodes=True, min_edge_thickness=2.0):
        distance_sim = self._compute_distance_similarity(node_vectors)
        transition_sim = self._compute_transition_similarity(transition_matrix)
        similarity = self._compute_combined_similarity(distance_sim, transition_sim)
        


        # nodes = [1,3,4,5,6,7,8,9,10,11,12,13,14,15]
        # node_dict = {i: nodes[i] for i in range(len(nodes))}
        if show_cluster_nodes:
            cluster_nodes = {label: np.where(labels == label)[0].tolist() for label in np.unique(labels)}
            print("Nodes in each cluster:")
            for cluster, nodes in cluster_nodes.items():
                print(f"Cluster {cluster}: {[node  for node in nodes]}")
        
        if combine_nodes:
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            # Initialize the combined graph
            G_combined = nx.Graph()
            
            # Add super-nodes to the combined graph
            for cluster_idx in range(n_clusters):
                G_combined.add_node(cluster_idx, size=np.sum(labels == unique_labels[cluster_idx]))
            
            # Compute weights between super-nodes based on similarity and transition counts
            edge_labels = {}
            for i, label_i in enumerate(unique_labels):
                for j, label_j in enumerate(unique_labels):
                    if i < j:  # Avoid duplicate computation
                        nodes_i = np.where(labels == label_i)[0]
                        nodes_j = np.where(labels == label_j)[0]
                        weight = np.sum(similarity[np.ix_(nodes_i, nodes_j)])
                        transition_count = np.sum(transition_matrix[np.ix_(nodes_i, nodes_j)])
                        if weight > edge_threshold:  # Add edge only if weight exceeds threshold
                            if transition_count > 0:
                                G_combined.add_edge(i, j, weight=transition_count)
                                edge_labels[(i, j)] = f'{int(transition_count)}'
            
            # Normalize edge weights for visualization
            edge_weights = [G_combined[u][v]['weight'] for u, v in G_combined.edges()]
            edge_weights_normalized = [max(w / max(edge_weights) * 5, min_edge_thickness) for w in edge_weights] if edge_weights else []

            # Assign unique colors to clusters
            colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
            
            # Create cluster labels with the number of nodes
            node_labels = {cluster_idx: f'Cluster {unique_labels[cluster_idx]}\n({G_combined.nodes[cluster_idx]["size"]} nodes)'
                        for cluster_idx in range(n_clusters)}

            # Draw the combined graph
            plt.figure(figsize=figsize)
            nx.draw_networkx_nodes(
                G_combined, nx.circular_layout(G_combined), node_size=[G_combined.nodes[node]['size'] * node_size for node in G_combined.nodes()],
                node_color=colors, alpha=0.7
            )
            nx.draw_networkx_edges(
                G_combined, nx.circular_layout(G_combined), width=edge_weights_normalized, edge_color="black", alpha=0.7
            )
            nx.draw_networkx_labels(
                G_combined, nx.circular_layout(G_combined), labels=node_labels, font_size=10
            )
            nx.draw_networkx_edge_labels(
                G_combined, nx.circular_layout(G_combined), edge_labels=edge_labels, font_size=8
            )

            plt.title(f'Combined Graph with {n_clusters} Clusters')
            plt.axis('equal')
            plt.tight_layout()
            plt.show()
            plt.savefig('combined_graph_atari.png')
        else:
            # Render the original graph with individual nodes
            G = nx.Graph()
            n_nodes = len(node_vectors)
            
            # Add nodes with positions
            for i in range(n_nodes):
                pos = node_vectors[i][:2] if node_vectors[i].shape[0] > 2 else node_vectors[i]
                G.add_node(i, pos=pos)
            
            # Add edges based on similarity
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    weight = similarity[i, j]
                    if weight > edge_threshold:  # Add edge only if weight > threshold
                        G.add_edge(i, j, weight=weight)
            
            plt.figure(figsize=figsize)
            pos = nx.get_node_attributes(G, 'pos')
            
            # Set up colors for clusters
            unique_labels = np.unique(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            color_map = {label: color for label, color in zip(unique_labels, colors)}
            node_colors = [color_map[label] for label in labels]
            
            # Draw edges with varying thickness based on weight
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            edge_weights_normalized = [max(w / max(edge_weights) * 3, min_edge_thickness) for w in edge_weights] if edge_weights else []
            
            # Draw the graph
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size)
            nx.draw_networkx_edges(G, pos, width=edge_weights_normalized, alpha=0.5, edge_color="black")
            
            if with_labels:
                nx.draw_networkx_labels(G, pos)
            
            # Add title and cluster info
            plt.title(f'Spectral Clustering Result\nClusters: {len(unique_labels)}')
            
            # Add cluster legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color_map[label], 
                                        label=f'Cluster {label}', 
                                        markersize=10)
                            for label in unique_labels]
            plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
            
            
            plt.axis('equal')
            plt.tight_layout()
            plt.show()
            plt.savefig('spectral_clustering.png')







