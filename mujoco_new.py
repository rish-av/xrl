import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import d4rl
import gym
from torch.utils.data import Dataset, DataLoader
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from scipy.stats import entropy
import cv2




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.register_buffer('usage_count', torch.zeros(num_embeddings))
        self.register_buffer('perplexity_hist', torch.zeros(num_embeddings))
        self.total_samples = 0

    def get_codebook_metrics(self):
        """Calculate various metrics about codebook usage and distribution."""
        with torch.no_grad():
            # Calculate usage statistics
            total_usage = self.usage_count.sum().item()
            if total_usage == 0:
                return {
                    "active_codes": 0,
                    "perplexity": 0,
                    "usage_entropy": 0,
                    "dead_codes": self.num_embeddings,
                    "usage_histogram": self.usage_count.cpu().numpy()
                }
            
            # Calculate normalized usage
            normalized_usage = self.usage_count / total_usage
            
            # Calculate perplexity (effective codebook size)
            # High perplexity means codes are used more uniformly
            probs = normalized_usage[normalized_usage > 0]  # Only consider used codes
            perplexity = torch.exp(-torch.sum(probs * torch.log(probs)))
            
            # Calculate entropy of usage distribution
            # Higher entropy means more uniform usage
            entropy = -torch.sum(probs * torch.log2(probs))
            
            # Count active (used) and dead (unused) codes
            active_codes = (self.usage_count > 0).sum().item()
            dead_codes = (self.usage_count == 0).sum().item()
            
            # Calculate cosine similarities between codebook vectors
            normalized_embeddings = F.normalize(self.embeddings.weight, dim=1)
            similarities = torch.mm(normalized_embeddings, normalized_embeddings.t())
            
            # Get average similarity between different codes
            mask = torch.ones_like(similarities) - torch.eye(self.num_embeddings, device=similarities.device)
            avg_similarity = (similarities * mask).sum() / (mask.sum())
            
            return {
                "active_codes": active_codes,
                "dead_codes": dead_codes,
                "perplexity": perplexity.item(),
                "usage_entropy": entropy.item(),
                "avg_code_similarity": avg_similarity.item(),
                "most_used_codes": torch.topk(self.usage_count, k=5).indices.cpu().numpy(),
                "least_used_codes": torch.topk(self.usage_count, k=5, largest=False).indices.cpu().numpy(),
                "usage_histogram": self.usage_count.cpu().numpy(),
                "code_usage_pct": (active_codes / self.num_embeddings) * 100
            }

    def forward(self, inputs):
        # Convert inputs from BCL -> BLC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Track usage
        if self.training:
            self.usage_count += encodings.sum(0)
            self.total_samples += inputs.shape[0] * inputs.shape[1]
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()  # Straight through estimator
        
        # Convert quantized from BLC -> BCL
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        return quantized, loss, encoding_indices.view(input_shape[0], -1)

class TransformerVQVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_embeddings=512,
        embedding_dim=64,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    ):
        super().__init__()
        
        # Encoder
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # VQ Layer
        self.pre_vq_conv = nn.Linear(hidden_dim, embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.post_vq_conv = nn.Linear(embedding_dim, hidden_dim)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, src, src_mask=None):
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src, src_mask)
        z = self.pre_vq_conv(memory)
        quantized, vq_loss, indices = self.vq(z)
        return quantized, vq_loss, indices, self.post_vq_conv(quantized)

    def decode(self, memory, tgt, tgt_mask=None):
        tgt = self.input_proj(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory, tgt_mask)
        return self.output_proj(output)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        quantized, vq_loss, indices, memory = self.encode(src, src_mask)
        output = self.decode(memory, tgt, tgt_mask)
        return output, vq_loss, indices


class NonOverlappingTrajectoryDataset(Dataset):
    def __init__(self, env_name='halfcheetah-medium-v2', seq_length=50):
        self.env = gym.make(env_name)
        self.dataset = d4rl.qlearning_dataset(self.env)
        self.seq_length = seq_length
        
        # Combine observations, actions, and rewards
        self.trajectories = []
        current_traj = []
        
        for i in range(len(self.dataset['observations'])):
            obs = self.dataset['observations'][i]
            action = self.dataset['actions'][i]
            
            # Combine into single vector
            step_data = np.concatenate([obs, action])
            current_traj.append(step_data)
            
            if self.dataset['terminals'][i] or i == len(self.dataset['observations']) - 1:
                if len(current_traj) >= seq_length:
                    # Split into non-overlapping sequences of seq_length
                    for j in range(0, len(current_traj) - seq_length + 1, seq_length):
                        self.trajectories.append(current_traj[j:j + seq_length])
                current_traj = []

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = torch.FloatTensor(self.trajectories[idx])
        return traj




class TrajectoryDataset(Dataset):
    def __init__(self, env_name='halfcheetah-medium-v2', seq_length=50):
        self.env = gym.make(env_name)
        self.dataset = d4rl.qlearning_dataset(self.env)
        self.seq_length = seq_length
        
        # Combine observations, actions, and rewards
        self.trajectories = []
        current_traj = []
        
        for i in range(len(self.dataset['observations'])):
            obs = self.dataset['observations'][i]
            action = self.dataset['actions'][i]
            
            # Combine into single vector
            step_data = np.concatenate([obs, action])
            current_traj.append(step_data)
            
            if self.dataset['terminals'][i] or i == len(self.dataset['observations']) - 1:
                if len(current_traj) >= seq_length:
                    # Split into sequences of seq_length
                    for j in range(0, len(current_traj) - seq_length + 1):
                        self.trajectories.append(current_traj[j:j + seq_length])
                current_traj = []

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = torch.FloatTensor(self.trajectories[idx])
        return traj

def train_model(model, train_loader, num_epochs=20, learning_rate=1e-4, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            # Create causal mask for transformer
            seq_length = batch.size(1)
            causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
            causal_mask = causal_mask.to(device)
            
            # Forward pass
            output, vq_loss, indices = model(batch, batch, causal_mask, causal_mask)
            metrics = model.vq.get_codebook_metrics()
            # Calculate reconstruction loss
            recon_loss = F.mse_loss(output, batch)
            loss = recon_loss + vq_loss
            print("i: ", i, " ", metrics['active_codes'], metrics['perplexity'], metrics['avg_code_similarity'], loss)

            if loss.item() < 0.0006:
                break

            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if i == 5000:
                torch.save(model.state_dict(), 'vqvae_5000_mujoco.pth')
    
    torch.save(model.state_dict(), 'vqvae_t_mujoco.pth')
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.6f}")


def cluster_codebook_vectors(model, n_clusters=10, method='kmeans'):
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    
    # Get codebook vectors
    codebook = model.vq.embeddings.weight.detach().cpu().numpy()
    
    # Perform clustering based on selected method
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=0.5, min_samples=2)
    
    cluster_labels = clusterer.fit_predict(codebook)
    
    # Calculate clustering statistics
    cluster_stats = {
        'n_clusters': len(np.unique(cluster_labels)),
        'cluster_sizes': np.bincount(cluster_labels[cluster_labels >= 0]),
        'silhouette_score': silhouette_score(codebook, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
    }
    
    # Calculate cluster centers
    cluster_centers = np.zeros((cluster_stats['n_clusters'], codebook.shape[1]))
    for i in range(cluster_stats['n_clusters']):
        if method == 'dbscan' and i == -1:  # Skip noise points in DBSCAN
            continue
        cluster_centers[i] = codebook[cluster_labels == i].mean(axis=0)
    
    # Get codes in each cluster
    codes_per_cluster = {}
    for i in range(cluster_stats['n_clusters']):
        if method == 'dbscan' and i == -1:
            codes_per_cluster['noise'] = np.where(cluster_labels == -1)[0]
        else:
            codes_per_cluster[f'cluster_{i}'] = np.where(cluster_labels == i)[0]
            
    cluster_stats['codes_per_cluster'] = codes_per_cluster
    cluster_stats['cluster_centers'] = cluster_centers
    
    return cluster_labels, cluster_stats


def save_codebook_clusters(model, n_clusters=10, method='kmeans', save_path='clustering.png'):
    """
    Cluster and save visualization of codebook vectors clustering.
    
    Args:
        model: VQ-VAE model containing the codebook
        n_clusters: Number of clusters
        method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
        save_path: Path to save the visualization
    """
    from sklearn.decomposition import PCA
    
    # Get codebook and cluster it
    codebook = model.vq.embeddings.weight.detach().cpu().numpy()
    labels, stats = cluster_codebook_vectors(model, n_clusters, method)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    codebook_2d = pca.fit_transform(codebook)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(codebook_2d[:, 0], codebook_2d[:, 1], 
                         c=labels, cmap='tab10', s=100)
    
    # Add code indices as annotations
    for i, (x, y) in enumerate(codebook_2d):
        plt.annotate(f'{i}', (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    plt.title(f'Codebook Clusters ({method})\nSilhouette Score: {stats["silhouette_score"]:.3f}')
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_transition_graph(matrix: np.ndarray, path: str = "transition_graph.png"):
    """Save transition matrix as a directed graph visualization."""
    import networkx as nx
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges with weights for transitions above threshold
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i,j] > 0.1:  # Only show stronger transitions
                G.add_edge(f"Code {i}", f"Code {j}", weight=matrix[i,j])
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
    nx.draw_networkx_labels(G, pos)
    
    # Draw edges with varying thickness based on probability
    edges = G.edges()
    weights = [G[u][v]['weight'] * 2 for u,v in edges]
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', 
                          arrowsize=20)
    
    plt.title("Code Transition Graph")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()



def main():
    # Hyperparameters
    batch_size = 64
    seq_length = 60
    hidden_dim = 256
    num_embeddings = 128
    embedding_dim = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset and dataloader
    dataset = TrajectoryDataset(seq_length=seq_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Calculate input dimension (observation + action + reward)
    env = gym.make('halfcheetah-medium-v2')
    input_dim = env.observation_space.shape[0] + env.action_space.shape[0]
    
    # Create model
    model = TransformerVQVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim
    ).to(device)
    train_model(model, train_loader)
    
    return model


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
import os
import gym
import d4rl
import imageio
from tqdm import tqdm
import mujoco_py
import time

import numpy as np
import torch
from collections import defaultdict
import os
import gym
import d4rl
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans
import mujoco_py
import glfw

def save_video(frames, filename, fps=30):
    """Save frames as a video using OpenCV."""
    if len(frames) == 0:
        print(f"No frames to save for {filename}")
        return
        
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)
    
    video.release()

def set_mujoco_state(env, state):
    qpos = np.zeros(env.model.nq)  # Initialize with zeros
    qvel = np.zeros(env.model.nv)  # Initialize with zeros
    
    # Set joint angles (excluding root position)
    qpos[1:] = state[1:env.model.nq]  # Skip the root position
    
    # Set velocities (can optionally keep root velocity at 0)
    qvel[1:] = state[env.model.nq + 1:env.model.nq + env.model.nv]  # Skip the root velocity
    
    # Set the state
    env.set_state(qpos, qvel)

def analyze_cluster_behaviors(model, stats, save_dir="cluster_videos"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Analyze behaviors for each cluster
    behavior_analysis = {}
    
    for cluster_id, sequences in stats['cluster_raw_data'].items():
        sequences = np.array(sequences)
        
        # Calculate statistics
        action_sequences = [seq[:, -6:] for seq in sequences]  # Last 6 dimensions are actions
        
        analysis = {
            'n_sequences': len(sequences),
            'avg_action_magnitude': np.mean([np.mean(np.abs(act)) for act in action_sequences]),
            'action_std': np.mean([np.std(act) for act in action_sequences]),
            'most_active_joints': [np.argmax(np.mean(np.abs(act), axis=0)) for act in action_sequences],
            'sequence_length': sequences[0].shape[0] if len(sequences) > 0 else 0
        }
        
        behavior_analysis[cluster_id] = analysis
    
    # Save analysis
    with open(os.path.join(save_dir, 'behavior_analysis.txt'), 'w') as f:
        f.write("Half-Cheetah Behavior Analysis\n")
        f.write("============================\n\n")
        
        for cluster_id, analysis in behavior_analysis.items():
            f.write(f"Cluster {cluster_id}:\n")
            f.write(f"Number of sequences: {analysis['n_sequences']}\n")
            f.write(f"Average action magnitude: {analysis['avg_action_magnitude']:.3f}\n")
            f.write(f"Action standard deviation: {analysis['action_std']:.3f}\n")
            f.write(f"Most active joints: {analysis['most_active_joints']}\n")
            f.write(f"Sequence length: {analysis['sequence_length']}\n")
            f.write("\n")
    
    return behavior_analysis

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import numpy as np
import os
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS
import warnings
np.warnings = warnings

def create_clustered_transition_visualizations(model, dataloader, max_clusters=20, n_sequences=100, min_transitions=5, save_dir="cluster_analysis"):
    """Create visualizations of cluster transitions using XMeans for automatic cluster determination."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get codebook vectors
    codebook = model.vq.embeddings.weight.detach().cpu().numpy()
    
    # Initialize XMeans
    initial_centers = kmeans_plusplus_initializer(codebook, 2).initialize()
    xmeans_instance = xmeans(codebook, initial_centers, kmax=max_clusters)
    
    # Perform clustering
    xmeans_instance.process()
    
    # Get clusters and their centers
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()
    
    # Convert to cluster labels format
    n_clusters = len(clusters)
    cluster_labels = np.zeros(len(codebook), dtype=int)
    for i, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_labels[idx] = i
    
    # Calculate inter-cluster distances
    inter_cluster_distances = euclidean_distances(centers)
    
    # Calculate intra-cluster distances
    intra_cluster_distances = []
    for i in range(n_clusters):
        cluster_vectors = codebook[cluster_labels == i]
        if len(cluster_vectors) > 1:
            distances = euclidean_distances(cluster_vectors)
            avg_distance = (distances.sum() - distances.trace()) / (distances.size - len(cluster_vectors))
            intra_cluster_distances.append(avg_distance)
        else:
            intra_cluster_distances.append(0)
    
    # Count transitions between clusters
    cluster_transitions = defaultdict(int)
    code_to_cluster = {i: label for i, label in enumerate(cluster_labels)}
    
    # Process sequences to get transitions
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_sequences:
                break
                
            batch = batch.to(next(model.parameters()).device)
            _, _, indices, _ = model.encode(batch)
            
            for seq in indices:
                cluster_seq = [code_to_cluster[code.item()] for code in seq]
                for j in range(len(cluster_seq) - 1):
                    from_cluster = cluster_seq[j]
                    to_cluster = cluster_seq[j + 1]
                    if from_cluster != to_cluster:
                        cluster_transitions[(from_cluster, to_cluster)] += 1
    
    # Create two figures - one for weighted and one for unweighted
    for weighted in [True, False]:
        plt.figure(figsize=(15, 12))
        G = nx.DiGraph()
        
        # Normalize distances for layout
        max_dist = np.max(inter_cluster_distances)
        normalized_distances = inter_cluster_distances / max_dist
        
        # Use MDS for node positioning
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        pos_array = mds.fit_transform(normalized_distances)
        
        # Create position dictionary for nodes
        pos = {}
        max_intra_dist = max(intra_cluster_distances)
        normalized_sizes = [2000 * (1 - d/max_intra_dist) + 1000 for d in intra_cluster_distances]
        
        # Add nodes
        for i in range(n_clusters):
            cluster_size = sum(1 for label in cluster_labels if label == i)
            node_name = f"Cluster {i}\n({cluster_size} codes)\nIntra-dist: {intra_cluster_distances[i]:.2f}"
            G.add_node(node_name)
            pos[node_name] = pos_array[i]
        
        # Add edges
        for (from_cluster, to_cluster), count in cluster_transitions.items():
            if count >= min_transitions:
                from_name = f"Cluster {from_cluster}\n({sum(cluster_labels == from_cluster)} codes)\nIntra-dist: {intra_cluster_distances[from_cluster]:.2f}"
                to_name = f"Cluster {to_cluster}\n({sum(cluster_labels == to_cluster)} codes)\nIntra-dist: {intra_cluster_distances[to_cluster]:.2f}"
                G.add_edge(from_name, to_name, weight=count)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, 
                              node_color=[1-d/max_intra_dist for d in intra_cluster_distances],
                              node_size=normalized_sizes,
                              cmap=plt.cm.viridis,
                              alpha=0.7)
        
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        edges = G.edges()
        if weighted:
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1
            normalized_weights = [w/max_weight * 5 for w in weights]
            edge_width = normalized_weights
            # Draw edge labels (transition counts)
            edge_labels = {(u, v): G[u][v]['weight'] for u, v in edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        else:
            edge_width = [1] * len(edges)  # Uniform width for unweighted graph
        
        nx.draw_networkx_edges(G, pos, 
                              width=edge_width,
                              edge_color='black',
                              arrowsize=20,
                              alpha=0.6)
        
        graph_type = "Weighted" if weighted else "Unweighted"
        plt.title(f"Cluster Transition Graph - {graph_type} (XMeans: {n_clusters} clusters)\n(Node positions based on inter-cluster distances\nNode size inversely proportional to intra-cluster distance)")
        plt.savefig(os.path.join(save_dir, f"transition_graph_{graph_type.lower()}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create heatmap for inter-cluster distances
    plt.figure(figsize=(10, 8))
    sns.heatmap(inter_cluster_distances, 
                annot=True, 
                fmt='.2f', 
                cmap='viridis_r',
                xticklabels=[f'C{i}' for i in range(n_clusters)],
                yticklabels=[f'C{i}' for i in range(n_clusters)])
    plt.title('Inter-cluster Distances')
    plt.savefig(os.path.join(save_dir, "inter_cluster_distances.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create bar plot for intra-cluster distances
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_clusters), intra_cluster_distances)
    plt.xlabel('Cluster')
    plt.ylabel('Average Intra-cluster Distance')
    plt.title('Intra-cluster Distances')
    for i, v in enumerate(intra_cluster_distances):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.savefig(os.path.join(save_dir, "intra_cluster_distances.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'cluster_labels': cluster_labels,
        'transitions': dict(cluster_transitions),
        'n_codes_per_cluster': [sum(cluster_labels == i) for i in range(n_clusters)],
        'inter_cluster_distances': inter_cluster_distances,
        'intra_cluster_distances': intra_cluster_distances,
        'n_clusters': n_clusters
    }



from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize
from networkx.algorithms import community as nx_community



def multi_community_clustering(model, dataloader, 
                             method='louvain',  # 'louvain', 'girvan_newman', 'label_prop', 'fluid'
                             n_sequences=100,
                             similarity_weight=0.2,
                             edge_threshold=0.3,
                             resolution=2.0,
                             min_transitions=5,
                             save_dir="community_analysis"):
    """
    Cluster codebook vectors using various community detection algorithms.
    
    Args:
        model: VQ model with embeddings
        dataloader: Sequence dataloader
        method: Community detection method to use
        n_sequences: Number of sequences to analyze
        similarity_weight: Weight for similarity vs transitions
        edge_threshold: Minimum weight to create an edge
        resolution: Resolution parameter for community detection
        min_transitions: Minimum transitions for visualization
        save_dir: Output directory
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get codebook vectors
    codebook = model.vq.embeddings.weight.detach().cpu().numpy()
    codebook_size = len(codebook)
    
    # Count transitions
    code_transitions = defaultdict(int)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_sequences:
                break
            batch = batch.to(next(model.parameters()).device)
            _, _, indices, _ = model.encode(batch)
            for seq in indices:
                for j in range(len(seq) - 1):
                    code_transitions[(seq[j].item(), seq[j + 1].item())] += 1
    
    # Create and normalize transition matrix
    transitions = np.zeros((codebook_size, codebook_size))
    for (i, j), count in code_transitions.items():
        transitions[i, j] = count
    transitions = normalize(transitions, norm='l1', axis=1)
    
    # Calculate similarities using vector distances
    distances = euclidean_distances(codebook)
    max_dist = np.max(distances) if np.max(distances) > 0 else 1
    similarities = 1 - (distances / max_dist)
    
    # Create weighted graph
    G = nx.Graph()
    for i in range(codebook_size):
        G.add_node(i)
    
    # Add edges with combined weights
    for i in range(codebook_size):
        for j in range(i+1, codebook_size):
            trans_weight = (transitions[i,j] + transitions[j,i])/2
            sim_weight = similarities[i,j]
            weight = (1 - similarity_weight) * trans_weight + similarity_weight * sim_weight
            
            if weight > edge_threshold:
                G.add_edge(i, j, weight=weight)
    
    # Apply selected community detection method
    if method == 'louvain':
        communities = nx_community.louvain_communities(G, weight='weight', resolution=resolution)
    elif method == 'girvan_newman':
        # Get appropriate number of iterations based on graph size
        n_iter = min(int(np.sqrt(len(G.nodes))), 20)  # limit max iterations
        communities = list(nx_community.girvan_newman(G))[-n_iter]
    elif method == 'label_prop':
        communities = nx_community.label_propagation_communities(G)
    elif method == 'fluid':
        # Number of communities based on graph size
        k = max(int(np.sqrt(len(G.nodes)/2)), 3)
        communities = nx_community.asyn_fluidc(G, k)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    communities = list(communities)  # Convert to list if needed
    n_clusters = len(communities)
    
    # Convert to cluster labels
    cluster_labels = np.zeros(codebook_size, dtype=int)
    for i, comm in enumerate(communities):
        for node in comm:
            cluster_labels[node] = i
    
    # Calculate cluster centers and distances
    cluster_centers = []
    for comm in communities:
        comm_list = list(comm)
        center = np.mean(codebook[comm_list], axis=0)
        cluster_centers.append(center)
    cluster_centers = np.array(cluster_centers)
    
    inter_cluster_distances = euclidean_distances(cluster_centers)
    
    # Calculate coherence for each cluster
    intra_cluster_distances = []
    for comm in communities:
        comm_list = list(comm)
        if len(comm_list) > 1:
            comm_vectors = codebook[comm_list]
            distances = euclidean_distances(comm_vectors)
            avg_distance = distances.sum() / (len(comm_list) * (len(comm_list) - 1))
            intra_cluster_distances.append(1 - (avg_distance / max_dist))
        else:
            # For singletons, use nearest neighbors
            vector = codebook[comm_list[0]]
            dists = euclidean_distances([vector], codebook)[0]
            nearest_dists = np.sort(dists)[1:6]  # 5 nearest neighbors
            intra_cluster_distances.append(1 - (nearest_dists.mean() / max_dist))
    
    # Create visualization
    plt.figure(figsize=(20, 16))
    
    # Use MDS for layout
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pos_array = mds.fit_transform(inter_cluster_distances)
    
    # Create cluster graph
    CG = nx.DiGraph()
    
    # Create node names
    node_names = []
    for i, comm in enumerate(communities):
        node_name = f"Cluster {i}\n({len(comm)} codes)\nCoherence: {intra_cluster_distances[i]:.2f}"
        node_names.append(node_name)
        CG.add_node(node_name)
    
    # Create positions
    pos = {node_names[i]: pos_array[i] for i in range(n_clusters)}
    
    # Add edges
    edge_labels = {}
    for (from_code, to_code), count in code_transitions.items():
        if count >= min_transitions:
            from_cluster = cluster_labels[from_code]
            to_cluster = cluster_labels[to_code]
            if from_cluster != to_cluster:
                from_name = node_names[from_cluster]
                to_name = node_names[to_cluster]
                if CG.has_edge(from_name, to_name):
                    CG[from_name][to_name]['weight'] += count
                else:
                    CG.add_edge(from_name, to_name, weight=count)
                edge_labels[(from_name, to_name)] = count
    
    # Draw nodes
    node_sizes = [len(comm) * 100 for comm in communities]
    nx.draw_networkx_nodes(CG, pos, 
                          node_color=intra_cluster_distances,
                          node_size=node_sizes,
                          cmap=plt.cm.viridis,
                          alpha=0.7)
    
    nx.draw_networkx_labels(CG, pos, font_size=8)
    
    # Draw edges
    edges = CG.edges()
    if edges:
        weights = [CG[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        normalized_weights = [w/max_weight * 3 for w in weights]
        
        nx.draw_networkx_edges(CG, pos, 
                              width=normalized_weights,
                              edge_color='black',
                              arrowsize=15,
                              alpha=0.6)
        
        nx.draw_networkx_edge_labels(CG, pos, edge_labels=edge_labels, font_size=6)
    
    plt.title(f"Community Transition Graph ({method})\n{n_clusters} communities found\n" + 
             f"(similarity_weight={similarity_weight}, edge_threshold={edge_threshold})")
    
    plt.margins(0.2)
    plt.axis('off')
    plt.tight_layout(pad=2.0)
    
    plt.savefig(os.path.join(save_dir, f"community_transition_graph_{method}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'cluster_labels': cluster_labels,
        'communities': communities,
        'transitions': dict(code_transitions),
        'inter_cluster_distances': inter_cluster_distances,
        'intra_cluster_distances': intra_cluster_distances,
        'n_clusters': n_clusters,
        'graph': G,
        'node_names': node_names,
        'method': method
    }



def dbscan_behavior_clustering(model, dataloader, n_sequences=1000, eps=0.3, min_samples=3, 
                             min_transitions=5, save_dir="dbscan_analysis"):
    """
    Cluster codebook vectors using DBSCAN with combined behavior and transition affinity.
    
    Args:
        model: VQ model with embeddings
        dataloader: Data loader for sequences
        n_sequences: Number of sequences to analyze
        eps: Maximum distance between samples for DBSCAN
        min_samples: Minimum samples for DBSCAN core point
        min_transitions: Minimum transitions for visualization
        save_dir: Directory to save visualizations
    """
    from sklearn.cluster import DBSCAN

    os.makedirs(save_dir, exist_ok=True)
    
    # Get codebook vectors
    codebook = model.vq.embeddings.weight.detach().cpu().numpy()
    codebook_size = len(codebook)
    
    # Count transitions
    code_transitions = defaultdict(int)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_sequences:
                break
            batch = batch.to(next(model.parameters()).device)
            _, _, indices, _ = model.encode(batch)
            for seq in indices:
                for j in range(len(seq) - 1):
                    code_transitions[(seq[j].item(), seq[j + 1].item())] += 1
    
    # Create transition matrix
    transitions = np.zeros((codebook_size, codebook_size))
    for (i, j), count in code_transitions.items():
        transitions[i, j] = count
    
    # Create behavior similarity matrix (normalized to [0,1])
    behavior_sim = 1 / (1 + euclidean_distances(codebook) ** 2)  # Squared distances for sharper falloff
    behavior_sim = behavior_sim / behavior_sim.max()
    
    # Create transition similarity matrix (normalized to [0,1])
    transition_sim = transitions + transitions.T  # Make symmetric
    if transition_sim.max() > 0:
        # Remove weak transitions
        transition_sim[transition_sim < transition_sim.max() * 0.05] = 0
        transition_sim = transition_sim / transition_sim.max()
    
    # Create combined affinity matrix
    affinity = np.multiply(behavior_sim, transition_sim)
    
    # Convert affinity to distance (required for DBSCAN)
    distances = 1 - affinity
    
    # Apply DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    cluster_labels = clustering.fit_predict(distances)
    
    # Handle noise points (-1 labels) by assigning them to nearest cluster
    noise_points = np.where(cluster_labels == -1)[0]
    if len(noise_points) > 0:
        for point in noise_points:
            # Find nearest non-noise point
            dist_to_points = distances[point]
            valid_points = np.where(cluster_labels != -1)[0]
            if len(valid_points) > 0:
                nearest = valid_points[np.argmin(dist_to_points[valid_points])]
                cluster_labels[point] = cluster_labels[nearest]
    
    n_clusters = len(set(cluster_labels))
    communities = [np.where(cluster_labels == i)[0] for i in range(n_clusters)]
    
    # Calculate cluster centers and distances
    cluster_centers = np.array([codebook[comm].mean(axis=0) for comm in communities])
    inter_cluster_distances = euclidean_distances(cluster_centers)
    
    # Calculate cluster coherence
    intra_cluster_distances = []
    for comm in communities:
        if len(comm) > 1:
            comm_vectors = codebook[comm]
            distances = euclidean_distances(comm_vectors)
            avg_distance = distances.sum() / (len(comm) * (len(comm) - 1))
            intra_cluster_distances.append(1 - avg_distance/distances.max())
        else:
            vector = codebook[comm[0]]
            dists = euclidean_distances([vector], codebook)[0]
            nearest_dists = np.sort(dists)[1:6]
            intra_cluster_distances.append(1 - nearest_dists.mean()/dists.max())
    
    # Create visualization
    plt.figure(figsize=(30, 20))
    
    # Use MDS for initial layout
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pos_array = mds.fit_transform(inter_cluster_distances)
    
    # Create cluster graph
    CG = nx.DiGraph()
    
    # Create node names
    node_names = []
    for i, comm in enumerate(communities):
        node_name = f"Cluster {i}\n({len(comm)} codes)\nCoherence: {intra_cluster_distances[i]:.2f}"
        node_names.append(node_name)
        CG.add_node(node_name)
    
    # Create position dictionary with added jitter
    pos = {}
    for i, name in enumerate(node_names):
        x, y = pos_array[i]
        # Add small random offset
        pos[name] = (x + np.random.uniform(-0.1, 0.1),
                    y + np.random.uniform(-0.1, 0.1))
    
    # Add edges between clusters
    edge_labels = {}
    for (from_code, to_code), count in code_transitions.items():
        if count >= min_transitions:
            from_cluster = cluster_labels[from_code]
            to_cluster = cluster_labels[to_code]
            if from_cluster != to_cluster:
                from_name = node_names[from_cluster]
                to_name = node_names[to_cluster]
                if CG.has_edge(from_name, to_name):
                    CG[from_name][to_name]['weight'] += count
                else:
                    CG.add_edge(from_name, to_name, weight=count)
                edge_labels[(from_name, to_name)] = count
    
    # Draw the graph
    node_sizes = [len(comm) * 150 for comm in communities]  # Increased base size
    nx.draw_networkx_nodes(CG, pos, 
                          node_color=intra_cluster_distances,
                          node_size=node_sizes,
                          cmap=plt.cm.viridis,
                          alpha=0.7)
    
    nx.draw_networkx_labels(CG, pos, font_size=10)
    
    edges = CG.edges()
    if edges:
        weights = [CG[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        normalized_weights = [w/max_weight * 3 for w in weights]
        
        nx.draw_networkx_edges(CG, pos, 
                              width=normalized_weights,
                              edge_color='black',
                              arrowsize=20,
                              alpha=0.6,
                              connectionstyle="arc3,rad=0.2")  # Curved edges
        
        nx.draw_networkx_edge_labels(CG, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(f"DBSCAN Behavior Clusters\n{n_clusters} clusters found\n" + 
             f"(eps={eps}, min_samples={min_samples})")
    
    plt.margins(0.3)  # Increased margins
    plt.axis('off')
    plt.tight_layout(pad=2.0)
    
    plt.savefig(os.path.join(save_dir, "behavior_clusters.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'cluster_labels': cluster_labels,
        'communities': communities,
        'transitions': dict(code_transitions),
        'inter_cluster_distances': inter_cluster_distances,
        'intra_cluster_distances': intra_cluster_distances,
        'n_clusters': n_clusters,
        'affinity': affinity,
        'node_names': node_names
    }


from sklearn.cluster import SpectralClustering
from sklearn.manifold import MDS
from scipy.sparse import csr_matrix
import scipy.sparse as sp

def create_better_affinity(codebook_vectors, transition_counts):
    """Create improved affinity matrix"""
    # More sensitive behavior similarity
    distances = euclidean_distances(codebook_vectors)
    sigma = np.median(distances)  # Adaptive bandwidth
    behavior_sim = np.exp(-distances**2 / (2 * sigma**2))
    
    # Stronger transition weighting
    transition_sim = transition_counts + transition_counts.T
    transition_sim[transition_sim < np.percentile(transition_sim, 90)] = 0  # Keep only top 10%
    transition_sim = transition_sim / transition_sim.max()
    
    # Combine with geometric mean
    affinity = np.sqrt(np.multiply(behavior_sim, transition_sim))
    
    # Apply threshold
    affinity[affinity < 0.1] = 0
    
    return (affinity + affinity.T) / 2

def normalized_laplacian_matrix(affinity):
    """Compute normalized Laplacian"""
    A = csr_matrix(affinity)
    n_nodes = A.shape[0]
    diags = A.sum(axis=1).flatten()
    D = sp.spdiags(diags, [0], n_nodes, n_nodes, format='csr')
    L = D - A
    Dinv = sp.spdiags(1/diags, [0], n_nodes, n_nodes, format='csr')
    return Dinv.dot(L)

def estimate_n_clusters(affinity):
    """Estimate number of clusters from eigenvalues"""
    L = normalized_laplacian_matrix(affinity)
    eigenvals = np.sort(np.abs(np.linalg.eigvals(L.toarray())))  # Using numpy's eigvals
    gaps = np.diff(eigenvals[:20])
    n_clusters = np.argmax(gaps) + 1
    return max(2, min(n_clusters, int(np.sqrt(len(affinity)))))

def create_better_layout(G):
    """Create improved graph layout"""
    pos = nx.spring_layout(G, k=2, iterations=50)
    pos = nx.kamada_kawai_layout(G, pos=pos)
    
    # Add repulsion
    for _ in range(10):
        for node in pos:
            dx = dy = 0
            for other in pos:
                if other != node:
                    x1, y1 = pos[node]
                    x2, y2 = pos[other]
                    dx += 0.1 / (x1 - x2 + 1e-6)  # Added epsilon to avoid division by zero
                    dy += 0.1 / (y1 - y2 + 1e-6)
            pos[node] = (pos[node][0] - dx, pos[node][1] - dy)
    
    return pos

def improved_spectral_clustering(model, dataloader, n_sequences=1000, min_transitions=5, save_dir="spectral_analysis"):
    """Main clustering function using improved methods"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get codebook vectors
    codebook = model.vq.embeddings.weight.detach().cpu().numpy()
    codebook_size = len(codebook)
    
    # Count transitions
    code_transitions = defaultdict(int)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_sequences:
                break
            batch = batch.to(next(model.parameters()).device)
            _, _, indices, _ = model.encode(batch)
            for seq in indices:
                for j in range(len(seq) - 1):
                    code_transitions[(seq[j].item(), seq[j + 1].item())] += 1
    
    # Create transition matrix
    transitions = np.zeros((codebook_size, codebook_size))
    for (i, j), count in code_transitions.items():
        transitions[i, j] = count
    
    # Use improved affinity matrix
    affinity = create_better_affinity(codebook, transitions)
    
    # Estimate optimal number of clusters
    n_clusters = estimate_n_clusters(affinity)
    
    # Apply improved spectral clustering
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        n_init=50,
        assign_labels='kmeans',
        random_state=42
    )
    cluster_labels = clustering.fit_predict(affinity)
    
    # Get cluster info
    communities = [np.where(cluster_labels == i)[0] for i in range(n_clusters)]
    
    # Calculate cluster centers and distances
    cluster_centers = np.array([codebook[comm].mean(axis=0) for comm in communities])
    inter_cluster_distances = euclidean_distances(cluster_centers)
    
    # Create visualization
    plt.figure(figsize=(30, 20))
    
    # Create cluster graph
    CG = nx.DiGraph()
    
    # Add nodes
    node_names = []
    for i, comm in enumerate(communities):
        node_name = f"Cluster {i}\n({len(comm)} codes)"
        node_names.append(node_name)
        CG.add_node(node_name)
    
    # Add edges
    edge_labels = {}
    for (from_code, to_code), count in code_transitions.items():
        if count >= min_transitions:
            from_cluster = cluster_labels[from_code]
            to_cluster = cluster_labels[to_code]
            if from_cluster != to_cluster:
                from_name = node_names[from_cluster]
                to_name = node_names[to_cluster]
                if CG.has_edge(from_name, to_name):
                    CG[from_name][to_name]['weight'] += count
                else:
                    CG.add_edge(from_name, to_name, weight=count)
                edge_labels[(from_name, to_name)] = count
    
    # Use improved layout
    pos = create_better_layout(CG)
    
    # Draw graph
    node_sizes = [len(comm) * 150 for comm in communities]
    nx.draw_networkx_nodes(CG, pos, 
                          node_size=node_sizes,
                          node_color=range(len(communities)),
                          cmap=plt.cm.viridis,
                          alpha=0.7)
    
    nx.draw_networkx_labels(CG, pos, font_size=10)
    
    edges = CG.edges()
    if edges:
        weights = [CG[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        normalized_weights = [w/max_weight * 3 for w in weights]
        
        nx.draw_networkx_edges(CG, pos, 
                              width=normalized_weights,
                              edge_color='black',
                              arrowsize=20,
                              alpha=0.6,
                              connectionstyle="arc3,rad=0.2")
        
        nx.draw_networkx_edge_labels(CG, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(f"Improved Spectral Clustering\n{n_clusters} clusters found")
    plt.margins(0.3)
    plt.axis('off')
    plt.tight_layout(pad=2.0)
    
    plt.savefig(os.path.join(save_dir, "spectral_clusters.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'cluster_labels': cluster_labels,
        'communities': communities,
        'transitions': dict(code_transitions),
        'affinity': affinity,
        'n_clusters': n_clusters
    }

if __name__ == "__main__":
    batch_size = 1
    seq_length = 60
    hidden_dim = 256
    num_embeddings = 128
    embedding_dim = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('halfcheetah-medium-v2')
    input_dim = env.observation_space.shape[0] + env.action_space.shape[0]


    dataset = NonOverlappingTrajectoryDataset(seq_length=seq_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = TransformerVQVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim
    ).to(device)


    model.load_state_dict(torch.load('vqvae_t_mujoco.pth', weights_only=True))
    # out = mcl_transition_clustering(model, train_loader, inflation=2.0, n_sequences=10000, sigma=0.05, min_transitions=3, save_dir="mcl_analysis")
    # out = community_transition_clustering(model, train_loader, n_sequences=1000, similarity_weight=0.25, edge_threshold=0.1, min_transitions=3, save_dir="community_analysis")
    
    out = improved_spectral_clustering(model, train_loader, n_sequences=1000, min_transitions=3, save_dir="spectral_analysis")
    print(out['inter_cluster_distances'])
    # print(out['inter_cluster_distances'])
    # cluster_labels = out['cluster_labels']
    # labels = {i:cluster_labels[i] for i in range(len(cluster_labels))}
    # with torch.no_grad():
    #     for i, batch in enumerate(train_loader):
    #         batch = batch.to(device)
    #         _, _, indices, _ = model.encode(batch)
            
    #         for seq in indices:
    #             c = []
    #             for j in range(len(seq) - 1):
    #                 c.append(labels[seq[j].item()])
    #             # print(c)
    #         if i == 10:
    #             break
    # create_clustered_transition_visualizations(model, train_loader, max_clusters=20, n_sequences=10000, min_transitions=3)
    # First render the videos for each cluster
    # stats = render_halfcheetah_sequences(
    #     model=model,
    #     dataloader=train_loader,
    #     n_clusters=10,
    #     n_sequences=1000,
    #     n_samples_per_cluster=5,
    #     save_dir="halfcheetah_cluster_videos"
    # )

    # # Then analyze the behaviors
    # behavior_analysis = analyze_cluster_behaviors(
    #     model=model,
    #     stats=stats,
    #     save_dir="halfcheetah_cluster_videos"
    # )
