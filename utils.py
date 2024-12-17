import torch
import os
import gym
import numpy as np
import cv2
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import d4rl 
import argparse
import torch
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import d4rl_atari
import gzip
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import mujoco
import numpy as np
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import networkx as nx
from typing import List, Tuple, Dict
import os
from scipy.spatial.distance import euclidean







OBJECT_TO_IDX = {
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,
    'key'           : 5,
    'ball'          : 6,
    'box'           : 7,
    'goal'          : 8,
    'lava'          : 9,
    'agent'         : 10,
}

COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5
}


import argparse


import numpy as np

def build_transition_matrix(all_indices, output_file='adj.json'):
    # Find the maximum index to determine matrix size
    max_index = max(max(sequence) for sequence in all_indices)
    size = max_index + 1

    # Initialize the transition matrix
    transition_matrix = np.zeros((size, size))

    # Build the transition matrix
    for sequence in all_indices:
        for i in range(len(sequence) - 1):
            current_idx = sequence[i]
            next_idx = sequence[i + 1]
            transition_matrix[current_idx][next_idx] += 1

    # Normalize the transition matrix
    row_sums = transition_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_matrix / row_sums[:, np.newaxis]

    # Convert the transition matrix to an adjacency list
    adjacency_list = {}
    for i, row in enumerate(transition_matrix):
        adjacency_list[i] = {j: weight for j, weight in enumerate(row) if weight > 0}

    # Save the adjacency list as a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(adjacency_list, json_file, indent=4)

    return transition_matrix

def create_similarity_matrix(codebook_vectors):
    """
    Create similarity matrix based on euclidean distances between codebook vectors
    """
    n = len(codebook_vectors)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = euclidean(codebook_vectors[i], codebook_vectors[j])
                similarity_matrix[i, j] = distance  # Convert distance to similarity
    
    # Normalize rows
    row_sums = similarity_matrix.sum(axis=1)
    similarity_matrix = similarity_matrix / row_sums[:, np.newaxis]
    return similarity_matrix

def mcl_iteration(matrix, expansion_power, inflation_power):
    """
    Perform one iteration of MCL algorithm
    """
    # Expansion step (matrix multiplication)
    expanded = np.linalg.matrix_power(matrix, expansion_power)
    
    # Inflation step (Hadamard power followed by normalization)
    inflated = np.power(expanded, inflation_power)
    
    # Normalize columns
    col_sums = inflated.sum(axis=0)
    inflated = inflated / col_sums[np.newaxis, :]
    
    return inflated

def mcl_graph_cut(codebook_vectors, transition_matrix, min_clusters=10, 
                  expansion_power=2, initial_inflation_power=2, 
                  max_inflation_power=20, inflation_step=0.5, 
                  max_iterations=100, convergence_threshold=1e-4):
    """
    Perform MCL clustering on codebook vectors considering transition weights.
    Ensures a minimum of 'min_clusters' clusters.
    """
    # Create initial similarity matrix based on Euclidean distances
    similarity_matrix = create_similarity_matrix(codebook_vectors)
    
    # Combine with transition matrix (element-wise multiplication)
    combined_matrix = similarity_matrix 
        
    # Normalize the combined matrix
    row_sums = combined_matrix.sum(axis=1)
    matrix = combined_matrix / row_sums[:, np.newaxis]
    
    inflation_power = initial_inflation_power
    clusters = []
    
    while True:
        current_matrix = matrix.copy()
        prev_matrix = None
        
        # MCL iteration
        for _ in range(max_iterations):
            # Expansion step
            matrix = np.linalg.matrix_power(current_matrix, expansion_power)
            
            # Inflation step
            matrix = np.power(matrix, inflation_power)
            # Normalize
            row_sums = matrix.sum(axis=1)
            matrix = matrix / row_sums[:, np.newaxis]
            
            # Check for convergence
            if prev_matrix is not None:
                diff = np.abs(matrix - prev_matrix).max()
                if diff < convergence_threshold:
                    break
                    
            prev_matrix = matrix.copy()
            current_matrix = matrix.copy()
        
        # Extract clusters from final matrix
        n = len(codebook_vectors)
        clusters = []
        for i in range(n):
            # Assign node to cluster with highest probability
            cluster = np.argmax(matrix[i])
            clusters.append(cluster)
        
        # Relabel clusters to be consecutive integers starting from 0
        unique_clusters = sorted(set(clusters))
        num_clusters = len(unique_clusters)
        
        if num_clusters >= min_clusters:
            # Enough clusters found, stop the loop
            cluster_map = {c: i for i, c in enumerate(unique_clusters)}
            clusters = [cluster_map[c] for c in clusters]
            break
        else:
            # Increase inflation power to get more clusters
            inflation_power += inflation_step
            if inflation_power > max_inflation_power:
                # Prevent infinite loop
                print("Unable to achieve the minimum number of clusters with reasonable inflation power.")
                cluster_map = {c: i for i, c in enumerate(unique_clusters)}
                clusters = [cluster_map[c] for c in clusters]
                break
            # Reset matrix to initial state
            matrix = combined_matrix / combined_matrix.sum(axis=1)[:, np.newaxis]
    
    return clusters

def visualize_clusters(codebook_vectors, clusters, transition_matrix=None):
    """
    Visualize the clustering results
    """
    # Convert codebook vectors to numpy array
    vectors = np.array(codebook_vectors)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Scatter plot of clusters
    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(vectors[:, 0], vectors[:, 1], c=clusters, cmap='tab10')
    ax1.set_title('Cluster Assignments')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    
    # Plot 2: Transition matrix heatmap
    ax2 = fig.add_subplot(132)
    if transition_matrix is not None:
        im = ax2.imshow(transition_matrix, cmap='viridis')
        plt.colorbar(im, ax=ax2, label='Transition Weight')
    ax2.set_title('Transition Matrix')
    ax2.set_xlabel('Node Index')
    ax2.set_ylabel('Node Index')
    
    # Plot 3: Graph visualization
    ax3 = fig.add_subplot(133)
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            if transition_matrix is not None and transition_matrix[i,j] > 0.1:
                ax3.plot([vectors[i,0], vectors[j,0]], 
                        [vectors[i,1], vectors[j,1]], 
                        'gray', alpha=transition_matrix[i,j])
    scatter = ax3.scatter(vectors[:, 0], vectors[:, 1], c=clusters, cmap='tab10')
    ax3.set_title('Graph Structure')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('mcl_cut.png')
    return fig


def get_args_atari():
    parser = argparse.ArgumentParser(description="Arguments for Atari D4RL Transformer Model")
    
    # Environment and Dataset
    parser.add_argument('--env_name', type=str, default='seaquest-medium-v2', help='Name of the Atari environment')
    
    # Model Parameters
    parser.add_argument('--model_dim', type=int, default=256, help='Dimensionality of model embeddings')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads in the Transformer')
    parser.add_argument('--num_encoder_layers', type=int, default=4, help='Number of layers in the Transformer encoder')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of layers in the Transformer decoder')
    parser.add_argument('--num_embeddings', type=int, default=128, help='Number of embeddings for the VQ layer')

    # Training Parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    
    # Checkpoints
    parser.add_argument('--model_save_path', type=str, default='bexrl_atari_model.pth', help='Path to save the trained model')
    parser.add_argument('--model_load_path', type=str, default='', help='Path to load a pre-trained model (leave empty if not loading)')

    # WandB Logging
    parser.add_argument('--log', action='store_true', help='Enable logging to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='xrlwithbehaviors', help="Wandb project name")
    parser.add_argument('--wandb_run_name', type=str, default='xrlwithbehavior', help="Wandb run name")
    parser.add_argument('--wandb_entity', type=str, default='mail-rishav9', help="Wandb entity name")

    # Miscellaneous
    parser.add_argument('--train', action='store_true', help='Set this flag to train the model')
    parser.add_argument('--model_type', type=str, choices=['sequence', 'state'], default='sequence', help='Model type: sequence or state')
    
    args = parser.parse_args()
    return args




def get_args_mujoco():
    parser = argparse.ArgumentParser(description="Transformer Model Parameters")
    parser.add_argument('--env_name', type=str, default='halfcheetah-medium-v2', help="Mujoco environment name")
    parser.add_argument('--model_dim', type=int, default=128, help="Transformer model dimension")
    parser.add_argument('--num_heads', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--num_encoder_layers', type=int, default=4, help="Number of encoder layers")
    parser.add_argument('--num_decoder_layers', type=int, default=4, help="Number of decoder layers")
    parser.add_argument('--num_embeddings', type=int, default=64, help="VQ-VAE codebook size")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--model_save_path', type=str, default='model_mujoco.pth', help="Path to save the model")
    parser.add_argument('--model_load_path', type=str, default='', help="Path to load the model")
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--segment_length', type=int, default=25, help="Segment length for behavior isolation")
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='xrlwithbehaviors', help="Wandb project name")
    parser.add_argument('--wandb_run_name', type=str, default='xrlwithbehavior', help="Wandb run name")
    parser.add_argument('--wandb_entity', type=str, default='mail-rishav9', help="Wandb entity name")
    parser.add_argument('--model_type', type=str, default='sequence', help="Model type: one id for each state or sequence")


    args = parser.parse_args()
    return args



def get_args_minigrid():
    parser = argparse.ArgumentParser(description="Transformer Model Parameters")
    parser.add_argument('--env_name', type=str, default='minigrid-fourrooms-v0', help="Minigrid env name")
    parser.add_argument('--model_dim', type=int, default=256, help="Transformer model dimension")
    parser.add_argument('--num_heads', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--num_encoder_layers', type=int, default=4, help="Number of encoder layers")
    parser.add_argument('--num_decoder_layers', type=int, default=4, help="Number of decoder layers")
    parser.add_argument('--num_embeddings', type=int, default=64, help="VQ-VAE codebook size")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--model_save_path', type=str, default='model_t.pth', help="Path to save the model")
    parser.add_argument('--model_load_path', type=str, default='', help="Path to load the model")
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='xrlwithbehaviors', help="Wandb project name")
    parser.add_argument('--wandb_run_name', type=str, default='xrlwithbehavior', help="Wandb run name")
    parser.add_argument('--wandb_entity', type=str, default='mail-rishav9', help="Wandb entity name")
    parser.add_argument('--model_type', type=str, default='sequence', help="Model type: one id for each state or sequence")


    args = parser.parse_args()
    return args



def get_args():
    parser = argparse.ArgumentParser(description="Transformer Model Parameters")

    parser.add_argument('--feature_dim', type=int, default=23, help="Combined state-action dimension")
    parser.add_argument('--model_dim', type=int, default=128, help="Transformer model dimension")
    parser.add_argument('--num_heads', type=int, default=4, help="Number of attention heads")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of transformer layers")
    parser.add_argument('--num_embeddings', type=int, default=128, help="VQ-VAE codebook size")
    parser.add_argument('--segment_length', type=int, default=25, help="Segment length for behavior isolation")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--weights_path', default='')
    parser.add_argument('--train', action='store_true')

    args = parser.parse_args()
    return args



def split_into_trajectories(dataset, seq_len=5):
    trajectories = []
    current_traj = {'observations': [], 'actions': []}

    for i in range(len(dataset['observations'])):
        current_traj['observations'].append(dataset['observations'][i])
        current_traj['actions'].append(dataset['actions'][i])
        
        # If 'done' is True, finalize the trajectory and start a new one
        if dataset['terminals'][i] and len(current_traj['observations']) > seq_len:
            trajectories.append({
                'observations': np.array(current_traj['observations']),
                'actions': np.array(current_traj['actions'])
            })
            current_traj = {'observations': [], 'actions': []}

    return trajectories



def one_hot_encode(actions, num_actions):
    # Convert actions to one-hot encoded format
    actions = actions.astype(int) 
    return np.eye(num_actions)[actions]

def extract_overlapping_segments_minigrid(dataset, segment_length, num_segments):
    trajectories = split_into_trajectories(dataset, seq_len=segment_length)
    
    segments = []
    for trajectory in trajectories:
        observations = [obs.flatten() for obs in trajectory['observations']]
        actions = one_hot_encode(trajectory['actions'], len(np.unique(dataset['actions']))) 
        traj_segments = []

        # Calculate segments for the current trajectory
        max_index = min(len(observations) - segment_length + 1, num_segments)
        for i in range(max_index):
            obs_segment = observations[i:i + segment_length]
            act_segment = actions[i:i + segment_length]
            segment = np.concatenate((obs_segment, act_segment), axis=-1)
            traj_segments.append(segment)
        
        segments.append(traj_segments)  # Append list of segments for this trajectory

    return segments



def extract_overlapping_segments(d4rl_dataset, segment_length, num_segments):
    observations = d4rl_dataset['observations']
    actions = d4rl_dataset['actions']
    segments = [
        np.concatenate((observations[i:i + segment_length], actions[i:i + segment_length]), axis=-1)
        for i in range(num_segments)
    ]
    return segments


def save_cluster_frames(env, cluster_states, save_dir, n_frames=100):
    import os
    import numpy as np
    from PIL import Image
    
    def render_frame(env, state):
        env.reset()
        mujoco_model = env.unwrapped.model
        mujoco_data = env.unwrapped.data

        nq = mujoco_model.nq
        nv = mujoco_model.nv

        qpos = np.insert(state[:nq - 1], 0, 0)
        qvel = state[nq - 1:nq + nv - 1]

        mujoco_data.qpos[:] = qpos
        mujoco_data.qvel[:] = qvel
        mujoco.mj_forward(mujoco_model, mujoco_data)

        img = env.render()
        return img
    
    os.makedirs(save_dir, exist_ok=True)
    
    for cluster_idx, states in enumerate(cluster_states):
        cluster_dir = os.path.join(save_dir, f'cluster_{cluster_idx}')
        os.makedirs(cluster_dir, exist_ok=True)
        
        if len(states) <= n_frames:
            states_to_render = states
        else:
            indices = np.linspace(0, len(states) - 1, n_frames, dtype=int)
            states_to_render = states[indices]
        
        for frame_idx, state in enumerate(states_to_render):
            try:
                img = render_frame(env, state)
                
                if img is not None:
                    img = Image.fromarray(img)
                    frame_path = os.path.join(cluster_dir, f'frame_{frame_idx:04d}.png')
                    img.save(frame_path)
                
            except Exception as e:
                print(f"Error rendering frame {frame_idx} for cluster {cluster_idx}: {str(e)}")
                continue
        
        print(f"Saved {len(states_to_render)} frames for cluster {cluster_idx}")


def render_frame(env, state):
    env.reset()
    mujoco_model = env.unwrapped.model
    mujoco_data = env.unwrapped.data

    nq = mujoco_model.nq  # Number of position variables
    nv = mujoco_model.nv  # Number of velocity variables

    # Add zero to the initial position to match qpos dimensions
    qpos = np.insert(state[:nq - 1], 0, 0)  # Add zero as the root position
    qvel = state[nq - 1:nq + nv - 1]

    # Update the MuJoCo simulation state
    mujoco_data.qpos[:] = qpos
    mujoco_data.qvel[:] = qvel
    mujoco.mj_forward(mujoco_model, mujoco_data)  # Propagate the state

    # Render the frame
    img = env.render()
    return img


def render_target_and_predicted_frames(env_name, target_states, predicted_states, save_dir="rendered_frames"):
    assert len(target_states) == len(predicted_states), "Target and predicted states must have the same length"
    import gymnasium 
    os.makedirs(save_dir, exist_ok=True)
    target_dir = os.path.join(save_dir, "target_frames")
    predicted_dir = os.path.join(save_dir, "predicted_frames")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(predicted_dir, exist_ok=True)

    env = gymnasium.make('HalfCheetah-v4', render_mode='rgb_array')
    os.environ["MUJOCO_GL"] = "osmesa"  # Force CPU rendering

    for i, (target, predicted) in enumerate(zip(target_states[0], predicted_states[0])):
        target_frame = render_frame(env, target)
        predicted_frame = render_frame(env, predicted)
        cv2.imwrite(os.path.join(target_dir, f"frame_{i:04d}.png"), cv2.cvtColor(target_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(predicted_dir, f"frame_{i:04d}.png"), cv2.cvtColor(predicted_frame, cv2.COLOR_RGB2BGR))

    env.close()


from sklearn.manifold import MDS
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
import json
from time import time

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn.manifold import MDS
import os
import matplotlib.pyplot as plt

def analyze_quantized_sequences2(quantized_sequences: List[List[int]], 
                              codebook_vectors: np.ndarray,
                              n_cuts: int = 2,
                              distance_weight: float = 8,
                              transition_weight: float = 0.1,
                              save_dir: str = None):
    
    sequences = np.array(quantized_sequences)
    n_clusters = len(codebook_vectors)
    
    # Compute transitions
    transitions = np.zeros((n_clusters, n_clusters), dtype=int)
    for seq in sequences:
        np.add.at(transitions, (seq[:-1], seq[1:]), 1)
    
    # Spectral clustering
    distances = cdist(codebook_vectors, codebook_vectors, metric='euclidean')
    distances = distances / np.max(distances)
    affinity_matrix = np.exp(-distances)
    spectral = SpectralClustering(n_clusters=n_cuts, affinity='precomputed', random_state=42)
    partition_labels = spectral.fit_predict(affinity_matrix)
    partitions = [np.where(partition_labels == i)[0].tolist() for i in range(n_cuts)]
    
    def get_average_sequence_length(partition_nodes):
        partition_set = set(partition_nodes)
        sequence_lengths = []
        
        for seq in quantized_sequences:
            current_length = 0
            for node in seq:
                if node in partition_set:
                    current_length += 1
                else:
                    if current_length > 1:
                        sequence_lengths.append(current_length)
                    current_length = 0
            if current_length > 1:
                sequence_lengths.append(current_length)
        
        return np.mean(sequence_lengths) if sequence_lengths else 0
    
    avg_sequence_lengths = [get_average_sequence_length(p) for p in partitions]
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'partition_labels.npy'), partition_labels)
        
        fig = plt.figure(figsize=(24, 12))
        ax1 = plt.subplot(121)
        
        # Create undirected graph for layout
        G = nx.Graph()
        for i in range(n_cuts):
            G.add_node(i)
        
        # Calculate partition transitions
        partition_counts = np.zeros((n_cuts, n_cuts))
        for i in range(n_clusters):
            for j in range(n_clusters):
                if transitions[i,j] > 0:
                    p_u = next(k for k, p in enumerate(partitions) if i in p)
                    p_v = next(k for k, p in enumerate(partitions) if j in p)
                    partition_counts[p_u, p_v] += transitions[i,j]
        
        # Add edges
        for i in range(n_cuts):
            for j in range(i + 1, n_cuts):
                if partition_counts[i,j] > 0 or partition_counts[j,i] > 0:
                    G.add_edge(i, j)
        
        # Use spring layout with more space
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        colors = plt.cm.rainbow(np.linspace(0, 1, n_cuts))
        max_partition_size = max(len(p) for p in partitions)
        
        for i, partition in enumerate(partitions):
            node_size = 2000 + (len(partition) / max_partition_size) * 4000
            nx.draw_networkx_nodes(G, pos,
                                 nodelist=[i],
                                 node_color=[colors[i]],
                                 node_size=node_size,
                                 alpha=0.7,
                                 label=f'Partition {i}')
            
            label = f'Partition {i}\nNodes: {len(partition)}\nAvg Seq Length: {avg_sequence_lengths[i]*3:.1f}'
            plt.annotate(label,
                        xy=pos[i],
                        xytext=(0, 20),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        bbox=dict(facecolor='white', 
                                edgecolor='gray',
                                alpha=0.9,
                                pad=0.5),
                        zorder=5)
        
        # Draw edges showing both directions
        max_count = np.max(partition_counts)
        
        for (i, j) in G.edges():
            # Draw two curved edges for each connection
            if partition_counts[i,j] > 0:
                width = 1 + 3 * (partition_counts[i,j] / max_count)
                # Forward direction (positive curve)
                nx.draw_networkx_edges(G, pos,
                                     [(i, j)],
                                     width=width,
                                     edge_color='gray',
                                     arrows=True,
                                     arrowsize=20,
                                     alpha=0.5,
                                     connectionstyle='arc3,rad=0.2')
                
                # Add forward count
                edge_center = ((pos[i][0] + pos[j][0])/2 + 0.05, 
                             (pos[i][1] + pos[j][1])/2 + 0.05)
                plt.annotate(f'{int(partition_counts[i,j])}',
                            xy=edge_center,
                            ha='center',
                            va='center',
                            bbox=dict(facecolor='white',
                                    edgecolor='none',
                                    alpha=0.7,
                                    pad=0.1))
            
            if partition_counts[j,i] > 0:
                width = 1 + 3 * (partition_counts[j,i] / max_count)
                # Backward direction (negative curve)
                nx.draw_networkx_edges(G, pos,
                                     [(j, i)],
                                     width=width,
                                     edge_color='gray',
                                     arrows=True,
                                     arrowsize=20,
                                     alpha=0.5,
                                     connectionstyle='arc3,rad=0.2')
                
                # Add backward count
                edge_center = ((pos[i][0] + pos[j][0])/2 - 0.05,
                             (pos[i][1] + pos[j][1])/2 - 0.05)
                plt.annotate(f'{int(partition_counts[j,i])}',
                            xy=edge_center,
                            ha='center',
                            va='center',
                            bbox=dict(facecolor='white',
                                    edgecolor='none',
                                    alpha=0.7,
                                    pad=0.1))
        
        ax1.set_title('Aggregated Transition Graph\n(Node sizes reflect number of original nodes)', 
                     pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
        plt.axis('equal')
        
        # Second subplot: Partition transitions matrix
        ax2 = plt.subplot(122)
        transition_probs = partition_counts / np.maximum(partition_counts.sum(axis=1, keepdims=True), 1)
        
        im = ax2.imshow(transition_probs, cmap='YlOrRd')
        plt.colorbar(im, ax=ax2, label='Transition Probability')
        
        for i in range(n_cuts):
            for j in range(n_cuts):
                if transition_probs[i,j] > 0:
                    text = f'{transition_probs[i,j]:.2f}\n({int(partition_counts[i,j])})'
                    ax2.text(j, i, text, ha='center', va='center')
        
        ax2.set_title('Partition Transition Probabilities\n(Raw counts in parentheses)')
        ax2.set_xlabel('To Partition')
        ax2.set_ylabel('From Partition')
        ax2.set_xticks(range(n_cuts))
        ax2.set_yticks(range(n_cuts))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'transition_analysis.png'), 
                   bbox_inches='tight', dpi=300, pad_inches=0.5)
        plt.close()
    
    return G, partitions, distances



def get_partition_labels(quantized_sequences, codebook_vectors, n_cuts=10):
    # Convert sequences to codebook indices if not already done
    sequences = np.array(quantized_sequences)
    
    # Compute distances between codebook vectors
    distances = cdist(codebook_vectors, codebook_vectors, metric='euclidean')
    distances = distances / np.max(distances)
    
    # Create affinity matrix for spectral clustering
    affinity_matrix = np.exp(-distances)
    
    # Perform spectral clustering
    spectral = SpectralClustering(n_clusters=n_cuts, 
                                 affinity='precomputed',
                                 random_state=42)
    
    partition_labels = spectral.fit_predict(affinity_matrix)
    
    # Map sequence indices to partition labels
    sequence_labels = partition_labels[sequences]
    
    return sequence_labels, partition_labels

def analyze_quantized_sequences(quantized_sequences: List[List[int]], 
                              codebook_vectors: np.ndarray,
                              n_cuts: int = 2,
                              distance_weight: float = 8,
                              transition_weight: float = 0.1,
                              save_dir: str = None):
    
    sequences = np.array(quantized_sequences)
    n_clusters = len(codebook_vectors)
    
    # Compute transitions
    transitions = np.zeros((n_clusters, n_clusters), dtype=int)
    for seq in sequences:
        np.add.at(transitions, (seq[:-1], seq[1:]), 1)
    
    # Normalize transitions
    transition_matrix = transitions.astype(float)
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix /= row_sums
    
    # Compute distances
    distances = cdist(codebook_vectors, codebook_vectors, metric='euclidean')
    distances = distances / np.max(distances)
    
    # Combine distance and transition information for node positioning
    combined_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            # Higher value means nodes should be closer
            combined_matrix[i,j] = (distance_weight * (1 - distances[i,j]) + 
                                  transition_weight * (transition_matrix[i,j] + transition_matrix[j,i]) / 2)
    
    # Create graph with simplified edges
    graph = nx.DiGraph()
    distance_threshold = np.percentile(distances[distances > 0], 75)
    
    # Build initial graph
    for i in range(n_clusters):
        for j in range(n_clusters):
            if transitions[i,j] > 0 or distances[i,j] < distance_threshold:
                graph.add_edge(i, j,
                             weight=combined_matrix[i,j],
                             transition_count=transitions[i,j],
                             distance=distances[i,j])
    
    # Spectral clustering
    affinity_matrix = np.exp(-distances)  # Use distances for clustering
    spectral = SpectralClustering(n_clusters=n_cuts, 
                                 affinity='precomputed',
                                 random_state=42)
    partition_labels = spectral.fit_predict(affinity_matrix)
    partitions = [np.where(partition_labels == i)[0].tolist() for i in range(n_cuts)]
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'partition_labels.npy'), partition_labels)
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(24, 12))
        
        # First subplot: Main graph
        ax1 = plt.subplot(121)
        
        # Use MDS on combined matrix for node positioning
        mds = MDS(n_components=2, 
                 dissimilarity='precomputed',
                 random_state=42,
                 normalized_stress='auto',
                 max_iter=1000)
        
        # Convert combined matrix to distances (higher value = closer nodes)
        positioning_matrix = 1 - combined_matrix/np.max(combined_matrix)
        pos_array = mds.fit_transform(positioning_matrix)
        pos_array = pos_array * 1.5
        pos = {i: (pos_array[i][0], pos_array[i][1]) for i in range(n_clusters)}
        
        # Draw nodes
        colors = plt.cm.rainbow(np.linspace(0, 1, n_cuts))
        for i, partition in enumerate(partitions):
            nx.draw_networkx_nodes(graph, pos, 
                                 nodelist=partition,
                                 node_color=[colors[i]], 
                                 node_size=1200,
                                 alpha=0.7,
                                 label=f'Partition {i}')
        
        # Create simplified inter-partition edges
        partition_edges = []
        partition_weights = np.zeros((n_cuts, n_cuts))
        partition_counts = np.zeros((n_cuts, n_cuts))
        
        # Compute average positions for each partition
        partition_centers = {}
        for i, partition in enumerate(partitions):
            center = np.mean([pos[node] for node in partition], axis=0)
            partition_centers[i] = center
        
        # Count total transitions between partitions
        for u, v, data in graph.edges(data=True):
            p_u = next(i for i, p in enumerate(partitions) if u in p)
            p_v = next(i for i, p in enumerate(partitions) if v in p)
            if p_u != p_v:
                partition_weights[p_u, p_v] += data['weight']
                partition_counts[p_u, p_v] += data['transition_count']
        
        # Draw single edge between partitions with total weight
        for i in range(n_cuts):
            for j in range(n_cuts):
                if partition_counts[i,j] > 0:
                    # Draw curved edges between partitions
                    center1 = partition_centers[i]
                    center2 = partition_centers[j]
                    width = 2 * partition_weights[i,j] / np.max(partition_weights)
                    
                    # Add edge label for transition count
                    edge_label_pos = ((center1[0] + center2[0])/2, 
                                    (center1[1] + center2[1])/2)
                    plt.annotate(f'{int(partition_counts[i,j])}',
                               xy=edge_label_pos,
                               xytext=(5, 5),
                               textcoords='offset points',
                               bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
                    
                    # Draw edge
                    nx.draw_networkx_edges(graph, 
                                         {i: center1, j: center2},
                                         [(i, j)],
                                         width=width,
                                         edge_color='gray',
                                         arrows=True,
                                         arrowsize=20,
                                         alpha=0.5)
        
        # Draw node labels
        nx.draw_networkx_labels(graph, pos, 
                              labels={node: f'C{node}' for node in graph.nodes()},
                              font_size=8,
                              bbox=dict(facecolor='white', 
                                      edgecolor='none', 
                                      alpha=0.7))
        
        ax1.set_title('Transition Graph\n(Node positions reflect combined distance and transition strength)', 
                     pad=20)
        ax1.legend(loc='upper right')
        plt.axis('equal')
        
        # Second subplot: Partition transitions matrix
        ax2 = plt.subplot(122)
        transition_probs = partition_counts / np.maximum(partition_counts.sum(axis=1, keepdims=True), 1)
        
        im = ax2.imshow(transition_probs, cmap='YlOrRd')
        plt.colorbar(im, ax=ax2, label='Transition Probability')
        
        for i in range(n_cuts):
            for j in range(n_cuts):
                if transition_probs[i,j] > 0:
                    text = f'{transition_probs[i,j]:.2f}\n({int(partition_counts[i,j])})'
                    ax2.text(j, i, text, ha='center', va='center')
        
        ax2.set_title('Partition Transition Probabilities\n(Raw counts in parentheses)')
        ax2.set_xlabel('To Partition')
        ax2.set_ylabel('From Partition')
        ax2.set_xticks(range(n_cuts))
        ax2.set_yticks(range(n_cuts))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'transition_analysis.png'), 
                   bbox_inches='tight', dpi=300, pad_inches=0.5)
        plt.close()
    
    return graph, partitions, distances


def evaluate_partition_quality(distances, partitions):
    """Evaluate the quality of partitioning based on distances"""
    n_partitions = len(partitions)
    quality_metrics = {
        'intra_partition_distances': [],  # Average distance within partitions
        'inter_partition_distances': []   # Average distance between partitions
    }
    
    # Calculate intra-partition distances
    for partition in partitions:
        if len(partition) > 1:
            intra_distances = []
            for i in partition:
                for j in partition:
                    if i < j:
                        intra_distances.append(distances[i,j])
            quality_metrics['intra_partition_distances'].append(np.mean(intra_distances))
    
    # Calculate inter-partition distances
    for i in range(n_partitions):
        for j in range(i+1, n_partitions):
            inter_distances = []
            for node1 in partitions[i]:
                for node2 in partitions[j]:
                    inter_distances.append(distances[node1,node2])
            quality_metrics['inter_partition_distances'].append(np.mean(inter_distances))
    
    return quality_metrics



def save_cluster_frames(env, cluster_states, save_dir, n_frames=100):
    import os
    import numpy as np
    from PIL import Image
    
    def render_frame(env, state):
        env.reset()
        mujoco_model = env.unwrapped.model
        mujoco_data = env.unwrapped.data

        nq = mujoco_model.nq
        nv = mujoco_model.nv

        qpos = np.insert(state[:nq - 1], 0, 0)
        qvel = state[nq - 1:nq + nv - 1]

        mujoco_data.qpos[:] = qpos
        mujoco_data.qvel[:] = qvel
        mujoco.mj_forward(mujoco_model, mujoco_data)

        img = env.render()
        return img
    
    os.makedirs(save_dir, exist_ok=True)
    
    for cluster_idx, states in enumerate(cluster_states):
        cluster_dir = os.path.join(save_dir, f'cluster_{cluster_idx}')
        os.makedirs(cluster_dir, exist_ok=True)
        
        if len(states) <= n_frames:
            states_to_render = states
        else:
            indices = np.linspace(0, len(states) - 1, n_frames, dtype=int)
            states_to_render = states[indices]
        
        for frame_idx, state in enumerate(states_to_render):
            try:
                img = render_frame(env, state)
                
                if img is not None:
                    img = Image.fromarray(img)
                    frame_path = os.path.join(cluster_dir, f'frame_{frame_idx:04d}.png')
                    img.save(frame_path)
                
            except Exception as e:
                print(f"Error rendering frame {frame_idx} for cluster {cluster_idx}: {str(e)}")
                continue
        
        print(f"Saved {len(states_to_render)} frames for cluster {cluster_idx}")

def get_states_by_cluster(all_states, partition_labels):
    import numpy as np
    n_clusters = len(np.unique(partition_labels))
    cluster_states = [[] for _ in range(n_clusters)]
    
    for state, label in zip(all_states, partition_labels):
        print(label)
        cluster_states[label].append(state)
    
    return [np.array(states) for states in cluster_states]

def build_cluster_transition_graph(codebook_vectors: np.ndarray, 
                                 quantized_sequences: List[List[int]], 
                                 initial_centers: int = 5,
                                 max_centers: int = 20,
                                 save_path: str = None,
                                 plot_path: str = None) -> Tuple[nx.DiGraph, Dict[int, int]]:
    
    initial_centers = kmeans_plusplus_initializer(codebook_vectors, initial_centers).initialize()
    xmeans_instance = xmeans(codebook_vectors, initial_centers, max_centers)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()
    
    cluster_mapping = {}
    for cluster_idx, cluster in enumerate(clusters):
        for codebook_idx in cluster:
            cluster_mapping[codebook_idx] = cluster_idx
    
    graph = nx.DiGraph()
    transition_counts = {}
    
    for sequence in quantized_sequences:
        cluster_seq = [cluster_mapping[idx] for idx in sequence]
        for i in range(len(cluster_seq) - 1):
            current_cluster = cluster_seq[i]
            next_cluster = cluster_seq[i + 1]
            
            if (current_cluster, next_cluster) not in transition_counts:
                transition_counts[(current_cluster, next_cluster)] = 0
            transition_counts[(current_cluster, next_cluster)] += 1
    
    for (src, dst), count in transition_counts.items():
        graph.add_edge(src, dst, weight=count)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nx.write_gpickle(graph, save_path)
    
    if plot_path:
        plt.figure(figsize=(12, 8))
        
        # Use MDS to project cluster centers to 2D while preserving distances
        mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
        centers_2d = mds.fit_transform(centers)
        
        # Create position dictionary for networkx
        pos = {i: (centers_2d[i][0], centers_2d[i][1]) for i in range(len(centers))}
        
        # Draw edges
        edges = graph.edges()
        weights = [graph[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        normalized_weights = [3 * w / max_weight for w in weights]
        
        nx.draw_networkx_nodes(graph, pos, 
                             node_color='lightblue',
                             node_size=1000, 
                             alpha=0.6)
        
        nx.draw_networkx_edges(graph, pos, 
                             width=normalized_weights,
                             edge_color='gray', 
                             arrows=True, 
                             arrowsize=20, 
                             alpha=0.5)
        
        nx.draw_networkx_labels(graph, pos, 
                              labels={node: f'C{node}' for node in graph.nodes()})
        
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels)
        
        plt.title('Cluster Transition Graph\n(Node positions reflect actual cluster distances)')
        plt.axis('equal')
        
        # os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    return graph, cluster_mapping



def plot_pca_clusters(codebook, n_components=2, n_clusters=10, title="PCA and KMeans Clustering", save_path=None):
    if n_components not in [2, 3]:
        raise ValueError("n_components must be 2 or 3 for visualization.")
    
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(codebook)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced_vectors)
    
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        for cluster in range(n_clusters):
            cluster_points = reduced_vectors[labels == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster+1}", alpha=0.7)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for cluster in range(n_clusters):
            cluster_points = reduced_vectors[labels == cluster]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f"Cluster {cluster+1}", alpha=0.7)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
    
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def extract_non_overlapping_segments(d4rl_dataset, segment_length, num_segments):
    observations = d4rl_dataset['observations']
    actions = d4rl_dataset['actions']
    segments = []
    
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        if end_idx > len(observations):
            break  # Stop if there aren't enough steps left for a full segment
        
        # Concatenate observations and actions for the segment
        segment = np.concatenate((observations[start_idx:end_idx], actions[start_idx:end_idx]), axis=-1)
        segments.append(segment)
    
    return segments

def generate_causal_mask(size):
    mask = torch.tril(torch.ones(size, size))  # Lower triangular matrix
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask



def plot_distances_from_center(segments, encoder, name='distance_from_center_minigrid.png', env_name='halfcheetah-medium-v2'):
    with torch.no_grad():
        encoder.eval()
        distances = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Calculate the center of the codebook vectors
        codebook_vectors = encoder.vq_layer.embedding
        center_vector = codebook_vectors.mean(dim=0)

        for i in range(len(segments)):
            quantized, _, encoding_indices = encoder(torch.tensor(segments[i]).to(device).unsqueeze(0).float())
            codebook_vector = encoder.vq_layer.embedding[encoding_indices.view(-1)]

            # Compute distance from the center
            distance = torch.norm(codebook_vector - center_vector).mean().item()
            distances.append(distance)

            render_segment(segments[i], env_name, f'segment_viz/segment_{i}.mp4') # Render the segment
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(distances)), distances, marker='o', label='Distance from Center')

        for idx, distance in enumerate(distances):
            plt.text(idx, distance, f'{idx}', ha='right', va='bottom')
        plt.xlabel('Segment Index')
        plt.ylabel('Average Distance from Center of Codebook Vector')
        plt.title('Distance from Center of Codebook Vector for Each Segment')
        plt.legend()
        plt.show()
        plt.savefig(name)



def plot_distance_state(segment, encoder, name='distance_prev_state.png', env_name='halfcheetah-medium-v2'):

    with torch.no_grad():
        encoder.eval()
        distances = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'


        all_vecs = []
        for i in range(len(segment)):
            q, _, encoding_indices = encoder(torch.tensor(segment[i]).to(device).unsqueeze(0).float())
            codebook_vectors = encoder.vq_layer.embedding[encoding_indices.view(-1)]
            all_vecs.append(codebook_vectors)
        all_vecs = torch.cat(all_vecs, dim=0)
        print(all_vecs.shape)
        for i in range(all_vecs.shape[0] - 1):
            distance = torch.norm(all_vecs[i] - all_vecs[i + 1]).mean().item()
            distances.append(distance)
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(distances)), distances, marker='o', label='Distances')
        plt.xlabel('State Index')
        plt.ylabel('Average Distance Between Consecutive Codebook Vectors')
        plt.title('Distance Between Codebook Vectors of Consecutive States')
        plt.legend()
        plt.show()
        plt.savefig(name)


def plot_distances_minigrid(segments, encoder, name='distance_from_prevcodebook_minigrid.png', env_name='halfcheetah-medium-v2'):

    with torch.no_grad():
        encoder.eval()
        distances = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for i in range(len(segments) - 1):
            quantized, _, encoding_indices = encoder(torch.tensor(segments[i]).to(device).unsqueeze(0).float())
            quantized1, _, encoding_indices1 = encoder(torch.tensor(segments[i + 1]).to(device).unsqueeze(0).float())

            codebook_vector = encoder.vq_layer.embedding[encoding_indices.view(-1)]
            codebook_vector1 = encoder.vq_layer.embedding[encoding_indices1.view(-1)]

            distance = torch.norm(codebook_vector - codebook_vector1).mean().item()
            distances.append(distance)


            if distance > 0.5:
                render_segment(segments[i], env_name, f'seq_adj/segment_{i}.mp4') # Render the segment
                render_segment(segments[i+1], env_name, f'seq_adj/segment_{i+1}.mp4')
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(distances)), distances, marker='o', label='Distances')
        plt.xlabel('Segment Index')
        plt.ylabel('Average Distance Between Consecutive Codebook Vectors')
        plt.title('Distance Between Codebook Vectors of Consecutive Segments')
        plt.legend()
        plt.show()
        plt.savefig(name)




def plot_distances(segments, encoder, quantizer):
    with torch.no_grad():
        encoder.eval()
        quantizer.eval()
        distances = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for i in range(len(segments) - 1):
            encoded_i = encoder(torch.tensor(segments[i]).to(device).unsqueeze(0))
            quantized_i, _, encoding_indices_i = quantizer(encoded_i[:, -1, :])
            
            encoded_i1 = encoder(torch.tensor(segments[i + 1]).to(device).unsqueeze(0))
            quantized_i1, _, encoding_indices_i1 = quantizer(encoded_i1[:, -1, :])
            
            codebook_vector_i = quantizer.embedding.weight[encoding_indices_i.view(-1)]
            codebook_vector_i1 = quantizer.embedding.weight[encoding_indices_i1.view(-1)]
            
            distance = torch.norm(codebook_vector_i - codebook_vector_i1).mean().item()
            distances.append(distance)

        # Find extreme points based on deviation from the mean or local maxima
        # mean_distance = np.mean(distances)
        # std_distance = np.std(distances)
        # threshold = mean_distance + 1.5 * std_distance  # Points significantly above mean + threshold are considered extremes

        # extreme_points = [i for i, dist in enumerate(distances) if dist > threshold]

        # Plotting the distances with extreme points highlighted
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(distances)), distances, marker='o', label='Distances')
        # plt.scatter(extreme_points, [distances[i] for i in extreme_points], color='red', label='Extreme Points')
        plt.xlabel('Segment Index')
        plt.ylabel('Average Distance Between Consecutive Codebook Vectors')
        plt.title('Distance Between Codebook Vectors of Consecutive Segments')
        plt.legend()
        plt.show()
        plt.savefig('distance_from_prevcodebook.png')




def cluster_codebook_vectors(codebook_embeddings, num_clusters=10, random_state=42):
    # Convert to numpy array if input is a tensor
    if isinstance(codebook_embeddings, torch.Tensor):
        codebook_embeddings = codebook_embeddings.detach().cpu().numpy()

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(codebook_embeddings)
    centers = kmeans.cluster_centers_
    return clusters, kmeans, centers


def visualize_codebook_clusters(codebook_embeddings, clusters, cluster_centers, env_name, method='pca'):
    if isinstance(codebook_embeddings, torch.Tensor):
        codebook_embeddings = codebook_embeddings.detach().cpu().numpy()
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")
    embeddings_2d = reducer.fit_transform(codebook_embeddings)

    if method=='pca':
        centers_2d = reducer.transform(cluster_centers)

    # Create the plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='tab10', alpha=0.7, label="Data Points")
    
    if method == 'pca':
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=100, label="Cluster Centers")

    # Annotate each point with the code index
    for i, (x, y) in enumerate(embeddings_2d):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

    plt.colorbar(scatter, ticks=range(np.max(clusters)+1))
    plt.title(f'Codebook Clusters Visualized using {method.upper()} with Code Indices and Cluster Centers')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'codebookcluster_{env_name}_{method}.png')
    plt.show()


#render halfcheetah
def render_halfcheetah(env, state):
    env.reset()
    env.env.sim.set_state_from_flattened(np.insert(state, 0, [0,0]))
    env.env.sim.forward()
    img = env.render(mode="rgb_array")
    return img



class D4RLDatasetMujoco(Dataset):
    def __init__(self, states, actions, next_states, sequence_length=4):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.states) - self.sequence_length

    def __getitem__(self, idx):
        state_seq = self.states[idx:idx + self.sequence_length]
        action_seq = self.actions[idx:idx + self.sequence_length]
        next_state_seq = self.next_states[idx + 1:idx + self.sequence_length + 1]
        state_action_seq = torch.cat((torch.tensor(state_seq), torch.tensor(action_seq)), dim=-1)
        return state_action_seq, torch.tensor(next_state_seq)


class FlatObsWrapper:
    def __init__(self, env):
        self.env = env

    def flatten_obs(self, obs):
        full_grid = self.env.grid.encode()
        full_grid[self.env.agent_pos[0]][self.env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            self.env.agent_dir
        ])
        full_grid = full_grid[1:-1, 1:-1]  # remove outer walls
        return full_grid.ravel()



class D4RLDiscrete(Dataset):
    def __init__(self, states, actions, dones, seq_len=5, num_actions=6, env=None):
        self.states = [state.flatten() for state in states]
        self.actions = actions
        self.dones = dones
        self.num_actions = num_actions 
        self.seq_len = seq_len
        self.episodes = self.create_episodes(self.states, actions, dones)
        

    def create_episodes(self, states, actions, dones):
        episodes = []
        episode = {'states': [], 'actions': []}
        for i in range(len(states)):
            episode['states'].append(states[i])
            # Convert each action to a one-hot vector
            one_hot_action = F.one_hot(torch.tensor(int(actions[i])), num_classes=self.num_actions).float()
            episode['actions'].append(one_hot_action)
            if dones[i]:
                if len(episode['states']) > self.seq_len:
                    episodes.append(episode)
                episode = {'states': [], 'actions': []}
        if episode['states'] and len(episode['states']) > self.seq_len:
            episodes.append(episode)
        return episodes

    def __len__(self):
        return sum(len(ep['states']) - 1 for ep in self.episodes)

    def __getitem__(self, idx):
        episode = random.choice(self.episodes)
        episode_length = len(episode['states'])
        start_idx = random.randint(0, episode_length - self.seq_len - 1)
        state_seq = episode['states'][start_idx:start_idx + self.seq_len]
        action_seq = episode['actions'][start_idx:start_idx + self.seq_len]
        next_state_seq = episode['states'][start_idx + 1:start_idx + self.seq_len + 1]
        state_action_seq = torch.cat((torch.tensor(state_seq), torch.stack(action_seq)), dim=-1)
        return state_action_seq, torch.tensor(next_state_seq)



class D4RLAtariBeXRL(Dataset):
    def __init__(self, states, actions, dones, seq_len=5, num_actions=6, env=None):
        self.states = states  # Each state remains in its original frame shape [channels, height, width]
        self.actions = actions
        self.dones = dones
        self.num_actions = np.max(actions) + 1
        self.seq_len = seq_len
        self.episodes = self.create_episodes(states, actions, dones)

    def create_episodes(self, states, actions, dones):
        """
        Organizes the data into episodes based on `dones` signals, ensuring each episode has a minimum length.
        """
        episodes = []
        episode = {'states': [], 'actions': []}
        
        for i in range(len(states)):
            episode['states'].append(states[i])
            
            # Convert each action to a one-hot vector
            # one_hot_action = F.one_hot(torch.tensor(int(actions[i])), num_classes=self.num_actions).float()
            episode['actions'].append(actions[i])
            
            if dones[i]:  # End of episode
                if len(episode['states']) > self.seq_len:  # Only keep if episode is longer than seq_len
                    episodes.append(episode)
                episode = {'states': [], 'actions': []}
        
        # Add the last episode if it wasn't added due to lack of a terminal signal
        if episode['states'] and len(episode['states']) > self.seq_len:
            episodes.append(episode)
        
        return episodes

    def __len__(self):
        return sum(len(ep['states']) - 1 for ep in self.episodes)

    def __getitem__(self, idx):
        """
        Retrieves a sequence of states and actions for a randomly chosen episode.
        """
        episode = random.choice(self.episodes)
        episode_length = len(episode['states'])
        
        # Randomly choose a starting index such that a full sequence can be obtained
        start_idx = random.randint(0, episode_length - self.seq_len - 1)
        
        # Get sequences of states and actions
        state_seq = episode['states'][start_idx:start_idx + self.seq_len]
        action_seq = episode['actions'][start_idx:start_idx + self.seq_len]
        next_state_seq = episode['states'][start_idx + 1:start_idx + self.seq_len + 1]

        # Stack states and actions into tensors separately
        state_seq = torch.stack([torch.tensor(state) for state in state_seq])  # Shape: [seq_len, channels, height, width]
        next_state_seq = torch.stack([torch.tensor(state) for state in next_state_seq])  # Shape: [seq_len, channels, height, width]
        # Return states, actions, and next states separately
        return state_seq, torch.tensor(action_seq), next_state_seq



def render_segment(segment, env_name="halfcheetah-medium-v2", output_path="segment_video.mp4"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make(env_name)
    env.reset()
    video_frames = []
    
    # Replay the segment in the environment
    for state_action in segment:
        state = state_action[:env.observation_space.shape[0]]
        action = state_action[env.observation_space.shape[0]:]
        
        # Render the environment using the given state
        frame = render_halfcheetah(env, state)
        video_frames.append(frame)

    # Prepare the VideoWriter to save the video
    height, width, _ = video_frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for frame in video_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        out.write(frame_bgr)
        
    out.release()
    env.close()
    print(f"Segment video saved as {output_path}")



def render_videos_for_behavior_codes_opencv(model, num_videos=5, segment_length=10, env_name="halfcheetah-medium-v2", output_dir="behavior_videos4"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)  # Load D4RL dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert the dataset into state-action pairs
    states = dataset['observations']
    actions = dataset['actions']
    
    behavior_examples = {}
    num_segments = 80

    for segment_idx in range(num_segments):
        segment_data = np.concatenate([
            np.hstack([states[i], actions[i]]) for i in range(segment_idx * segment_length, (segment_idx + 1) * segment_length)
        ]).reshape(segment_length, -1)
        
        # Convert segment data to tensor
        segment = torch.tensor(segment_data, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, _, encoding_indices = model(segment)
            behavior_code = encoding_indices[0, -1].item()
        

        if behavior_code not in behavior_examples:
            behavior_examples[behavior_code] = []
        if len(behavior_examples[behavior_code]) < num_videos:
            behavior_examples[behavior_code].append(segment)

    # Render and save videos
    for behavior_code, segments in behavior_examples.items():
        print(f"Rendering videos for behavior code {behavior_code}")
        for i, segment in enumerate(segments):
            video_frames = []
            env.reset()
            
            # Replay segment in the environment
            for state_action in segment.squeeze(0).cpu().numpy():
                state = state_action[:env.observation_space.shape[0]]
                action = state_action[env.observation_space.shape[0]:]
                frame = render_halfcheetah(env, state)
                video_frames.append(frame)
            
            # Prepare the VideoWriter
            height, width, _ = video_frames[0].shape
            video_path = os.path.join(output_dir, f"behavior_{behavior_code}_video_{i + 1}.mp4")
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
            
            for frame in video_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            print(f"Saved video for behavior code {behavior_code} as {video_path}")

    env.close()
    print("Rendering complete.")


def _stack(observations, terminals, n_channels=4):
    rets = []
    t = 1
    for i in range(observations.shape[0]):
        if t < n_channels:
            padding_shape = (n_channels - t, ) + observations.shape[1:]
            padding = np.zeros(padding_shape, dtype=np.uint8)
            observation = observations[i - t + 1:i + 1]
            observation = np.vstack([padding, observation])
        else:
            # avoid copying data
            observation = observations[i - n_channels + 1:i + 1]

        rets.append(observation)

        if terminals[i]:
            t = 1
        else:
            t += 1
    return rets


def _load(name, dir_path):
    path = os.path.join(dir_path, name + '.gz')
    with gzip.open(path, 'rb') as f:
        print('loading {}...'.format(path))
        return np.load(f, allow_pickle=False)

class OfflineEnvAtari:

    def __init__(self,
                 game=None,
                 index=None,
                 start_epoch= 0,
                 last_epoch= 1,
                 stack=False,
                 path='./datasets'):
        self.game = game
        self.index = index
        self.start_epoch = start_epoch
        self.last_epoch = last_epoch
        self.stack = stack
        self.path = path

    def get_dataset(self):
        observation_stack = []
        action_stack = []
        reward_stack = []
        terminal_stack = []
        for epoch in range(self.start_epoch, self.last_epoch):

            observations = _load('observation', self.path)
            actions = _load('action', self.path)
            rewards = _load('reward', self.path)
            terminals = _load('terminal', self.path)

            # sanity check
            assert observations.shape == (1000000, 84, 84)
            assert actions.shape == (1000000, )
            assert rewards.shape == (1000000, )
            assert terminals.shape == (1000000, )

            observation_stack.append(observations)
            action_stack.append(actions)
            reward_stack.append(rewards)
            terminal_stack.append(terminals)

        if len(observation_stack) > 1:
            observations = np.vstack(observation_stack)
            actions = np.vstack(action_stack).reshape(-1)
            rewards = np.vstack(reward_stack).reshape(-1)
            terminals = np.vstack(terminal_stack).reshape(-1)
        else:
            observations = observation_stack[0]
            actions = action_stack[0]
            rewards = reward_stack[0]
            terminals = terminal_stack[0]

        # memory-efficient stacking
        if self.stack:
            observations = _stack(observations, terminals)
        else:
            observations = observations.reshape(-1, 1, 84, 84)

        data_dict = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'terminals': terminals
        }

        return data_dict
    

class AtariGrayscaleDataset(Dataset):
    def __init__(self, env_name, max_seq_len=50, image_size=(84, 84), frame_skip=4):
        # Initialize the dataset
        self.data = OfflineEnvAtari(stack=False, path='/home/rishav/scratch/d4rl_dataset/Seaquest/1/10').get_dataset()
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.frame_skip = frame_skip

        # Extract relevant data
        self.frames = self.data['observations']  # Grayscale image frames
        self.actions = self.data['actions']
        self.terminals = self.data['terminals']

        # Split trajectories
        self.trajectories = self._split_trajectories()

    def _split_trajectories(self):
        """
        Splits the dataset into trajectories based on terminal flags.
        Applies frame-skipping by selecting every `frame_skip`-th frame.
        """
        trajectories = []
        trajectory = {'frames': [], 'actions': []}

        for i in range(len(self.frames)):
            if i % self.frame_skip == 0:  # Apply frame-skip
                trajectory['frames'].append(self.frames[i])
                trajectory['actions'].append(self.actions[i])

            if self.terminals[i]:  # End of trajectory
                if len(trajectory['frames']) > self.max_seq_len:  # Ignore trajectories with single frames
                    trajectories.append(trajectory)
                trajectory = {'frames': [], 'actions': []}

        return trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        """
        Samples a random sequence from the trajectory.
        """
        trajectory = self.trajectories[idx]
        frames = trajectory['frames']
        actions = trajectory['actions']

        # Sample a random sequence from the trajectory
        traj_len = len(frames) - 1  # Exclude the last frame since it has no "next frame"
        seq_len = min(self.max_seq_len, traj_len)
        start_idx = np.random.randint(0, traj_len - seq_len + 1)

        frame_seq = frames[start_idx:start_idx + seq_len]
        action_seq = actions[start_idx:start_idx + seq_len]
        next_frame_seq = frames[start_idx + 1:start_idx + seq_len + 1]  # Next frames are shifted by 1

        # Normalize and convert images to tensor
        frame_seq = torch.tensor(frame_seq, dtype=torch.float32) / 255.0  # Normalize pixel values
        next_frame_seq = torch.tensor(next_frame_seq, dtype=torch.float32) / 255.0

        return frame_seq, torch.tensor(action_seq, dtype=torch.float32), next_frame_seq
    

class BehaviorAnalyzer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.behavior_cache = defaultdict(list)
        
    def extract_behaviors(self, dataloader, num_samples=1000):
        self.model.eval()
        codes = []
        transitions = []
        
        with torch.no_grad():
            for i, (obs_seq, action_seq) in enumerate(dataloader):
                if i * obs_seq.shape[0] >= num_samples:
                    break
                    
                obs_seq = obs_seq.to(self.device)
                _, _, encoding_indices = self.model(obs_seq)
                
                codes.extend(encoding_indices.cpu().numpy())
                
                for seq in encoding_indices:
                    for j in range(len(seq)-1):
                        transitions.append((seq[j].item(), seq[j+1].item()))
        
        return np.array(codes), transitions
    
    def analyze_transitions(self, transitions):
        unique_codes = len(self.model.vq.embedding.weight)
        transition_matrix = np.zeros((unique_codes, unique_codes))
        
        for from_code, to_code in transitions:
            transition_matrix[from_code, to_code] += 1
            
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                    where=row_sums!=0)
        
        return transition_matrix
    
    def visualize_behaviors(self, codes, save_path=None):
        embeddings = self.model.vq.embedding.weight.detach().cpu().numpy()
        
        tsne = TSNE(n_components=2, random_state=42)
        embedded_codes = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(embedded_codes[:, 0], embedded_codes[:, 1], 
                            c=np.arange(len(embedded_codes)), 
                            cmap='viridis')
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Behavior Embeddings')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def visualize_transitions(self, transition_matrix, save_path=None):
        plt.figure(figsize=(12, 10))
        sns.heatmap(transition_matrix, cmap='Blues')
        plt.title('Behavior Transition Matrix')
        plt.xlabel('To Behavior')
        plt.ylabel('From Behavior')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()