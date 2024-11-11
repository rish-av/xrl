import torch
import os
import gym
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import d4rl 
import pandas as pd
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np


def get_args_mujoco():
    parser = argparse.ArgumentParser(description="Transformer Model Parameters")
    parser.add_argument('--env_name', type=str, default='halfcheetah-medium-v2', help="Mujoco environment name")
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
    parser.add_argument('--segment_length', type=int, default=25, help="Segment length for behavior isolation")
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='xrlwithbehaviors', help="Wandb project name")
    parser.add_argument('--wandb_run_name', type=str, default='xrlwithbehavior', help="Wandb run name")
    parser.add_argument('--wandb_entity', type=str, default='mail-rishav9', help="Wandb entity name")


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

def extract_overlapping_segments(d4rl_dataset, segment_length, num_segments):
    observations = d4rl_dataset['observations']
    actions = d4rl_dataset['actions']
    segments = [
        np.concatenate((observations[i:i + segment_length], actions[i:i + segment_length]), axis=-1)
        for i in range(num_segments)
    ]
    return segments


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
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        threshold = mean_distance + 1.5 * std_distance  # Points significantly above mean + threshold are considered extremes

        extreme_points = [i for i, dist in enumerate(distances) if dist > threshold]

        # Plotting the distances with extreme points highlighted
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(distances)), distances, marker='o', label='Distances')
        plt.scatter(extreme_points, [distances[i] for i in extreme_points], color='red', label='Extreme Points')
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


def visualize_codebook_clusters(codebook_embeddings, clusters, cluster_centers, method='pca'):
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
    plt.savefig(f'codebookcluster_{method}.png')
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
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))
            
            for frame in video_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            print(f"Saved video for behavior code {behavior_code} as {video_path}")

    env.close()
    print("Rendering complete.")
