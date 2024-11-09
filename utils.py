

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




import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Transformer Model Parameters")

    parser.add_argument('--feature_dim', type=int, default=23, help="Combined state-action dimension")
    parser.add_argument('--model_dim', type=int, default=128, help="Transformer model dimension")
    parser.add_argument('--num_heads', type=int, default=4, help="Number of attention heads")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of transformer layers")
    parser.add_argument('--num_embeddings', type=int, default=64, help="VQ-VAE codebook size")
    parser.add_argument('--segment_length', type=int, default=25, help="Segment length for behavior isolation")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=150, help="Number of epochs")
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--weights_path', default='')

    args = parser.parse_args()
    return args



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
    centers_2d = reducer.transform(cluster_centers)

    # Create the plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='tab10', alpha=0.7, label="Data Points")
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
    plt.savefig('codebookcluster2.png')
    plt.show()


#render halfcheetah
def render_halfcheetah(env, state):
    env.reset()
    env.env.sim.set_state_from_flattened(np.insert(state, 0, [0,0]))
    env.env.sim.forward()
    img = env.render(mode="rgb_array")
    return img



class D4RLDataset(Dataset):
    def __init__(self, env_name='halfcheetah-medium-v2', segment_length=25, num_items=100000):
        self.env = gym.make(env_name)
        self.dataset = self.env.get_dataset()
        self.segment_length = segment_length
        self.observations = torch.tensor(self.dataset['observations'], dtype=torch.float32)
        self.actions = torch.tensor(self.dataset['actions'], dtype=torch.float32)
        self.data = torch.cat([self.observations[:num_items], self.actions[:num_items]], dim=-1)

    def __len__(self):
        return len(self.data) // self.segment_length

    def __getitem__(self, idx):
        start_idx = idx * self.segment_length
        segment = self.data[start_idx:start_idx + self.segment_length]
        return segment  # Shape: [segment_length, state_action_dim]


def render_videos_for_behavior_codes_opencv(model, num_videos=5, segment_length=10, env_name="halfcheetah-medium-v2", output_dir="behavior_videos"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env = gym.make(env_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    behavior_examples = {}  
    env.reset()
    for _ in range(1000): 
        segment = []
        for _ in range(segment_length):
            action = env.action_space.sample()
            state, _, done, _ = env.step(action)
            state_action = np.concatenate((state, action))
            segment.append(state_action)
            if done:
                env.reset()
                break
        segment = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device) 
        with torch.no_grad():
            _, _, encoding_indices = model(segment)  
            behavior_code = encoding_indices[0, -1].item() 
        if behavior_code not in behavior_examples:
            behavior_examples[behavior_code] = []
        if len(behavior_examples[behavior_code]) < num_videos:
            behavior_examples[behavior_code].append(segment)
        if all(len(videos) >= num_videos for videos in behavior_examples.values()):
            break

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
                print(frame.shape)
            
                
            # Prepare the VideoWriter
            height, width, _ = video_frames[0].shape
            video_path = os.path.join(output_dir, f"behavior_{behavior_code}_video_{i + 1}.mp4")
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
            
            # Write frames to video file
            for frame in video_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            print(f"Saved video for behavior code {behavior_code} as {video_path}")

    env.close()
    print("Rendering complete.")