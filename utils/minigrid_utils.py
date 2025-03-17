from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import cv2
np.bool = np.bool_

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pickle
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train Atari Model with Custom Model Name in WandB")
    
    # Model hyperparameters
    parser.add_argument('--max_seq_len', type=int, default=60, help="Maximum sequence length")
    parser.add_argument('--embed_dim', type=int, default=128, help="Embedding dimension")
    parser.add_argument('--nhead', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of transformer layers")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--num_embeddings', type=int, default=128, help="Number of embeddings for VQ")

    # Logging and experiment tracking
    parser.add_argument('--log', action='store_true', help="Enable logging with WandB")
    parser.add_argument('--wandb_project', type=str, default="atari-xrl", help="WandB project name")
    parser.add_argument('--wandb_entity', type=str, default="mail-rishav9", help="WandB entity name")
    parser.add_argument('--wandb_run_name', type=str, default="xrl_atari", help="Base name for WandB run")
    parser.add_argument('--patch_size', type=int, default=4, help="Patch size for image transformer")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument('--load_checkpoint', type=str, default=None, help="Path to load model checkpoint")
    parser.add_argument('--epochs', type=int, default=2000, help="Number of training epochs")
    parser.add_argument('--scheduler_step_size', type=int, default=50, help="Step size for learning rate scheduler")
    parser.add_argument('--frame_skip', type=int, default=4, help="Number of frames to skip in dataset")
    parser.add_argument('--dataset_path', type=str, default="/home/ubuntu/xrl/combined_data.pkl", help="Path to dataset")

    return parser.parse_args()


def get_dataloader(tokenized_trajectories, batch_size=1, offline_data=None, train=False):
    pass

class MiniGridSaved(Dataset):
    def __init__(self, path, max_seq_len=30, transform=None, action_dim=6):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.dataset = data
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.action_dim = action_dim
        self.episodes = self._get_episodes()
    def _get_episodes(self):
        """Split the dataset into episodes based on terminal flags."""
        episodes = []
        episode_start = 0
        
        for i in range(len(self.dataset['terminals'])):
            if self.dataset['terminals'][i] or i == len(self.dataset['terminals']) - 1:
                # Make sure we have enough states for both current and next state
                if (i + 1) - episode_start > 1:  # Need at least 2 states
                    episode = {
                        'start': episode_start,
                        'end': i + 1,  # Include the terminal state
                        'length': i + 1 - episode_start
                    }
                    episodes.append(episode)
                episode_start = i + 1
        return episodes
    def __len__(self):
        return len(self.episodes)
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        states = self.dataset['observations'][episode['start']:episode['end']]
        states = np.array([state.flatten() for state in states])    
        actions = self.dataset['actions'][episode['start']+1:episode['end']] + [np.array(0)]
        next_states = self.dataset['observations'][episode['start'] + 1:episode['end'] + 1]
        next_states = np.array([state.flatten() for state in next_states])
        return states, actions, next_states


class MiniGridDataset(Dataset):
    def __init__(self, env_name, max_seq_len=30, transform=None, action_dim=6):
        self.dataset = gym.make(env_name).get_dataset()
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.action_dim = action_dim
        self.episodes = self._get_episodes()
        
    def _get_episodes(self):
        """Split the dataset into episodes based on terminal flags."""
        episodes = []
        episode_start = 0
        
        for i in range(len(self.dataset['terminals'])):
            if self.dataset['terminals'][i] or i == len(self.dataset['terminals']) - 1:
                # Make sure we have enough states for both current and next state
                if (i + 1) - episode_start > 1:  # Need at least 2 states
                    episode = {
                        'start': episode_start,
                        'end': i + 1,  # Include the terminal state
                        'length': i + 1 - episode_start
                    }
                    episodes.append(episode)
                episode_start = i + 1
                
        return episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode = self.episodes[idx]
        states = self.dataset['observations'][episode['start']:episode['end']]
        states = np.array([state.flatten() for state in states])    

        actions = self.dataset['actions'][episode['start']:episode['end']]

        next_states = self.dataset['observations'][episode['start'] + 1:episode['end'] + 1]
        next_states = np.array([state.flatten() for state in next_states])

        return states, actions, next_states

def collate_fn(batch):
    input_states, input_actions, target_states = zip(*batch)
    corrected_acts = []
    for acts in input_actions:
        corrected_acts.append(torch.tensor([float(act.item()) for act in acts]))
    
    input_states = [torch.tensor(state, dtype=torch.float32) for state in input_states]
    input_actions = [torch.tensor(action) for action in corrected_acts]
    target_states = [torch.tensor(target, dtype=torch.float32) for target in target_states]

    input_states = torch.nn.utils.rnn.pad_sequence(input_states, batch_first=True, padding_value=0)
    input_actions = torch.nn.utils.rnn.pad_sequence(input_actions, batch_first=True, padding_value=0)
    target_states = torch.nn.utils.rnn.pad_sequence(target_states, batch_first=True, padding_value=0)

    # Create a mask to indicate valid positions (non-padding tokens)
    mask = (target_states != 0)  # Mask is True for non-padding tokens, False for padding tokens

    return input_states, input_actions.long(), target_states, mask




def save_model_checkpoint(model, epoch, name='model', save_dir="checkpoints"):
    """
    Save the model checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{name}_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved at {checkpoint_path}")



def cluster_and_visualize_codebook(codebook_vectors, labels=None, method='pca', random_state=42, pca_3d=False):
   
    # Step 1: Perform clustering using xmeans
    initial_centers = kmeans_plusplus_initializer(codebook_vectors, 3).initialize()  # Start with 2 clusters
    xmeans_instance = xmeans(data=codebook_vectors, initial_centers=initial_centers)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    centers = np.array(xmeans_instance.get_centers())

    # Generate cluster labels
    cluster_labels = np.zeros(len(codebook_vectors), dtype=int)
    for cluster_idx, cluster in enumerate(clusters):
        cluster_labels[cluster] = cluster_idx
    
    label_map = {}
    for i, j in zip(cluster_labels, labels):
        label_map[j] = i


    num_clusters = len(clusters)

    # Step 2: Dimensionality reduction
    if method == 'pca':
        n_components = 3 if pca_3d else 2
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=random_state)
    reduced_vectors = reducer.fit_transform(codebook_vectors)
    reduced_centers = reducer.transform(centers)

    # Step 3: Visualization
    if pca_3d and method == 'pca':
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for cluster_id in range(num_clusters):
            cluster_points = reduced_vectors[cluster_labels == cluster_id]
            indices = np.where(cluster_labels == cluster_id)[0]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {cluster_id}')
            # Annotate points with labels (if provided)
            if labels is not None:
                for idx, (x, y, z) in zip(indices, cluster_points):
                    print("I am here ###############", labels[idx])
                    ax.text(x, y, z, labels[idx], fontsize=8)
        ax.scatter(reduced_centers[:, 0], reduced_centers[:, 1], reduced_centers[:, 2],
                   color='black', marker='x', s=100, label='Centroids')
        ax.set_title('3D Cluster Visualization')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
    else:
        plt.figure(figsize=(8, 6))
        for cluster_id in range(num_clusters):
            cluster_points = reduced_vectors[cluster_labels == cluster_id]
            indices = np.where(cluster_labels == cluster_id)[0]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')
            # Annotate points with labels (if provided)
            if labels is not None:
                for idx, (x, y) in zip(indices, cluster_points):
                    plt.text(x, y, labels[idx], fontsize=8)
        plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], color='black', marker='x', s=100, label='Centroids')
        plt.title('Cluster Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

    plt.legend()
    plt.savefig('cluster_visualization_with_labels.png')
    plt.show()

    return label_map



from collections import defaultdict
def bifurcate_segments(lists, states, actions):
    segments_dict = defaultdict(list)
    state_action_segs = defaultdict(list)

    for lst, state, action in zip(lists, states, actions):
        current_segment = []
        current_segment_state_action = []

        for i, val in enumerate(lst):
            if not current_segment or val == current_segment[-1]:
                current_segment.append(val)
                current_segment_state_action.append((state[i], action[i]))
            else:
                segments_dict[current_segment[0]].append(current_segment)
                current_segment = [val]

                state_action_segs[current_segment[0]].append(current_segment_state_action)
                current_segment_state_action = [(state[i], action[i])]

        
        if current_segment:
            segments_dict[current_segment[0]].append(current_segment)
            state_action_segs[current_segment[0]].append(current_segment_state_action)
        
        # print(f"Segments dict: {segments_dict.keys()}")


    return dict(segments_dict), dict(state_action_segs)


from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
class DiscreteBCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DiscreteBCPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)  # Output logits for each action
        )

    def forward(self, state):
        return self.fc(state)  # No softmax; handled by CrossEntropyLoss

# Custom dataset class
class BCDataset(Dataset):
    def __init__(self, data):
        self.data = data  # List of (s, a) tuples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action = self.data[idx]
        return torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.long)  # Action as integer index

# Function to train BC policy
def train_bc_policy(dataset, state_dim, action_dim, epochs=50, batch_size=64, lr=1e-3):
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    policy = DiscreteBCPolicy(state_dim, action_dim).cuda()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        policy.train()
        total_loss = 0
        for states, actions in train_loader:
            states, actions = states.cuda(), actions.cuda()
            optimizer.zero_grad()
            logits = policy(states)  # Raw logits
            loss = loss_fn(logits, actions)  # CrossEntropyLoss expects raw logits
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")

    # Evaluate on test set
    policy.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for states, actions in test_loader:
            states, actions = states.cuda(), actions.cuda()
            logits = policy(states)
            test_loss += loss_fn(logits, actions).item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == actions).sum().item()
            total += actions.size(0)

    accuracy = correct / total
    print(f"Test Loss: {test_loss / len(test_loader)}, Accuracy: {accuracy:.4f}")
    return policy

# Main function to train policies
def train_policies(dataset, action_dim):
    policies = {}
    all_data = []

    # Collect data per key and overall
    for k, trajs in dataset.items():
        data = [(item[0][0].flatten(), item[0][1]) for item in trajs]
        all_data.extend(data)
        print(f"Training policy for key {k} with {len(data)} samples")
        policies[k] = train_bc_policy(BCDataset(data), state_dim=len(data[0][0]), action_dim=action_dim)

    # Train a single BC policy on the entire dataset
    print("Training full dataset policy...")
    full_bc_policy = train_bc_policy(BCDataset(all_data), state_dim=len(all_data[0][0]), action_dim=action_dim)

    return full_bc_policy, policies


import torch
import numpy as np

def run_policy_with_mse_discrete(env, full_model, bc_models, episodes=5, step_interval=6):
    mse_results = {name: [] for name in bc_models.keys()}
    all_state_seqs = []

    for ep in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle gym reset API change
            state = state[0]
        done = False
        step = 0

        state_seq = []
        while not done:
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).cuda()
            state_seq.append(env.render())

            # Get full model logits
            full_action_logits = full_model(state_tensor).cpu().detach().numpy()[0]

            if step % step_interval == 0:  # Every 6th step
                for name, bc_model in bc_models.items():
                    # Get BC model logits
                    bc_action_logits = bc_model(state_tensor).cpu().detach().numpy()[0]

                    # Compute MSE between logits directly
                    mse = np.mean((full_action_logits - bc_action_logits) ** 2)
                    mse_results[name].append(mse)

                all_state_seqs.append(state_seq)
                state_seq = []

            # Select action using argmax on full_model logits
            full_action = np.argmax(full_action_logits)
            state, reward, done, _, _ = env.step(full_action)
            step += 1

            if step > 40: 
                done = True

    return mse_results, all_state_seqs



def find_max_occurring_element(arr):
    frequency = Counter(arr)
    max_element = max(frequency, key=frequency.get)
    return max_element



def render_minigrid(agent_pos, agent_orientation, goal_pos):

    import gym
    from gym_minigrid.envs import FourRoomsEnv
    env = FourRoomsEnv(goal_pos=goal_pos)
    obs = env.reset()
    env.agent_pos = agent_pos
    env.agent_dir = agent_orientation
    return env.render(mode='rgb_array')

def get_seq2seq_codes(model, tokenized_trajectories, samples=100, offline_data=None, map_m=None):
    dataloader = get_dataloader(tokenized_trajectories, batch_size=1, offline_data=offline_data, train=False)
    codes = []
    total_samples = 0
    sample_img = []
    mapped_codes = []

    for idx, batch in enumerate(dataloader):
        input_seqs, target_seqs, real_traj = batch
        input_seqs = input_seqs.to('cuda')
        target_seqs = target_seqs.to('cuda')

        with torch.no_grad():
            model.eval()
            predicted_frames, vq_loss, indices = model(input_seqs, None, sampling_probability=0.0)
            indices = indices.reshape((input_seqs.shape[0], input_seqs.shape[1]))
            for j, idx_2 in enumerate(indices):
                mask = input_seqs[j].cpu().numpy() != 0
                masked_idx_2 = idx_2.cpu().numpy()[mask]
                # print("offline seq {} input seq {}".format(len(offline_data[idx]), len(input_seqs[j])))

                if map_m is not None:
                    mapped_idx = [map_m[item] for item in masked_idx_2]
                    mapped_codes.append(mapped_idx)
                else:
                    codes.append(masked_idx_2)


    if map_m is not None:
        return codes, mapped_codes, sample_img
    return codes, sample_img

def get_sequence_codes(model, dataset):
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)
    codes = []
    for idx, batch in enumerate(dataloader):
        states, actions, target_states, mask = batch
        states = states.to('cuda')
        actions = actions.to('cuda').long()
        target_states = target_states.to('cuda')

        with torch.no_grad():
            model.eval()
            predicted_frames, vq_loss, indices = model(states, actions, target_states, sampling_probability=0.0)
            indices = indices.reshape((states.shape[0], states.shape[1]))
            valid_mask = mask.any(dim=2)
            for i, idx in enumerate(indices):
                valid_indices = idx[valid_mask[i]].tolist()
                codes.append(valid_indices)  
    return codes



def render_trajectory(traj, k, grid_size=(9, 9), goal_pos=None, lava_pos=None, walls_pos=None):
    directions = {0: '<', 1: '>', 2: '->', 3: '<'}
    agent_positions = traj['agent_pos']
    agent_directions = traj['agent_dir']
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, grid_size[1] - 0.5)
    ax.set_ylim(-0.5, grid_size[0] - 0.5)
    ax.set_xticks(range(grid_size[1]))
    ax.set_yticks(range(grid_size[0]))
    ax.grid(color='gray', linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')

    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='black'))

    if walls_pos:
        for wall in walls_pos:
            ax.add_patch(plt.Rectangle((wall[0], wall[1]), 1, 1, color='gray'))

    if lava_pos:
        for lava in lava_pos:
            ax.add_patch(plt.Rectangle((lava[0],lava[1]), 1, 1, color='red'))

    for i, (pos, direction) in enumerate(zip(agent_positions, agent_directions)):
        ax.text(pos[0], pos[1], f"{directions[direction]},{i}", 
                ha='center', va='center', color='white', 
                bbox=dict(boxstyle="round", facecolor='red' if i == len(agent_positions) - 1 else 'gray', edgecolor='none'))
    
    if goal_pos:
        for goal in goal_pos:
            ax.add_patch(plt.Rectangle((goal[0], goal[1]), 
                                    1, 1, color="lime", alpha=0.7))
    
    plt.savefig(f'trajectory_{k}.png')


def diversity_loss(embeddings):
    similarity = torch.matmul(embeddings, embeddings.t())
    identity = torch.eye(embeddings.shape[0], device=embeddings.device)
    return F.mse_loss(similarity, identity)

def entropy_regularization(quantized_indices, num_embeddings):
    code_probs = F.one_hot(quantized_indices, num_embeddings).float().mean(0)
    entropy = -(code_probs + 1e-8).log() * code_probs
    return -entropy.sum() 