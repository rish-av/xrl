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