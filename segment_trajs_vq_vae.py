import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
import d4rl
from models import VQ_VAE_Segment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQ_VAE_Segment(state_dim=17, action_dim=6, latent_dim=32, num_embeddings=512).to(device)
model.load_state_dict(torch.load('vqvae_model.pth')) 
model.eval()

env = gym.make('halfcheetah-medium-replay-v2')
dataset = env.get_dataset()
def create_trajectories(dataset, seq_len):
    states = dataset['observations']
    actions = dataset['actions']
    num_items = len(states)
    
    traj_data = []
    for i in range(0, num_items - seq_len, seq_len):
        state_seq = states[i:i+seq_len]
        action_seq = actions[i:i+seq_len]
        traj = torch.tensor(np.concatenate([state_seq, action_seq], axis=-1), dtype=torch.float32)
        traj_data.append(traj)
    return traj_data

seq_len = 50
traj_data = create_trajectories(dataset, seq_len)

def segment_trajectory(traj, model):
    with torch.no_grad():
        traj = traj.unsqueeze(0).to(device)  # Add batch dimension
        _, _, encoding_indices = model(traj)
    return encoding_indices.cpu().numpy()

def plot_segmentation(encoding_indices, traj_num):
    plt.figure(figsize=(10, 6))
    plt.plot(encoding_indices.flatten(), label='Encoding Indices')
    plt.title(f"Trajectory {traj_num} - Segmentation based on VQ-VAE Encoding Indices")
    plt.xlabel("Timestep")
    plt.ylabel("Encoding Index")
    plt.legend()
    plt.show()

for traj_num, traj in enumerate(traj_data[:5]): 
    encoding_indices = segment_trajectory(traj, model)
    plot_segmentation(encoding_indices, traj_num)