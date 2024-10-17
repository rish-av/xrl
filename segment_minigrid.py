import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
import d4rl
from models import VQ_VAE_Segment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQ_VAE_Segment(state_dim=147, action_dim=0, latent_dim=32, num_embeddings=512).to(device)
model.load_state_dict(torch.load('vq_vae_minigrid.pth')) 
model.eval()

env = gym.make('minigrid-fourrooms-v0')
dataset = env.get_dataset()
def process_trajectories(dataset):
    states = dataset['observations']
    actions = dataset['actions']
    dones = dataset['terminals']
    
    trajectories = []
    trajectory = []
    
    for i in range(len(states)):
        # print(states[i].flatten().shape, actions[i])
        traj_step = np.concatenate([states[i].flatten()], axis=-1)
        trajectory.append(traj_step)
        if dones[i]:
            trajectories.append(torch.tensor(trajectory, dtype=torch.float32))
            trajectory = [] 
    return trajectories
trajectories = process_trajectories(dataset)

def segment_trajectory(traj, model):
    with torch.no_grad():
        traj = traj.unsqueeze(0).to(device)  # Add batch dimension
        _, _, encoding_indices = model(traj)
    return encoding_indices.cpu().numpy()


def plot_segmentation_multiple(encoding_indices_list, num_trajectories):
    plt.figure(figsize=(12, 8))
    
    # cumulative_timestep = 0 
    for traj_num, encoding_indices in enumerate(encoding_indices_list):
        timesteps = range(0, len(encoding_indices))
        plt.plot(timesteps, encoding_indices.flatten(), label=f'Sequence {traj_num + 1}')
        # cumulative_timestep += len(encoding_indices)
    plt.title(f"Segmentation trajectories based on VQ-VAE Encoding Indices")
    plt.xlabel("Cumulative Timestep")
    plt.ylabel("Encoding Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig('seg_minigrid.png')
    plt.show()

encoding_indices_list = []
for traj_num, traj in enumerate(trajectories[:20]):  
    encoding_indices = segment_trajectory(traj, model)
    encoding_indices_list.append(encoding_indices)
plot_segmentation_multiple(encoding_indices_list, num_trajectories=len(encoding_indices_list))