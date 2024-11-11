import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import d4rl
from models import VQ_VAE_Segment
from torch.nn.utils.rnn import pad_sequence

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

print(f"Total trajectories found {len(trajectories)}")
def data_loader(trajectories, batch_size):
    for i in range(0, len(trajectories), batch_size):
        batch = trajectories[i:i + batch_size]
        padded_batch = pad_sequence(batch, batch_first=True)
        yield padded_batch

state_dim = 147 #flattened obs 
action_dim = 0
latent_dim = 32  
num_embeddings = 512 
learning_rate = 1e-3
batch_size = 8
epochs = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VQ_VAE_Segment(state_dim, action_dim, latent_dim, num_embeddings).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
def train(model, trajectories, epochs, batch_size):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        batch_idx = 0

        for traj_batch in data_loader(trajectories, batch_size):
            traj_batch = traj_batch.to(device)
            optimizer.zero_grad()
            recon_traj, vq_loss, encoding_indices = model(traj_batch)
            recon_loss = F.mse_loss(recon_traj, traj_batch)
            total_loss = recon_loss + vq_loss
            total_loss.backward()
            optimizer.step()
            total_loss += total_loss.item()
            batch_idx += 1

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / batch_idx:.4f}')
train(model, trajectories, epochs, batch_size)
torch.save(model.state_dict(),'vq_vae_minigrid.pth')