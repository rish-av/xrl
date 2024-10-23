import torch
import torch.optim as optim
import torch.nn.functional as F
import d4rl  # Import D4RL
import gym
import numpy as np
from models import VQ_VAE_Segment, VQ_Many2One
from torch.utils.data import random_split

# Hyperparameters
learning_rate = 1e-3
epochs = 500
batch_size = 256
seq_len = 16 
step_size = 200  

state_dim = 17  
action_dim = 6   
latent_dim = 32  
num_embeddings = 128  
hidden_dim  = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQ_Many2One(state_dim, action_dim, hidden_dim, latent_dim, num_embeddings).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
env = gym.make('halfcheetah-medium-v2')
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

traj_data = create_trajectories(dataset, seq_len)
train_size = int(0.9 * len(traj_data))
val_size = len(traj_data) - train_size
train_data, val_data = random_split(traj_data, [train_size, val_size])

def data_loader(subset, batch_size):
    data_list = list(subset)  
    for i in range(0, len(data_list), batch_size):
        yield torch.stack(data_list[i:i + batch_size]).to(device)


model.train()
for epoch in range(epochs):
    epoch_loss = 0
    batch_idx = 0
    for traj_batch in data_loader(train_data, batch_size):
        optimizer.zero_grad() 
        reconstructed_traj, vq_loss, encoding_indices = model(traj_batch, seq_len)
        recon_loss = F.mse_loss(reconstructed_traj, traj_batch)
        total_loss = recon_loss + vq_loss
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
        batch_idx += 1
    # scheduler.step()

    avg_train_loss = epoch_loss / batch_idx
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, LR: {learning_rate}')
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_batch_idx = 0
        for val_batch in data_loader(val_data, batch_size):
            reconstructed_traj, vq_loss, encoding_indices = model(val_batch, seq_len)
            recon_loss = F.mse_loss(reconstructed_traj, val_batch)
            total_loss = recon_loss + vq_loss
            val_loss += total_loss.item()
            val_batch_idx += 1
        avg_val_loss = val_loss / val_batch_idx
        print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}')

    model.train()

torch.save(model.state_dict(), 'vq_m2o_halfcheetah.pth')
print("Training complete!")