import torch
import torch.optim as optim
import torch.nn.functional as F
import d4rl  # Import D4RL
import gym
import numpy as np
from models import VQ_VAE_Segment
# Hyperparameters
learning_rate = 1e-3
epochs = 1000
batch_size = 256
seq_len = 50 
step_size = 200  
gamma = 0.5 

state_dim = 17  
action_dim = 6   
latent_dim = 32  
num_embeddings = 512  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQ_VAE_Segment(state_dim, action_dim, latent_dim, num_embeddings).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

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

traj_data = create_trajectories(dataset, seq_len)
def data_loader(traj_data, batch_size):
    for i in range(0, len(traj_data), batch_size):
        yield torch.stack(traj_data[i:i+batch_size]).to(device)

model.train()
for epoch in range(epochs):
    epoch_loss = 0
    batch_idx = 0
    for traj_batch in data_loader(traj_data, batch_size):
        optimizer.zero_grad() 
        reconstructed_traj, vq_loss, encoding_indices = model(traj_batch)
        recon_loss = F.mse_loss(reconstructed_traj, traj_batch)
        total_loss = recon_loss + vq_loss
        # print(f'Recon loss {recon_loss}, Vq loss {vq_loss}')
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
        batch_idx += 1
    scheduler.step()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / batch_idx:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')


torch.save(model.state_dict(), 'vq_vae.pth')
print("Training complete!")