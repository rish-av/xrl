import d4rl
import gym
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from transformer_models import TrajectoryTransformerVQ
from torch.utils.data import DataLoader, Dataset
import os
import wandb
from datetime import datetime
import cv2

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--track', action='store_true')
args = parser.parse_args()


os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
if args.track:
    run_name = f'traj_trans' + datetime.now().strftime("%d-%m-%y-%H-%M-%S")
    wandb.init(project='xrl_lstm_seqs', entity='mail-rishav9', name=run_name)
    wandb.config.update({
        'seq_len':50,
        'batch_size':50
    })

class D4RLDataset(Dataset):
    def __init__(self, env_name='halfcheetah-medium-v2', seq_len=50):
        env = gym.make(env_name)
        dataset = env.get_dataset()
        self.states = torch.tensor(dataset['observations'], dtype=torch.float32)
        self.actions = torch.tensor(dataset['actions'], dtype=torch.float32)
        self.next_states = torch.tensor(dataset['next_observations'], dtype=torch.float32)
        self.rewards = torch.tensor(dataset['rewards'], dtype=torch.float32)
        self.dones = torch.tensor(dataset['terminals'], dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return 50000 - self.seq_len

    def __getitem__(self, idx):
        states_seq = self.states[idx:idx + self.seq_len]            # 0-49
        actions_seq = self.actions[idx:idx + self.seq_len]
        target_states_seq = self.states[idx + 1:idx + self.seq_len + 1]   # 1-50
        
        return states_seq, actions_seq, target_states_seq

def create_d4rl_dataloader(env_name='halfcheetah-medium-v2', batch_size=256, seq_len=50):
    dataset = D4RLDataset(env_name=env_name, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train(model, dataloader, num_epochs=2, learning_rate=1e-4, track=False):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_state_loss = 0
        total_quant_loss = 0

        for states, actions, target_states in dataloader:
            states, actions, target_states = states.to(device), actions.to(device), target_states.to(device)
            optimizer.zero_grad()
            
            # Forward pass through the model
            predicted_states, quantization_loss = model(states, actions, target_states)
            
            # Calculate state prediction loss
            state_loss = mse_loss_fn(predicted_states, target_states)
            
            # Combine losses
            loss = state_loss + quantization_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_state_loss += state_loss.item()
            total_quant_loss += quantization_loss.item()
            if track:
                import wandb
                wandb.log({
                    'state loss': state_loss.item(),
                    'quantization loss': quantization_loss.item(),
                    'total loss': loss.item()
                })
            # print(state_loss, quantization_loss, loss)
        
        # Calculate average losses
        avg_loss = total_loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


import matplotlib.pyplot as plt
import torch


def plot_encoding_indices(model, dataloader, device):
    model.eval()
    indices = []
    
    for i in range(50):
        with torch.no_grad():
            states, actions, _ = next(iter(dataloader))
            states, actions = states.to(device), actions.to(device)
            
            # Forward pass through encoder and quantizer
            encoded_trajectory = model.encoder(states, actions)
            _, _, encoding_indices = model.quantizer(encoded_trajectory)
        
        # Exclude EOS token index by selecting only up to `seq_len`
        # encoding_indices = encoding_indices[:, :-1].view(states.size(0), states.size(1))
        encoding_indices_np = encoding_indices.cpu().numpy()
        indices.append(encoding_indices_np)
    
    plt.figure(figsize=(10, 6))
    for item in indices:
        plt.plot(range(item.shape[0]), item, label=f"Trajectory {i+1}")
    
    plt.xlabel("Time Step (t)")
    plt.ylabel("Encoding Indices")
    plt.title("Encoding Indices for Each (State, Action) Pair")
    plt.show()
    plt.savefig('transformer_indices2.png')




batch_size = 50
seq_len = 50
state_dim = 17  
action_dim = 6  
num_embeddings = 32  
hidden_dim = 16
num_heads = 4
num_layers = 6
max_len = 100

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dataloader = create_d4rl_dataloader(env_name='halfcheetah-medium-v2', batch_size=batch_size, seq_len=seq_len)
model = TrajectoryTransformerVQ(state_dim, action_dim, hidden_dim, num_heads, num_layers, max_len, num_embeddings).to(device)
train(model, dataloader, 100)
torch.save(model.state_dict(), 'transformer_vq_100.pth')

# model.load_state_dict(torch.load('/home/rishav/scratch/xrl/transformer_vq_100.pth'))
# with torch.no_grad():
#     model.eval()

#     #     plot_encoding_indices(model, dataloader, device)

#     env = gym.make("HalfCheetah-v2")
#     def save_halfcheetah_state_as_image(env, state, filename="halfcheetah_state.png"):
#         env.reset()
#         env.env.sim.set_state_from_flattened(np.insert(state, 0, [0]))
#         env.env.sim.forward()
#         img = env.render(mode="rgb_array")

#         cv2.imwrite(filename, img)


#     for states, actions, target_states in dataloader:
#         states, actions, target_states = states.to(device), actions.to(device), target_states.to(device)
#         predicted_states, quantization_loss = model(states, actions, target_states)

#         for i in range(len(states[0])-1):
#             save_halfcheetah_state_as_image(env, predicted_states[0][i].cpu().numpy(), f'visuals/halfcheetah_pred_{i}.png')
#             save_halfcheetah_state_as_image(env, states[0][i+1].cpu().numpy(), f'visuals/halfcheetah_real_{i}.png')
#         break

#         print("\n")
#         print(f'state {states[0][1]}')
#         print(f'state {predicted_states[0][0]}')
#         print("\n")