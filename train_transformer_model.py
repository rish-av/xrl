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


os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

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
        return len(self.states) - self.seq_len

    def __getitem__(self, idx):
        states_seq = self.states[idx:idx + self.seq_len]
        actions_seq = self.actions[idx:idx + self.seq_len]
        next_states_seq = self.states[idx + 1:idx + self.seq_len + 1]    

        return states_seq, actions_seq, next_states_seq

def create_d4rl_dataloader(env_name='halfcheetah-medium-v2', batch_size=32, seq_len=50):
    dataset = D4RLDataset(env_name=env_name, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train(model, dataloader, num_epochs=10, learning_rate=1e-4):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for states, actions, next_states in dataloader:
            states, actions, next_states = states.to(device), actions.to(device), next_states.to(device)
            optimizer.zero_grad()
            predicted_states, quantization_loss = model(states, actions, next_states)
            state_loss = mse_loss_fn(predicted_states, next_states)
            loss = state_loss + quantization_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            wandb.log({
                'state loss':state_loss,
                'quant loss': quantization_loss,
                'total loss': loss
            })
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


def plot_encoding_indices(model, states, actions):
    model.eval()
    with torch.no_grad():
        encoded_trajectory = model.encoder(states, actions)
        _, _, encoding_indices = model.quantizer(encoded_trajectory)
    
    encoding_indices = encoding_indices.view(states.size(0), states.size(1)) 
    encoding_indices_np = encoding_indices.cpu().numpy() 
    
    plt.figure(figsize=(10, 6))
    for i in range(encoding_indices_np.shape[0]):
        plt.plot(range(encoding_indices_np.shape[1]), encoding_indices_np[i], label=f"Trajectory {i+1}")
    
    plt.xlabel("Time Step (t)")
    plt.ylabel("Encoding Indices")
    plt.title("Encoding Indices for Each (State, Action) Pair")
    plt.legend()
    plt.show()


batch_size = 50
seq_len = 50
state_dim = 17  
action_dim = 6  
num_embeddings = 15  
hidden_dim = 128
num_heads = 4
num_layers = 6
max_len = 100

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dataloader = create_d4rl_dataloader(env_name='halfcheetah-medium-v2', batch_size=batch_size, seq_len=seq_len)
model = TrajectoryTransformerVQ(state_dim, action_dim, hidden_dim, num_heads, num_layers, max_len, num_embeddings).to(device)
torch.save(model.state_dict(), 'transformer_vq.pth')
train(model, dataloader, 10)
