import torch
import torch.nn as nn
import torch.nn.functional as F
import d4rl
import os

from datetime import datetime

device  = 'cuda' if torch.cuda.is_available() else 'cpu'
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--track', action='store_true')
args = parser.parse_args()


os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
if args.track:
    import wandb
    print("Tracking with wandb!")
    run_name = f'traj_trans_no_quant' + datetime.now().strftime("%d-%m-%y-%H-%M-%S")
    wandb.init(project='transformer_vq', entity='mail-rishav9', name=run_name)
    wandb.config.update({
        'seq_len':60,
        'batch_size':50
    })

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs, beta=0.01):
        inputs_flattened = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(inputs_flattened ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(inputs_flattened, self.embeddings.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view_as(inputs)
        commitment_loss = F.mse_loss(quantized.detach(), inputs) + beta * F.mse_loss(quantized, inputs.detach())
        quantized = inputs + (quantized - inputs).detach()
        return quantized, commitment_loss, encoding_indices

class TransformerEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_heads, num_layers, max_len):
        super(TransformerEncoder, self).__init__()
        self.embedding_mlp = nn.Linear(state_dim + action_dim, hidden_dim)
        self.eos_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len + 1, hidden_dim))
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers, batch_first=True)
        self.num_heads = num_heads
    
    def generate_causal_mask(self, batch_size, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).bool().to(device)
        causal_mask = mask.unsqueeze(0).expand(self.num_heads * batch_size, -1, -1)
        return causal_mask

    
    def forward(self, states, actions):
        batch_size, seq_len = states.size(0), states.size(1)
        state_action_cat = torch.cat([states, actions], axis=-1)
        embeddings = self.embedding_mlp(state_action_cat)
        eos_token_expanded = self.eos_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([embeddings, eos_token_expanded], dim=1)
        embeddings = embeddings + self.positional_encoding[:, :seq_len + 1, :]
        causal_mask = self.generate_causal_mask(batch_size, seq_len + 1)
        encoded_trajectory = self.transformer(embeddings, embeddings, src_is_causal=True, src_mask=causal_mask)
        eos_representation = encoded_trajectory[:, -1, :]
        return eos_representation

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, state_dim, num_heads, num_layers, max_len):
        super(TransformerDecoder, self).__init__()
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.transformer_decoder = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_decoder_layers=num_layers, batch_first=True)
        self.fc_state = nn.Linear(hidden_dim, state_dim)
        self.max_len = max_len

    def forward(self, quantized_eos_embedding, target_states=None):
        batch_size = quantized_eos_embedding.size(0)
        decoded_states = []
        decoder_input = quantized_eos_embedding.unsqueeze(1)  # Initial input

        for t in range(self.max_len):
            if t > 0:
                decoder_input = torch.cat([decoder_input, self.state_embedding(decoded_states[-1]).unsqueeze(1)], dim=1)
            # Apply the causal mask to prevent future tokens from being attended to
            tgt_mask = torch.triu(torch.ones(t + 1, t + 1), diagonal=1).bool().to(quantized_eos_embedding.device)
            decoded_output = self.transformer_decoder(decoder_input, decoder_input, tgt_mask=tgt_mask)
            next_state = self.fc_state(decoded_output[:, -1, :])
            decoded_states.append(next_state)

        return torch.stack(decoded_states, dim=1)


class TrajectoryTransformerVQVAE(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_heads, num_layers, max_len, num_embeddings):
        super(TrajectoryTransformerVQVAE, self).__init__()
        self.encoder = TransformerEncoder(state_dim, action_dim, hidden_dim, num_heads, num_layers, max_len)
        self.quantizer = VectorQuantizer(num_embeddings, hidden_dim)
        self.decoder = TransformerDecoder(hidden_dim, state_dim, num_heads, num_layers, max_len)

    def forward(self, states, actions, target_states=None):
        encoded_trajectory = self.encoder(states, actions)
        quantized, quantization_loss, idx = self.quantizer(encoded_trajectory)
        predicted_states = self.decoder(encoded_trajectory, target_states)
        return predicted_states, quantization_loss
    

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define Dataset class for D4RL trajectories
class D4RLDataset(Dataset):
    def __init__(self, env_name='halfcheetah-medium-v2', seq_len=50):
        env = gym.make(env_name)
        dataset = env.get_dataset()
        self.states = torch.tensor(dataset['observations'], dtype=torch.float32)
        self.actions = torch.tensor(dataset['actions'], dtype=torch.float32)
        self.seq_len = seq_len
        self.current_idx = 0  # Start index

    def __len__(self):
        return (len(self.states) - self.seq_len) // self.seq_len

    def __getitem__(self, idx):
        start_idx = self.current_idx
        end_idx = start_idx + self.seq_len
        
        states_seq = self.states[start_idx:end_idx]
        actions_seq = self.actions[start_idx:end_idx]
        target_states_seq = self.states[start_idx + 1:end_idx + 1]
        self.current_idx = (self.current_idx + self.seq_len) % (len(self.states) - self.seq_len)
        
        return states_seq, actions_seq, target_states_seq


def train_model(model, dataloader, num_epochs=10, learning_rate=1e-4, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss_fn = nn.MSELoss()
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_state_loss, total_quant_loss = 0, 0, 0
        for states, actions, target_states in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            states, actions, target_states = states.to(device), actions.to(device), target_states.to(device)
            optimizer.zero_grad()
            predicted_states, quantization_loss = model(states, actions, target_states)
            state_loss = mse_loss_fn(predicted_states, target_states)
            loss = state_loss + quantization_loss
            loss.backward()
            optimizer.step()

            if args.track:
                wandb.log({
                    'state loss': state_loss.item(),
                    'quantization loss': quantization_loss.item(),
                    'total loss': loss.item()
                })
            
            total_loss += loss.item()
            total_state_loss += state_loss.item()
            total_quant_loss += quantization_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_state_loss = total_state_loss / len(dataloader)
        avg_quant_loss = total_quant_loss / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {avg_loss:.4f}, State Loss: {avg_state_loss:.4f}, Quantization Loss: {avg_quant_loss:.4f}")


def create_dataloader(env_name='halfcheetah-medium-v2', batch_size=32, seq_len=50):
    dataset = D4RLDataset(env_name=env_name, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


state_dim = 17           
action_dim = 6          
hidden_dim = 8         
num_heads = 4            
num_layers = 2           
max_len = 60             
num_embeddings = 15     

# Initialize model, dataloader, and device
model = TrajectoryTransformerVQVAE(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=hidden_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    max_len=max_len,
    num_embeddings=num_embeddings
)

dataloader = create_dataloader(env_name='halfcheetah-medium-v2', batch_size=32, seq_len=max_len)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Train the model
train_model(model, dataloader, num_epochs=500, learning_rate=1e-4, device=device)
torch.save(model.state_dict(), f'transformer_vq_new_2.pth')