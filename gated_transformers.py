import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import d4rl


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

class GTrXLEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(GTrXLEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.gate = nn.Parameter(torch.ones(d_model))  # gating mechanism
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention with gating
        attn_output, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        gated_attn = self.gate * attn_output  # Apply gating
        x = x + self.dropout(gated_attn)
        x = self.norm1(x)
        
        # Feed-forward layer with gating
        ffn_output = self.ffn(x)
        gated_ffn = self.gate * ffn_output
        x = x + self.dropout(gated_ffn)
        x = self.norm2(x)
        
        return x

class TransformerEncoderGTrXL(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_heads, num_layers, max_len):
        super(TransformerEncoderGTrXL, self).__init__()
        self.embedding_mlp = nn.Linear(state_dim + action_dim, hidden_dim)
        self.eos_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len + 1, hidden_dim))
        self.layers = nn.ModuleList([GTrXLEncoderLayer(hidden_dim, num_heads) for _ in range(num_layers)])
    
    def forward(self, states, actions):
        batch_size, seq_len = states.size(0), states.size(1)
        state_action_cat = torch.cat([states, actions], axis=-1)
        embeddings = self.embedding_mlp(state_action_cat)
        embeddings = torch.cat([embeddings, self.eos_token.expand(batch_size, -1, -1)], dim=1)
        embeddings = embeddings + self.positional_encoding[:, :seq_len + 1, :]
        
        for layer in self.layers:
            embeddings = layer(embeddings)
        
        eos_representation = embeddings[:, -1, :]
        return eos_representation

class TransformerDecoderGTrXL(nn.Module):
    def __init__(self, hidden_dim, state_dim, num_heads, num_layers, max_len):
        super(TransformerDecoderGTrXL, self).__init__()
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.layers = nn.ModuleList([GTrXLEncoderLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.fc_state = nn.Linear(hidden_dim, state_dim)
        self.max_len = max_len

    def forward(self, quantized_eos_embedding, target_states=None):
        batch_size = quantized_eos_embedding.size(0)
        decoded_states = []
        decoder_input = quantized_eos_embedding.unsqueeze(1)  # Initial input

        for t in range(self.max_len):
            if t > 0:
                decoder_input = torch.cat([decoder_input, self.state_embedding(decoded_states[-1]).unsqueeze(1)], dim=1)
            
            for layer in self.layers:
                decoder_input = layer(decoder_input)

            next_state = self.fc_state(decoder_input[:, -1, :])
            decoded_states.append(next_state)

        return torch.stack(decoded_states, dim=1)

class TrajectoryTransformerGTrXLVQVAE(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_heads, num_layers, max_len, num_embeddings):
        super(TrajectoryTransformerGTrXLVQVAE, self).__init__()
        self.encoder = TransformerEncoderGTrXL(state_dim, action_dim, hidden_dim, num_heads, num_layers, max_len)
        self.quantizer = VectorQuantizer(num_embeddings, hidden_dim)
        self.decoder = TransformerDecoderGTrXL(hidden_dim, state_dim, num_heads, num_layers, max_len)

    def forward(self, states, actions, target_states=None):
        encoded_trajectory = self.encoder(states, actions)
        quantized, quantization_loss, idx = self.quantizer(encoded_trajectory)
        print(idx)
        predicted_states = self.decoder(quantized, target_states)
        return predicted_states, quantization_loss

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


STATE_DIM = 64
ACTION_DIM = 16
HIDDEN_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 3
MAX_LEN = 50
NUM_EMBEDDINGS = 512
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instantiate the model
model = TrajectoryTransformerGTrXLVQVAE(STATE_DIM, ACTION_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, MAX_LEN, NUM_EMBEDDINGS)
model = model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
        # Calculate current start index based on `current_idx`
        start_idx = self.current_idx
        end_idx = start_idx + self.seq_len
        
        states_seq = self.states[start_idx:end_idx]
        actions_seq = self.actions[start_idx:end_idx]
        target_states_seq = self.states[start_idx + 1:end_idx + 1]
        
        # Update `current_idx` for the next batch
        self.current_idx = (self.current_idx + self.seq_len) % (len(self.states) - self.seq_len)
        
        return states_seq, actions_seq, target_states_seq
    

def create_dataloader(env_name='halfcheetah-medium-v2', batch_size=32, seq_len=60):
    dataset = D4RLDataset(env_name=env_name, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

dataloader = create_dataloader(env_name='halfcheetah-medium-v2', batch_size=32, seq_len=60)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for batch_idx, (states, actions, target_states) in enumerate(dataloader):
        states, actions, target_states = states.to(DEVICE), actions.to(DEVICE), target_states.to(DEVICE)

        # Forward pass
        predicted_states, quantization_loss = model(states, actions, target_states)

        # Calculate reconstruction loss
        reconstruction_loss = F.mse_loss(predicted_states, target_states)

        # Combine losses
        loss = reconstruction_loss + quantization_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()

        # Logging
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(dataloader)}], "
                  f"Reconstruction Loss: {reconstruction_loss.item():.4f}, "
                  f"Quantization Loss: {quantization_loss.item():.4f}, "
                  f"Total Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Average Loss: {avg_loss:.4f}")

print("Training complete.")