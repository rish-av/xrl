import gym
import d4rl
import torch
from torch.utils.data import Dataset, DataLoader

# Load the environment and dataset
env = gym.make('halfcheetah-medium-v2')
dataset = env.get_dataset()

# Prepare state-action pairs and next states
states = dataset['observations'][:100000]
actions = dataset['actions'][:100000]
next_states = dataset['next_observations'][:100000]


import torch
import torch.nn as nn 

def generate_causal_mask(size):
    mask = torch.tril(torch.ones(size, size))  # Lower triangular matrix
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Seq2SeqDataset(Dataset):
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
        
        # Combine state and action sequences for the input
        state_action_seq = torch.cat((torch.tensor(state_seq), torch.tensor(action_seq)), dim=-1)
        return state_action_seq, torch.tensor(next_state_seq)

# Create dataset and dataloader
sequence_length = 25
dataset = Seq2SeqDataset(states, actions, next_states, sequence_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


import torch
import torch.nn as nn
import torch.nn.functional as F

# Define EOS token ID (assuming it is distinct from other tokens)
EOS_TOKEN_ID = -1  # Use a value not in the input range to denote EOS

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]

# Generate Causal Mask
def generate_causal_mask(size):
    mask = torch.tril(torch.ones(size, size))  # Lower triangular matrix
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# EMA VQ Layer with VQ Loss
class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, epsilon=1e-5, commitment_cost=0.25):
        super(EMAVectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon
        self.commitment_cost = commitment_cost

        # Codebook for vector quantization
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.cluster_size = nn.Parameter(torch.zeros(num_embeddings), requires_grad=False)
        self.ema_w = nn.Parameter(self.embedding.clone(), requires_grad=False)

    def forward(self, x):
        # Flatten input to (batch_size * seq_len, embedding_dim)
        flat_x = x.reshape(-1, self.embedding_dim)
        
        # Compute distances and get encoding indices
        distances = torch.cdist(flat_x, self.embedding)
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding[encoding_indices].view(x.shape)
        
        # Compute VQ Loss (Codebook loss + Commitment loss)
        codebook_loss = F.mse_loss(quantized.detach(), x)  # Codebook loss
        commitment_loss = F.mse_loss(quantized, x.detach())  # Commitment loss
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # EMA updates for embedding (codebook)
        if self.training:
            encoding_one_hot = F.one_hot(encoding_indices, self.num_embeddings).type_as(flat_x)
            new_cluster_size = encoding_one_hot.sum(dim=0)
            self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)
            dw = encoding_one_hot.t() @ flat_x
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            # Normalize to prevent embedding drift
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            self.embedding.data = self.ema_w / cluster_size.unsqueeze(1)
        
        # Straight-through gradient
        quantized = x + (quantized - x).detach()
        
        return quantized, vq_loss, encoding_indices  # Return quantized output and VQ loss

# Encoder with Causal Mask, EMA VQ, and EOS Embedding
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_embeddings):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.eos_embedding = nn.Parameter(torch.randn(1, 1, model_dim))  # Learnable EOS embedding
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.vq_layer = EMAVectorQuantizer(num_embeddings, model_dim)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding(seq_len)
        
        # Append EOS embedding at the end of sequence
        eos_embedding_expanded = self.eos_embedding.expand(x.size(0), -1, -1)  # Match batch size
        x = torch.cat([x, eos_embedding_expanded], dim=1)  # Append EOS token to each sequence
        
        # Apply causal mask to encoder
        causal_mask = generate_causal_mask(x.size(1)).to(x.device)
        x = self.transformer_encoder(x.permute(1, 0, 2), mask=causal_mask).permute(1, 0, 2)
        
        # Apply EMA VQ layer to quantize encoder outputs and get VQ loss
        quantized, vq_loss, encidx = self.vq_layer(x)
        
        return quantized[:, -1, :], vq_loss, encidx  # Return only the last quantized state with VQ loss

# Decoder with Causal Mask
class TransformerDecoder(nn.Module):
    def __init__(self, model_dim, output_dim, num_heads, num_layers):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(output_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, tgt, memory):
        seq_len = tgt.size(1)
        tgt = self.embedding(tgt) + self.positional_encoding(seq_len)
        
        # Apply causal mask to decoder
        causal_mask = generate_causal_mask(seq_len).to(tgt.device)
        tgt = self.transformer_decoder(tgt.permute(1, 0, 2), memory.unsqueeze(0), tgt_mask=causal_mask)
        
        return self.fc_out(tgt.permute(1, 0, 2))

# Seq2Seq Transformer with EMA VQ and EOS Embedding
class Seq2SeqTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, output_dim, num_heads, num_encoder_layers, num_decoder_layers, num_embeddings):
        super(Seq2SeqTransformer, self).__init__()
        self.encoder = TransformerEncoder(input_dim, model_dim, num_heads, num_encoder_layers, num_embeddings)
        self.decoder = TransformerDecoder(model_dim, output_dim, num_heads, num_decoder_layers)
    
    def forward(self, src, tgt):
        memory, vq_loss, encidx = self.encoder(src)  # Only the final quantized state and VQ loss from encoder
        output = self.decoder(tgt, memory)  # Decode future states from the quantized encoder output
        return output, vq_loss, encidx  # Return both the prediction output and VQ loss







# Hyperparameters
input_dim = states.shape[1] + actions.shape[1]  # state + action dimensions
model_dim = 256        # Dimension of transformer model
output_dim = states.shape[1]       # Dimension of output features (matching input_dim if predicting next state)
num_heads = 8          # Number of attention heads
num_encoder_layers = 4 # Number of encoder layers
num_decoder_layers = 4 # Number of decoder layers
num_embeddings = 64   # Number of VQ embeddings
batch_size = 32
learning_rate = 1e-4
num_epochs = 200 

# Initialize the model, loss function, and optimizer
model = Seq2SeqTransformer(input_dim, model_dim, output_dim, num_heads, num_encoder_layers, num_decoder_layers, num_embeddings).to('cuda')
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
model.load_state_dict(torch.load('seq2seq_eos_transformer.pth'))
for epoch in range(num_epochs):
    epoch_loss = 0
    for src, tgt in dataloader:
        optimizer.zero_grad()
        
        # Prepare inputs and targets
        src = src.to('cuda')

        tgt_input = tgt[:, :-1, :].to('cuda')  # Use all but last next state as target input for the decoder
        tgt_output = tgt[:, 1:, :].to('cuda')  # Shifted target sequence to predict next state
        
        # Forward pass
        output, vq_loss, encidx = model(src, tgt_input)
        
        # Compute loss
        loss = criterion(output, tgt_output) + vq_loss
        loss.backward()
        # print(loss.item(), vq_loss.item(), torch.unique(encidx))
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader)} num unique IDs {len (torch.unique(encidx))}")

torch.save(model.state_dict(), 'seq2seq_eos_transformer.pth')