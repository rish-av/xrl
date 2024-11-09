import torch
import torch.nn as nn
import torch.optim as optim
import gym
import d4rl
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from argparse import ArgumentParser
import os
from datetime import datetime

parser = ArgumentParser()
parser.add_argument('--track', action='store_true')
parser.add_argument('--num_embeddings', default=64, type=int)
parser.add_argument('--commitment_cost', default=0.20, type=float)
parser.add_argument('--embedding_dim', default=128, type=int)

args = parser.parse_args()

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

embedding_dim = args.embedding_dim
seq_len = 25  # Starting sequence length
num_heads = 8
num_layers = 4
num_embeddings = args.num_embeddings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 512
commitment_cost = args.commitment_cost

if args.track:
    import wandb
    print("Tracking with wandb!")
    run_name = f'traj_trans_no_quant' + datetime.now().strftime("%d-%m-%y-%H-%M-%S")
    wandb.init(project='transformer_vq', entity='mail-rishav9', name=run_name)
    wandb.config.update({
        'embedding_dim': embedding_dim,
        'num_embed': num_embeddings,
        'commitment_cost': commitment_cost,
    })

import torch.nn.functional as F

def merge_similar_vectors(quantizer, similarity_threshold=0.9):
    embeddings = quantizer.embeddings.weight.data
    num_embeddings = embeddings.size(0)
    merged_indices = set()

    for i in range(num_embeddings):
        if i in merged_indices:
            continue
        for j in range(i + 1, num_embeddings):
            if j in merged_indices:
                continue
            similarity = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
            if similarity > similarity_threshold:
                # Merge vectors i and j
                embeddings[i] = (embeddings[i] + embeddings[j]) / 2
                merged_indices.add(j)

    # Remove merged vectors
    remaining_indices = [i for i in range(num_embeddings) if i not in merged_indices]
    quantizer.embeddings = nn.Embedding(len(remaining_indices), quantizer.embedding_dim)
    quantizer.embeddings.weight.data = embeddings[remaining_indices]
    quantizer.num_embeddings = len(remaining_indices)



class GatedTransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(GatedTransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.gru1 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.gru2 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        attn_output = self.dropout(attn_output)
        attn_output = self.norm1(attn_output + src)

        # First GRU gating
        gru_output1, _ = self.gru1(attn_output)
        gru_output1 = self.dropout(gru_output1)
        gru_output1 = self.norm2(gru_output1 + attn_output)

        # Second GRU gating
        gru_output2, _ = self.gru2(gru_output1)
        gru_output2 = self.dropout(gru_output2)

        return gru_output2


# Define the VQ-VAE Quantizer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.orthogonality_weight = 0.1

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embeddings.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        
        
        if args.track:
            wandb.log({'unique_indices': len(torch.unique(encoding_indices))})
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embeddings.weight).view(inputs.shape)
        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs)
        q_latent_loss = nn.functional.mse_loss(quantized, inputs.detach())
        
        if args.track:
            wandb.log({ 
                    'q_latent_loss': q_latent_loss,
                    'e_latent_loss': e_latent_loss
                })
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # orthogonality_loss_value = self.orthogonality_loss(self.embeddings.weight)
        # loss += self.orthogonality_weight * orthogonality_loss_value
        quantized = inputs + (quantized - inputs).detach()
        self.log_codebook_distances()

        return quantized, loss, encoding_indices

    def log_codebook_distances(self):
        with torch.no_grad():
            codebook_vectors = self.embeddings.weight
            distances = torch.cdist(codebook_vectors, codebook_vectors, p=2)
            if args.track:
                wandb.log({"codebook_distances": wandb.Histogram(distances.cpu().numpy())})
    
    def orthogonality_loss(self, embeddings):
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        identity = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
        orthogonality_penalty = torch.norm(similarity_matrix - identity)
        return orthogonality_penalty


# Define the Transformer-based Encoder-Decoder Model with End-of-Sequence Input to Decoder
class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, num_layers, num_embeddings):
        super(TrajectoryTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(embedding_dim, input_dim)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        memory = self.encoder(src_emb, src_mask)
        end_of_sequence_representation = memory[:, -1, :]  # Shape: (batch_size, embedding_dim)
        quantized, vq_loss, idx = self.quantizer(end_of_sequence_representation)
        tgt_emb[:, 0, :] = quantized
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        output = self.output_layer(output)
        return output, vq_loss, idx


class GatedTrajectoryTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, num_layers, num_embeddings):
        super(GatedTrajectoryTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.encoder_layers = nn.ModuleList(
            [GatedTransformerLayer(embedding_dim, num_heads) for _ in range(num_layers)]
        )
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.decoder_layers = nn.ModuleList(
            [GatedTransformerLayer(embedding_dim, num_heads) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(embedding_dim, input_dim)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode the source sequence
        src_emb = self.embedding(src)
        memory = self.encoder(src_emb, src_mask)
        
        # Get quantized representation from the encoder's final output
        end_of_sequence_representation = memory[:, -1, :]
        quantized, vq_loss, idx = self.quantizer(end_of_sequence_representation)

        # Initialize decoder input with quantized vector as the first token
        tgt_emb = self.embedding(tgt)
        tgt_emb[:, 0, :] = quantized  # Set the first token for the decoder input

        outputs = []
        for t in range(tgt_emb.size(1)):
            if t > 0:
                tgt_emb[:, t, :] = self.embedding(outputs[-1])

            # Pass the input so far into the decoder
            tgt_mask = generate_square_subsequent_mask(t + 1).to(tgt.device)
            output_step = self.decoder(tgt_emb[:, :t + 1, :], memory, tgt_mask=tgt_mask, memory_mask=src_mask)
            outputs.append(self.output_layer(output_step[:, -1, :]))

        # Stack the outputs and reshape them to the target's shape
        outputs = torch.stack(outputs, dim=1)
        return outputs, vq_loss, idx



# Function to generate subsequent mask
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Function to update dataloader with new sequence length
def update_dataloader(seq_len, sequences):
    inputs = []
    targets = []
    for i in range(len(sequences) - seq_len):
        inputs.append(sequences[i:i+seq_len])
        targets.append(sequences[i+1:i+seq_len+1])
    inputs_tensor = torch.stack(inputs)
    targets_tensor = torch.stack(targets)
    dataset = TensorDataset(inputs_tensor, targets_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load data and prepare initial dataloader

def train(seq_len):
    env = gym.make('halfcheetah-medium-v2')
    dataset = env.get_dataset()
    observations = dataset['observations'][:100000]
    actions = dataset['actions'][:100000]

    sequences = np.hstack((observations, actions))
    sequences = torch.tensor(sequences, dtype=torch.float32)

    dataloader = update_dataloader(seq_len, sequences)
    input_dim = sequences.shape[1]

    model = TrajectoryTransformer(input_dim, embedding_dim, num_heads, num_layers, num_embeddings)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    max_seq_len = 50
    num_epochs = 50

    best_loss = float('inf')
    best_model_state = None 
    for epoch in range(num_epochs):
        if epoch == 5:
            seq_len = 25  # Increase sequence length by 10, up to max_seq_len
            dataloader = update_dataloader(seq_len, sequences)
            print(f"Updated sequence length to {seq_len}")

        model.train()
        total_loss = 0
        for batch in dataloader:
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()
            src_mask = generate_square_subsequent_mask(seq_len).to(device)
            tgt_mask = generate_square_subsequent_mask(seq_len).to(device)
            output, vq_loss, idx = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            reconstruction_loss = criterion(output, tgt)
            loss = reconstruction_loss + vq_loss
            if args.track:
                wandb.log({
                    'reconstruction_loss': reconstruction_loss,
                    'seq_len':seq_len
                })
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(torch.unique(idx))
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # if epoch > 0 and epoch%15:
        #     merge_similar_vectors(model.quantizer, 0.9)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
            print(f"New best model found at epoch {epoch+1} with loss {best_loss:.4f}")
            torch.save(best_model_state, f'best_model_embedding_{embedding_dim}_codebook_size_{num_embeddings}.pth')
    
    torch.save(model.state_dict(), f'trained_embedding_{embedding_dim}_codebook_size_{num_embeddings}.pth')


# train(seq_len)
