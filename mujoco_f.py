import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import argparse
from datetime import datetime
torch.autograd.set_detect_anomaly(True)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='Enable wandb logging')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--initial_tf', type=float, default=0.5)
    parser.add_argument('--tf_decay', type=float, default=0.85)
    parser.add_argument('--tag', type=str, default="vqvae")
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_tokens', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--context_len', type=int, default=50)
    parser.add_argument('--dataset_type', type=str, default='overlap', help='Overlap sequences')
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


class D4RLStateActionDatasetNo(Dataset):
    def __init__(self, states, actions, context_len=20):
        self.context_len = context_len
        self.data = []

        # Step by context_len instead of 1 to avoid overlap
        for i in range(0, len(states) - context_len - 1, context_len):
            seq_states = states[i:i+context_len]
            seq_actions = actions[i:i+context_len]
            target_states = states[i+1:i+1+context_len]
            self.data.append({
                "states": seq_states,
                "actions": seq_actions,
                "targets": target_states
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.data[idx]["states"], dtype=torch.float32),
            "actions": torch.tensor(self.data[idx]["actions"], dtype=torch.float32),
            "targets": torch.tensor(self.data[idx]["targets"], dtype=torch.float32),
        }



class D4RLStateActionDataset(Dataset):
    def __init__(self, states, actions, context_len=20):
        self.context_len = context_len
        self.data = []

        for i in range(len(states) - context_len - 1):
            seq_states = states[i:i+context_len]
            seq_actions = actions[i:i+context_len]
            target_states = states[i+1:i+1+context_len]
            self.data.append({
                "states": seq_states,
                "actions": seq_actions,
                "targets": target_states
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.data[idx]["states"], dtype=torch.float32),
            "actions": torch.tensor(self.data[idx]["actions"], dtype=torch.float32),
            "targets": torch.tensor(self.data[idx]["targets"], dtype=torch.float32),
        }

def load_d4rl_data(env_name):
    env = gym.make(env_name)
    dataset = env.get_dataset()
    states = dataset['observations']
    actions = dataset['actions']
    return states, actions



class VectorQuantizer(nn.Module):
    def __init__(self, num_tokens, latent_dim, beta, decay=0.99, temp_init=1.0, temp_min=0.1, anneal_rate=0.00003):
        super().__init__()
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        self.beta = beta
        self.decay = decay
        
        # Temperature annealing parameters
        self.register_buffer('temperature', torch.tensor(temp_init))
        self.temp_min = temp_min
        self.anneal_rate = anneal_rate
        
        # Codebook
        self.codebook = nn.Embedding(num_tokens, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_tokens, 1.0 / num_tokens)
        
        # EMA related buffers
        self.register_buffer('ema_cluster_size', torch.zeros(num_tokens))
        self.register_buffer('ema_w', torch.zeros(num_tokens, latent_dim))
        self.register_buffer('usage_count', torch.zeros(num_tokens))

    def anneal_temperature(self):
        self.temperature = torch.max(
            torch.tensor(self.temp_min, device=self.temperature.device),
            self.temperature * np.exp(-self.anneal_rate)
        )

    def forward(self, latent):
        batch_size, seq_len, latent_dim = latent.shape
        flat_input = latent.reshape(-1, latent_dim)
        
        # Calculate cosine similarity
        latent_normalized = F.normalize(flat_input, dim=-1)
        codebook_normalized = F.normalize(self.codebook.weight, dim=-1)
        cosine_sim = torch.matmul(latent_normalized, codebook_normalized.t())
        
        # Calculate distances for quantization
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True) 
            - 2 * torch.matmul(flat_input, self.codebook.weight.t())
            + torch.sum(self.codebook.weight ** 2, dim=1)
        )
        
        # Apply temperature scaling to distances
        scaled_distances = distances / max(self.temperature.item(), 1e-5)
        
        # Softmax with temperature
        soft_assign = F.softmax(-scaled_distances, dim=-1)
        
        # Get indices and hard assignment
        indices = soft_assign.argmax(dim=-1)
        hard_assign = F.one_hot(indices, self.num_tokens).float()
        assign = hard_assign + soft_assign - soft_assign.detach()
        
        # Update usage count
        self.usage_count[indices] += 1
        
        # Calculate average cosine similarity with chosen codes
        selected_cosine_sim = cosine_sim[torch.arange(len(indices)), indices].mean()
        
        # EMA update
        if self.training:
            cluster_size = hard_assign.sum(0)
            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * cluster_size
            
            embed_sum = torch.matmul(hard_assign.t(), flat_input)
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * embed_sum
            
            # Normalize
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.num_tokens * 1e-5) * n
            embed_normalized = self.ema_w / cluster_size.unsqueeze(-1)
            self.codebook.weight.data.copy_(embed_normalized)
            
            # Anneal temperature
            self.anneal_temperature()
        
        # Calculate Euclidean distances between codebook vectors
        codebook_distances = torch.cdist(self.codebook.weight, self.codebook.weight)
        mask = ~torch.eye(codebook_distances.shape[0], dtype=bool, device=codebook_distances.device)
        masked_distances = codebook_distances[mask]
        avg_euclidean = masked_distances.mean()
        min_euclidean = masked_distances.min()
        
        # Quantize
        quantized = torch.matmul(assign, self.codebook.weight)
        quantized = quantized.view(batch_size, seq_len, latent_dim)
        
        # Loss
        commitment_loss = self.beta * F.mse_loss(latent.detach(), quantized)
        codebook_loss = F.mse_loss(latent, quantized.detach())
        
        # Straight-through estimator
        quantized = latent + (quantized - latent).detach()
        
        # Calculate perplexity
        avg_probs = torch.mean(hard_assign, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return (quantized, indices, commitment_loss, codebook_loss, 
                perplexity, selected_cosine_sim, avg_euclidean, min_euclidean)

# Function to train the VQVAE_TeacherForcing model
def train_vqvae_with_new_vq(model, dataloader, optimizer_main, optimizer_vq, args):
    scheduler_main = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_main, mode='min', factor=0.5, patience=2, verbose=True
    )

    teacher_forcing_ratio = args.initial_tf
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_reconstruction = 0
        total_commitment = 0
        total_codebook = 0
        total_perplexity = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer_main.zero_grad()
            optimizer_vq.zero_grad()

            states = batch['states'].to(args.device)
            actions = batch['actions'].to(args.device)
            targets = batch['targets'].to(args.device)

            outputs = model(states, actions, targets, teacher_forcing_ratio)
            (predicted_states, loss, reconstruction_loss, commitment_loss, 
             codebook_loss, perplexity, cosine_sim, 
             avg_euclidean, min_euclidean) = outputs

            # Combine losses and do a single backward pass
            total_loss = reconstruction_loss + commitment_loss + codebook_loss
            total_loss.backward()
            
            # Step both optimizers
            optimizer_main.step()
            optimizer_vq.step()

            total_loss += loss.item()
            total_reconstruction += reconstruction_loss.item()
            total_commitment += commitment_loss.item()
            total_codebook += codebook_loss.item()
            total_perplexity += perplexity.item()

            if batch_idx % 1500 == 0:
                teacher_forcing_ratio *= args.tf_decay

            if batch_idx % 1500 == 0:
                torch.save(model.state_dict(), f"vqvae_model_epoch_{epoch}_batch_{batch_idx}.pt")

        avg_epoch_loss = total_loss / len(dataloader)
        scheduler_main.step(avg_epoch_loss)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), f"vqvae_best_model_epoch_{epoch}.pt")

        print(
            f"Epoch {epoch + 1}, "
            f"Total Loss: {avg_epoch_loss:.4f}, "
            f"Reconstruction: {total_reconstruction / len(dataloader):.4f}, "
            f"Commitment: {total_commitment / len(dataloader):.4f}, "
            f"Codebook: {total_codebook / len(dataloader):.4f}, "
            f"Perplexity: {total_perplexity / len(dataloader):.4f}"
        )

        teacher_forcing_ratio *= args.tf_decay



class VQVAE_TeacherForcing(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, num_tokens, hidden_size, n_layers, n_heads, context_len, beta,
                 temp_init=1.0, temp_min=0.1, anneal_rate=0.00003, ema_decay=0.99):
        super().__init__()

        # Embedding and normalization layers
        self.state_embedding = nn.Sequential(
            nn.Identity(),
            nn.Linear(state_dim, hidden_size)
        )
        self.action_embedding = nn.Sequential(
            nn.Identity(),
            nn.Linear(action_dim, hidden_size)
        )
        self.pos_embedding = nn.Embedding(context_len, hidden_size)

        # Causal masks
        self.register_buffer(
            "encoder_mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        )
        self.register_buffer(
            "decoder_mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        )

        # Transformer encoder with norm_first=False
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size * 4, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Latent projection
        self.to_latent = nn.Sequential(
            nn.Identity(),
            nn.Linear(hidden_size, latent_dim)
        )

        # Updated Vector Quantizer Integration
        self.vector_quantizer = VectorQuantizer(
            num_tokens, latent_dim, beta, decay=ema_decay
        )

        # Decoder layers
        self.from_latent = nn.Sequential(
            nn.Identity(),
            nn.Linear(latent_dim, hidden_size)
        )

        # Transformer decoder with norm_first=True
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size * 4, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_state = nn.Linear(hidden_size, state_dim)

        self.model_config = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "latent_dim": latent_dim,
            "num_tokens": num_tokens,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "context_len": context_len,
            "beta": beta
        }

    def encode(self, states, actions):
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)
        combined_emb = state_emb + action_emb

        # Add positional embeddings
        position_ids = torch.arange(states.size(1), device=states.device).unsqueeze(0)
        position_emb = self.pos_embedding(position_ids)
        combined_emb += position_emb

        # Encoder with causal mask
        encoded = self.encoder(combined_emb, mask=self.encoder_mask)

        # Clone to avoid in-place modification
        latent = self.to_latent(encoded)

        return latent

    def decode(self, quantized, targets, teacher_forcing_ratio=0.5):
        batch_size, seq_len = targets.shape[0], targets.shape[1]

        # Transform quantized vectors to decoder space
        decoder_memory = self.from_latent(quantized)

        current_input = torch.zeros_like(targets[:, 0]).unsqueeze(1)
        outputs = []
        tf_count = 0

        for t in range(seq_len):
            # Create causal mask for current timestep
            tgt_mask = torch.ones((t + 1, t + 1), device=current_input.device).triu_(1).bool()

            decoder_input = self.state_embedding(current_input)
            decoder_output = self.decoder(
                decoder_input,
                decoder_memory,
                tgt_mask=tgt_mask  # For self-attention
            )
            pred = self.output_state(decoder_output[:, -1:])
            outputs.append(pred)

            if t < seq_len - 1:
                if torch.rand(1).item() < teacher_forcing_ratio:
                    next_input = targets[:, t:t + 1]
                    tf_count += 1
                else:
                    next_input = pred.detach()
                current_input = torch.cat([current_input, next_input], dim=1)

        outputs = torch.cat(outputs, dim=1)
        print(f"Teacher forcing ratio: {tf_count / seq_len}")
        return outputs

    def forward(self, states, actions, targets, teacher_forcing_ratio=0.5):
        latent = self.encode(states, actions)
        quantized, indices, commitment_loss, codebook_loss, perplexity, cosine_sim, avg_euclidean, min_euclidean = self.vector_quantizer(latent)
        predicted_states = self.decode(quantized, targets, teacher_forcing_ratio)
        reconstruction_loss = F.mse_loss(predicted_states, targets)

        # Calculate total loss
        total_loss = reconstruction_loss + commitment_loss + codebook_loss

        print(f"loss: {reconstruction_loss.item():.4f}, commitment: {commitment_loss.item():.4f}, "
              f"codebook: {codebook_loss.item():.4f}, perplexity: {perplexity.item():.4f}, "
              f"cosine_similarity: {cosine_sim.item():.4f}, "
              f"avg_euclidean: {avg_euclidean.item():.4f}, "
              f"min_euclidean: {min_euclidean.item():.4f}")

        return (predicted_states, total_loss, reconstruction_loss, commitment_loss,
                codebook_loss, perplexity, cosine_sim, avg_euclidean, min_euclidean)


import warnings
np.warnings = warnings

if __name__ == "__main__":
    args = parse_args()
    context_len = args.context_len
    # Load data
    env_name = "halfcheetah-medium-v2"
    states, actions = load_d4rl_data(env_name)
    if args.dataset_type=='overlap':
        dataset = D4RLStateActionDataset(states, actions, context_len=context_len)
    else:
        print("####### creating dataset with no overlap ########")
        dataset = D4RLStateActionDatasetNo(states, actions, context_len=context_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model parameters
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    latent_dim = args.latent_dim
    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    n_layers = args.n_layers
    n_heads = args.n_heads
    beta = args.beta    

    # Create model
    model = VQVAE_TeacherForcing(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        n_layers=n_layers,
        n_heads=n_heads,
        context_len=context_len,
        beta=beta
    ).to("cuda")

    # model.load_state_dict(torch.load("/home/ubuntu/xrl/weights/vqvae_best_model.pt", weights_only=True))

    optimizer_main = torch.optim.Adam(
    [param for name, param in model.named_parameters() if "vector_quantizer" not in name],
    lr=1e-4
    )
    optimizer_vq = torch.optim.Adam(
        model.vector_quantizer.parameters(),
        lr=1e-5
    )
    train_vqvae_with_new_vq(model, dataloader, optimizer_main, optimizer_vq, args)