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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='Enable wandb logging')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--initial_tf', type=float, default=0.25)
    parser.add_argument('--tf_decay', type=float, default=0.95)
    return parser.parse_args()

class D4RLStateActionDatasetNo(Dataset):
    def __init__(self, states, actions, context_len=20):
        self.context_len = context_len
        self.data = []
        
        # Calculate state and action statistics for normalization
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0) + 1e-6
        self.action_mean = np.mean(actions, axis=0)
        self.action_std = np.std(actions, axis=0) + 1e-6
        
        # Normalize states and actions
        states = (states - self.state_mean) / self.state_std
        actions = (actions - self.action_mean) / self.action_std

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

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1)]

class VectorQuantizer(nn.Module):
    def __init__(self, num_tokens, latent_dim, beta, decay=0.99, 
                 temp_init=1.0, temp_min=0.05, anneal_rate=0.00005):  # Modified temperature parameters
        super().__init__()
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        self.beta = beta
        self.decay = decay
        
        # Modified temperature parameters
        self.register_buffer('temperature', torch.tensor(temp_init))
        self.temp_min = temp_min
        self.anneal_rate = anneal_rate
        
        # Initialize codebook with improved initialization
        self.codebook = nn.Embedding(num_tokens, latent_dim)
        self.init_codebook()
        
        # EMA related buffers
        self.register_buffer('ema_cluster_size', torch.zeros(num_tokens))
        self.register_buffer('ema_w', torch.zeros(num_tokens, latent_dim))
        self.register_buffer('usage_count', torch.zeros(num_tokens))
        
    def init_codebook(self):
        # Initialize codebook using Xavier uniform initialization
        nn.init.xavier_uniform_(self.codebook.weight)
        # Normalize codebook vectors
        with torch.no_grad():
            self.codebook.weight.data = F.normalize(self.codebook.weight.data, dim=-1)

    def anneal_temperature(self):
        # Cosine annealing for temperature
        self.temperature = torch.max(
            torch.tensor(self.temp_min, device=self.temperature.device),
            self.temperature * (np.cos(self.anneal_rate * np.pi) + 1) / 2
        )

    def forward(self, latent):
        batch_size, seq_len, latent_dim = latent.shape
        flat_input = latent.reshape(-1, latent_dim)
        
        # Normalize input and codebook for improved cosine similarity
        latent_normalized = F.normalize(flat_input, dim=-1)
        codebook_normalized = F.normalize(self.codebook.weight, dim=-1)
        
        # Calculate cosine similarity and scaled distances
        cosine_sim = torch.matmul(latent_normalized, codebook_normalized.t())
        distances = 2 - 2 * cosine_sim  # Equivalent to L2 distance between normalized vectors
        
        # Apply temperature scaling with improved numerical stability
        scaled_distances = distances / torch.clamp(self.temperature, min=1e-6)
        
        # Improved softmax calculation
        soft_assign = F.softmax(-scaled_distances, dim=-1)
        
        # Get hard assignments
        indices = soft_assign.argmax(dim=-1)
        hard_assign = F.one_hot(indices, self.num_tokens).float()
        
        # Straight-through estimator with improved gradient flow
        assign = hard_assign + soft_assign - soft_assign.detach()
        
        # Update usage statistics
        self.usage_count[indices] += 1
        
        # Calculate metrics
        selected_cosine_sim = cosine_sim[torch.arange(len(indices)), indices].mean()
        
        # EMA updates with improved stability
        if self.training:
            cluster_size = hard_assign.sum(0)
            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * cluster_size
            
            embed_sum = torch.matmul(hard_assign.t(), flat_input)
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * embed_sum
            
            # Normalize with improved numerical stability
            n = torch.clamp(self.ema_cluster_size.sum(), min=1e-6)
            cluster_size = ((self.ema_cluster_size + 1e-6) / (n + self.num_tokens * 1e-6)) * n
            embed_normalized = self.ema_w / cluster_size.unsqueeze(-1).clamp(min=1e-6)
            
            # Update codebook with EMA
            self.codebook.weight.data.copy_(embed_normalized)
            
            # Anneal temperature
            self.anneal_temperature()
        
        # Calculate codebook statistics
        codebook_distances = torch.cdist(self.codebook.weight, self.codebook.weight)
        mask = ~torch.eye(codebook_distances.shape[0], dtype=bool, device=codebook_distances.device)
        masked_distances = codebook_distances[mask]
        avg_euclidean = masked_distances.mean()
        min_euclidean = masked_distances.min()
        
        # Quantize
        quantized = torch.matmul(assign, self.codebook.weight)
        quantized = quantized.view(batch_size, seq_len, latent_dim)
        
        # Calculate losses with improved stability
        commitment_loss = self.beta * F.mse_loss(latent.detach(), quantized)
        codebook_loss = F.mse_loss(latent, quantized.detach())
        
        # Straight-through estimator
        quantized = latent + (quantized - latent).detach()
        
        # Calculate perplexity with improved numerical stability
        avg_probs = torch.mean(hard_assign, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return (quantized, indices, commitment_loss, codebook_loss, 
                perplexity, selected_cosine_sim, avg_euclidean, min_euclidean)

class VQVAE_TeacherForcing(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, num_tokens, hidden_size, n_layers, n_heads, context_len, beta):
        super().__init__()
        
        # Increased number of attention heads
        self.n_heads = n_heads * 2  # Doubled number of heads
        
        # Embedding layers with improved normalization
        self.state_embedding = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size)  # Additional normalization
        )
        
        self.action_embedding = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, hidden_size),
            nn.LayerNorm(hidden_size)  # Additional normalization
        )
        
        # Replace traditional positional embedding with relative positional encoding
        self.pos_encoding = RelativePositionalEncoding(hidden_size)
        
        # Improved masking
        self.register_buffer(
            "encoder_mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        )
        self.register_buffer(
            "decoder_mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        )

        # Enhanced transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=self.n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,  # Added dropout
            batch_first=True,
            norm_first=True
        )
        encoder_norm = nn.LayerNorm(hidden_size)  # Added explicit normalization
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=encoder_norm)

        # Enhanced latent projections
        self.to_latent = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),  # Added activation
            nn.Linear(hidden_size, latent_dim),
            nn.LayerNorm(latent_dim)  # Final normalization
        )

        # Improved Vector Quantizer
        self.vector_quantizer = VectorQuantizer(
            num_tokens, latent_dim, beta,
            temp_init=1.0,
            temp_min=0.05,  # Lower minimum temperature
            anneal_rate=0.00005  # Slower annealing
        )

        # Enhanced decoder input processing
        self.from_latent = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_size),
            nn.GELU(),  # Added activation
            nn.LayerNorm(hidden_size)
        )

        # Enhanced transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=self.n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,  # Added dropout
            batch_first=True,
            norm_first=True
        )
        decoder_norm = nn.LayerNorm(hidden_size)  # Added explicit normalization
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers, norm=decoder_norm)

        # Enhanced output projection
        self.output_state = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),  # Added activation
            nn.Linear(hidden_size, state_dim)
        )

        self.model_config = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "latent_dim": latent_dim,
            "num_tokens": num_tokens,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "n_heads": self.n_heads,
            "context_len": context_len,
            "beta": beta
        }

    def encode(self, states, actions):
        # Enhanced embedding process
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)
        combined_emb = state_emb + action_emb
        
        # Add relative positional encoding
        pos_enc = self.pos_encoding(combined_emb)
        combined_emb = combined_emb + pos_enc
        
        # Additional normalization
        combined_emb = F.dropout(combined_emb, p=0.1, training=self.training)
        
        # Enhanced encoding process
        encoded = self.encoder(combined_emb, mask=self.encoder_mask)
        return self.to_latent(encoded)

    def decode(self, quantized, targets, teacher_forcing_ratio=0.5):
        batch_size, seq_len = targets.shape[0], targets.shape[1]
        
        # Enhanced decoder memory processing
        decoder_memory = self.from_latent(quantized)
        
        current_input = torch.zeros_like(targets[:, 0]).unsqueeze(1)
        outputs = []  # Changed from tensor to list
        tf_count = 0
        
        for t in range(seq_len):
            tgt_mask = torch.ones((t+1, t+1), device=current_input.device).triu_(1).bool()
            
            # Enhanced decoder input processing
            decoder_input = self.state_embedding(current_input)
            decoder_input = decoder_input + self.pos_encoding(decoder_input)
            
            decoder_output = self.decoder(
                decoder_input,
                decoder_memory,
                tgt_mask=tgt_mask,
                memory_mask=None
            )
            
            pred = self.output_state(decoder_output[:, -1:])
            outputs.append(pred)  # Now appending to a list
            
            if t < seq_len - 1:
                if torch.rand(1).item() < teacher_forcing_ratio:
                    next_input = targets[:, t:t+1]
                    tf_count += 1
                else:
                    next_input = pred.detach()
                current_input = torch.cat([current_input, next_input], dim=1)
        
        outputs = torch.cat(outputs, dim=1)  # Concatenate the list of tensors at the end
        return outputs

    def forward(self, states, actions, targets, teacher_forcing_ratio=0.5):
        latent = self.encode(states, actions)
        quantized, indices, commitment_loss, codebook_loss, perplexity, cosine_sim, avg_euclidean, min_euclidean = self.vector_quantizer(latent)
        predicted_states = self.decode(quantized, targets, teacher_forcing_ratio)
        
        # Enhanced loss calculation with gradient scaling
        reconstruction_loss = F.mse_loss(predicted_states, targets)
        total_loss = reconstruction_loss + commitment_loss + codebook_loss
        
        print(f"loss: {reconstruction_loss.item():.4f}, commitment: {commitment_loss.item():.4f}, "
              f"codebook: {codebook_loss.item():.4f}, perplexity: {perplexity.item():.4f}, "
              f"temperature: {self.vector_quantizer.temperature:.4f}, "
              f"cosine_similarity: {cosine_sim.item():.4f}, "
              f"avg_euclidean: {avg_euclidean.item():.4f}, "
              f"min_euclidean: {min_euclidean.item():.4f}")
        
        return (predicted_states, total_loss, reconstruction_loss, commitment_loss, 
                codebook_loss, perplexity, cosine_sim, avg_euclidean, min_euclidean)

# Import the scheduler
from torch.optim.lr_scheduler import LambdaLR

# In the train function:
def train_vqvae_teacher_forcing(model, dataloader, optimizer, args):
    if args.log:
        run_name = f"vqvae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="vqvae-training",
            name=run_name,
            config={
                **model.model_config,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "initial_tf": args.initial_tf,
                "tf_decay": args.tf_decay
            }
        )
        wandb.watch(model)

    # Enhanced learning rate scheduler with warmup
    def warmup_cosine_schedule(step):
        warmup_steps = 1000
        max_steps = 10000
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.5 * (1. + np.cos(np.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    
    teacher_forcing_ratio = args.initial_tf
    best_loss = float('inf')
    patience = 0
    max_patience = 3

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_reconstruction = 0
        total_commitment = 0
        total_codebook = 0
        total_perplexity = 0
        total_cosine_sim = 0
        total_euclidean = 0
        total_min_euclidean = 0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            states = batch['states'].to("cuda")
            actions = batch['actions'].to("cuda")
            targets = batch['targets'].to("cuda")
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            (_, loss, reconstruction_loss, commitment_loss, 
             codebook_loss, perplexity, cosine_sim, 
             avg_euclidean, min_euclidean) = model(
                states, actions, targets, teacher_forcing_ratio)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            total_reconstruction += reconstruction_loss.item()
            total_commitment += commitment_loss.item()
            total_codebook += codebook_loss.item()
            total_perplexity += perplexity.item()
            total_cosine_sim += cosine_sim.item()
            total_euclidean += avg_euclidean.item()
            total_min_euclidean += min_euclidean.item()
            
            if batch_idx % 2500 == 0:
                torch.save(model.state_dict(), 
                         f"weights/improved_vqvae_epoch{epoch}_batch{batch_idx}.pt")
            
            if args.log:
                wandb.log({
                    "batch/total_loss": loss.item(),
                    "batch/reconstruction_loss": reconstruction_loss.item(),
                    "batch/commitment_loss": commitment_loss.item(),
                    "batch/codebook_loss": codebook_loss.item(),
                    "batch/perplexity": perplexity.item(),
                    "batch/teacher_forcing_ratio": teacher_forcing_ratio,
                    "batch/cosine_similarity": cosine_sim.item(),
                    "batch/avg_euclidean": avg_euclidean.item(),
                    "batch/min_euclidean": min_euclidean.item(),
                    "batch/codebook_usage": (model.vector_quantizer.usage_count > 0).sum().item() / model.vector_quantizer.num_tokens,
                    "batch/learning_rate": optimizer.param_groups[0]['lr']
                })
        
        # Calculate average epoch loss
        avg_epoch_loss = total_loss / len(dataloader)
        
        # Early stopping with patience
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience = 0
            torch.save(model.state_dict(), "weights/improved_vqvae_best.pt")
        else:
            patience += 1
            
        if patience >= max_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
            
        print(
            f"Epoch {epoch + 1}, "
            f"Total Loss: {avg_epoch_loss:.4f}, "
            f"Reconstruction: {total_reconstruction/len(dataloader):.4f}, "
            f"Commitment: {total_commitment/len(dataloader):.4f}, "
            f"Codebook: {total_codebook/len(dataloader):.4f}, "
            f"Perplexity: {total_perplexity/len(dataloader):.4f}, "
            f"Cosine Sim: {total_cosine_sim/len(dataloader):.4f}, "
            f"Avg Euclidean: {total_euclidean/len(dataloader):.4f}, "
            f"Min Euclidean: {total_min_euclidean/len(dataloader):.4f}, "
            f"Teacher Forcing: {teacher_forcing_ratio:.4f}, "
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        if args.log:
            wandb.log({
                "epoch/total_loss": avg_epoch_loss,
                "epoch/reconstruction": total_reconstruction/len(dataloader),
                "epoch/commitment": total_commitment/len(dataloader),
                "epoch/codebook": total_codebook/len(dataloader),
                "epoch/perplexity": total_perplexity/len(dataloader),
                "epoch/cosine_similarity": total_cosine_sim/len(dataloader),
                "epoch/avg_euclidean": total_euclidean/len(dataloader),
                "epoch/min_euclidean": total_min_euclidean/len(dataloader),
                "epoch/teacher_forcing_ratio": teacher_forcing_ratio
            })
        
        teacher_forcing_ratio *= args.tf_decay

    if args.log:
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    
    # Load data
    env_name = "halfcheetah-medium-v2"
    states, actions = load_d4rl_data(env_name)
    dataset = D4RLStateActionDataset(states, actions, context_len=20)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Enhanced model parameters
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    latent_dim = 128  # Increased latent dimension
    num_tokens = 128  # Increased number of tokens
    hidden_size = 256  # Increased hidden size
    n_layers = 8      # Increased number of layers
    n_heads = 16      # Increased number of heads
    beta = 0.25
    
    # Create model
    model = VQVAE_TeacherForcing(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        n_layers=n_layers,
        n_heads=n_heads,
        context_len=20,
        beta=beta
    ).to("cuda")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # model.load_state_dict(torch.load('/home/rishav/scratch/xrl/weights/vqvae_mujoco_teacher_forcing_no_skip_connect_run_2_tf0.45_schedule_llr0_2500.pt', weights_only=False), strict=False)
    
    # Load previous weights if continuing training
    # model.load_state_dict(torch.load('path_to_weights.pt'))
    
    train_vqvae_teacher_forcing(model, dataloader, optimizer, args)