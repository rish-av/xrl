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
import socket

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='Enable wandb logging')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--initial_tf', type=float, default=0.8)
    parser.add_argument('--tf_decay', type=float, default=0.95)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--temp_init', type=float, default=1.0)
    parser.add_argument('--temp_min', type=float, default=0.1)
    parser.add_argument('--anneal_rate', type=float, default=0.00003)
    return parser.parse_args()

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
        self.temperature = temp_init
        self.temp_min = temp_min
        self.anneal_rate = anneal_rate
        
        # Codebook
        self.codebook = nn.Embedding(num_tokens, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_tokens, 1.0 / num_tokens)
        
        # EMA related buffers
        self.register_buffer('ema_cluster_size', torch.zeros(num_tokens))
        self.register_buffer('ema_w', torch.zeros(num_tokens, latent_dim))

    def anneal_temperature(self):
        self.temperature = max(
            self.temp_min,
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
        scaled_distances = distances / max(self.temperature, 1e-5)
        
        # Softmax with temperature
        soft_assign = F.softmax(-scaled_distances, dim=-1)
        
        # Get indices and hard assignment
        indices = soft_assign.argmax(dim=-1)
        hard_assign = F.one_hot(indices, self.num_tokens).float()
        assign = hard_assign + soft_assign - soft_assign.detach()
        
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
        
        # Quantize
        quantized = torch.matmul(assign, self.codebook.weight)
        quantized = quantized.view(batch_size, seq_len, latent_dim)
        
        # Calculate losses
        commitment_loss = self.beta * F.mse_loss(latent.detach(), quantized)
        codebook_loss = F.mse_loss(latent, quantized.detach())
        
        # Calculate Euclidean distances between codebook vectors
        codebook_distances = torch.cdist(self.codebook.weight, self.codebook.weight)
        mask = ~torch.eye(codebook_distances.shape[0], dtype=bool, device=codebook_distances.device)
        masked_distances = codebook_distances[mask]
        avg_euclidean = masked_distances.mean()
        min_euclidean = masked_distances.min()
        
        # Straight-through estimator
        quantized = latent + (quantized - latent).detach()
        
        # Calculate perplexity
        avg_probs = torch.mean(hard_assign, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return (quantized, indices, commitment_loss, codebook_loss, 
                perplexity, selected_cosine_sim, avg_euclidean, min_euclidean)

class VQVAE_TeacherForcing(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, num_tokens, hidden_size, n_layers, n_heads, context_len, beta,
                 temp_init=1.0, temp_min=0.1, anneal_rate=0.00003, ema_decay=0.99):
        super().__init__()
        
        # ... same initialization code ...
        
        # Create causal masks for encoder and decoder
        self.register_buffer(
            "encoder_mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        )
        self.register_buffer(
            "decoder_mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        )
        
        self.context_len = context_len  # store for mask creation

    def encode(self, states, actions):
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)
        combined_emb = state_emb + action_emb

        # Add positional embeddings
        position_ids = torch.arange(states.size(1), device=states.device).unsqueeze(0)
        position_emb = self.pos_embedding(position_ids)
        combined_emb += position_emb

        # Apply causal encoder mask
        encoded = self.encoder(combined_emb, mask=self.encoder_mask)
        return self.to_latent(encoded)

    def decode(self, quantized, targets, teacher_forcing_ratio=0.5):
        batch_size, seq_len = targets.shape[0], targets.shape[1]
        quantized = self.from_latent(quantized)
        
        current_input = torch.zeros_like(targets[:, 0]).unsqueeze(1)
        outputs = []
        tf_count = 0
        
        for t in range(seq_len):
            # Create current step's causal mask
            tgt_mask = self.decoder_mask[:t+1, :t+1]
            
            decoder_input = self.state_embedding(current_input)
            # Pass both masks to the decoder
            decoder_output = self.decoder(
                decoder_input, 
                quantized,
                tgt_mask=tgt_mask,            # Causal mask for self-attention
                memory_mask=self.encoder_mask  # Causal mask for cross-attention
            )
            pred = self.output_state(decoder_output[:, -1:])
            outputs.append(pred)
            
            if t < seq_len - 1:
                if torch.rand(1).item() < teacher_forcing_ratio:
                    next_input = targets[:, t:t+1]
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
        
        print(f"loss: {reconstruction_loss.item():.4f}, commitment: {commitment_loss.item():.4f}, "
              f"codebook: {codebook_loss.item():.4f}, perplexity: {perplexity.item():.4f}, "
              f"temperature: {self.vector_quantizer.temperature:.4f}, "
              f"cosine_similarity: {cosine_sim.item():.4f}, "
              f"avg_euclidean: {avg_euclidean.item():.4f}, "
              f"min_euclidean: {min_euclidean.item():.4f}")
            
        return (predicted_states, reconstruction_loss, commitment_loss, codebook_loss, 
                perplexity, cosine_sim, avg_euclidean, min_euclidean)


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
                "tf_decay": args.tf_decay,
                "ema_decay": args.ema_decay
            },
            entity="mail-rishav9"
        )
        wandb.watch(model)

    teacher_forcing_ratio = args.initial_tf

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

            # Updated unpacking
            (_, reconstruction_loss, commitment_loss, 
             codebook_loss, perplexity, cosine_sim, 
             avg_euclidean, min_euclidean) = model(
                states, actions, targets, teacher_forcing_ratio)

            loss = reconstruction_loss + commitment_loss + codebook_loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_reconstruction += reconstruction_loss.item()
            total_commitment += commitment_loss.item()
            total_codebook += codebook_loss.item()
            total_perplexity += perplexity.item()
            total_cosine_sim += cosine_sim.item()
            total_euclidean += avg_euclidean.item()
            total_min_euclidean += min_euclidean.item()

            if batch_idx % 1500 == 0:
                torch.save(
                    model.state_dict(), 
                    f"weights/vqvae_mujoco_teacher_forcing_no_skip_{epoch}_{batch_idx}.pt"
                )

            if args.log and batch_idx % 10 == 0:
                metrics = {
                    "batch/total_loss": loss.item(),
                    "batch/reconstruction_loss": reconstruction_loss.item(),
                    "batch/commitment_loss": commitment_loss.item(),
                    "batch/codebook_loss": codebook_loss.item(),
                    "batch/perplexity": perplexity.item(),
                    "batch/teacher_forcing_ratio": teacher_forcing_ratio,
                    "batch/active_codes": (model.vector_quantizer.ema_cluster_size > 0).sum().item(),
                    "batch/temperature": model.vector_quantizer.temperature,
                    "batch/cosine_similarity": cosine_sim.item(),
                    "batch/avg_euclidean": avg_euclidean.item(),
                    "batch/min_euclidean": min_euclidean.item(),
                }
                wandb.log(metrics)
                
                # Log histograms and additional stats every 20 batches
                if batch_idx % 20 == 0:
                    codebook = model.vector_quantizer.codebook.weight.detach()
                    distances = torch.cdist(codebook, codebook)
                    mask = ~torch.eye(len(codebook), dtype=bool, device=codebook.device)
                    distances = distances[mask].cpu().numpy()
                    
                    # Log histograms and stats
                    wandb.log({
                        "batch/euclidean_distances_hist": wandb.Histogram(distances),
                        "batch/euclidean_distances_mean": distances.mean(),
                        "batch/euclidean_distances_std": distances.std(),
                        "batch/codebook_usage_hist": wandb.Histogram(
                            model.vector_quantizer.ema_cluster_size.cpu().numpy()
                        )
                    })

        # Decay teacher forcing ratio
        teacher_forcing_ratio *= args.tf_decay

        # Calculate epoch averages
        avg_metrics = {
            "epoch/loss": total_loss / len(dataloader),
            "epoch/reconstruction": total_reconstruction / len(dataloader),
            "epoch/commitment": total_commitment / len(dataloader),
            "epoch/codebook": total_codebook / len(dataloader),
            "epoch/perplexity": total_perplexity / len(dataloader),
            "epoch/cosine_similarity": total_cosine_sim / len(dataloader),
            "epoch/avg_euclidean": total_euclidean / len(dataloader),
            "epoch/min_euclidean": total_min_euclidean / len(dataloader),
            "epoch/teacher_forcing_ratio": teacher_forcing_ratio,
            "epoch/active_codes": (model.vector_quantizer.ema_cluster_size > 0).sum().item()
        }

        if args.log:
            wandb.log(avg_metrics)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Total Loss: {avg_metrics['epoch/loss']:.4f}")
        print(f"Reconstruction: {avg_metrics['epoch/reconstruction']:.4f}")
        print(f"Commitment: {avg_metrics['epoch/commitment']:.4f}")
        print(f"Codebook: {avg_metrics['epoch/codebook']:.4f}")
        print(f"Perplexity: {avg_metrics['epoch/perplexity']:.4f}")
        print(f"Cosine Similarity: {avg_metrics['epoch/cosine_similarity']:.4f}")
        print(f"Avg Euclidean: {avg_metrics['epoch/avg_euclidean']:.4f}")
        print(f"Min Euclidean: {avg_metrics['epoch/min_euclidean']:.4f}")
        print(f"Active Codes: {avg_metrics['epoch/active_codes']}")
        print(f"Teacher Forcing Ratio: {teacher_forcing_ratio:.4f}")

    if args.log:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    
    env_name = "halfcheetah-medium-v2"
    states, actions = load_d4rl_data(env_name)
    dataset = D4RLStateActionDataset(states, actions, context_len=20)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    model = VQVAE_TeacherForcing(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=64,
        num_tokens=512,
        hidden_size=128,
        n_layers=6,
        n_heads=8,
        context_len=20,
        beta=0.25,
        temp_init=args.temp_init,
        temp_min=args.temp_min,
        anneal_rate=args.anneal_rate,
        ema_decay=args.ema_decay
    ).to("cuda")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_vqvae_teacher_forcing(model, dataloader, optimizer, args)
    # from utils import render_halfcheetah
    # model.load_state_dict(torch.load('/home/rishav/scratch/xrl/vqvae_epoch_9.pth'))
    

