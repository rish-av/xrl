import numpy as np
import torch
from torch.utils.data import Dataset

import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import DataLoader
import torch.optim as optim

# import gym
# import d4rl_atari


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EnhancedVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, 
                 commitment_cost=0.25,
                 diversity_weight=2.0,  # Increased significantly
                 entropy_weight=2.0,    # Increased significantly
                 temperature=0.05,      # Lowered for harder assignments
                 min_usage_threshold=0.01):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.diversity_weight = diversity_weight
        self.entropy_weight = entropy_weight
        self.temperature = temperature
        self.min_usage_threshold = min_usage_threshold
        
        # Initialize embeddings with orthogonal vectors
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.orthogonal_(self.embed.weight.data)
        
        # Tracking buffers
        self.register_buffer('code_usage', torch.zeros(num_embeddings))
        
    def reinit_unused_codes(self):
        # Get usage statistics
        total_usage = self.code_usage.sum()
        usage_prob = self.code_usage / total_usage
        
        # Find unused codes
        unused_mask = usage_prob < self.min_usage_threshold
        
        if unused_mask.any():
            # Get statistics from used codes
            used_codes = ~unused_mask
            mean_embedding = self.embed.weight[used_codes].mean(0)
            std_embedding = self.embed.weight[used_codes].std(0)
            
            # Reinitialize unused codes with perturbation around used codes
            num_unused = unused_mask.sum()
            noise = torch.randn(num_unused, self.embedding_dim, device=self.embed.weight.device)
            new_embeddings = mean_embedding + noise * std_embedding
            
            # Normalize new embeddings
            new_embeddings = F.normalize(new_embeddings, dim=1)
            
            # Update unused codes
            self.embed.weight.data[unused_mask] = new_embeddings
            self.code_usage[unused_mask] = 0
    
    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # Normalize inputs and embeddings
        flat_input_norm = F.normalize(flat_input, dim=1)
        codebook_norm = F.normalize(self.embed.weight, dim=1)
        
        # Compute distances
        distances = torch.cdist(flat_input, self.embed.weight, p=2)
        
        # Add usage penalty to distances to encourage using rare codes
        if self.training:
            usage_probs = (self.code_usage + 1e-5) / (self.code_usage + 1e-5).sum()
            usage_penalty = 2.0 * torch.log(usage_probs + 1e-5)  # Higher penalty for frequently used codes
            distances = distances + usage_penalty.unsqueeze(0)
        
        # Get hard assignments
        encoding_indices = torch.argmin(distances, dim=1)
        hard_assignments = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Get quantized embeddings
        quantized = self.embed(encoding_indices)
        
        # Compute losses
        commitment_loss = F.mse_loss(quantized.detach(), flat_input)
        
        # Stronger diversity loss
        similarity_matrix = torch.matmul(codebook_norm, codebook_norm.t())
        off_diag = similarity_matrix - torch.eye(self.num_embeddings, device=similarity_matrix.device)
        diversity_loss = torch.pow(torch.clamp(torch.abs(off_diag), min=0.1), 2).mean()
        
        # Stronger entropy loss
        avg_probs = hard_assignments.mean(0)
        uniform_probs = torch.ones_like(avg_probs) / self.num_embeddings
        entropy_loss = F.kl_div(
            (avg_probs + 1e-10).log(),
            uniform_probs,
            reduction='sum',
            log_target=False
        )
        
        # Update usage statistics
        with torch.no_grad():
            if self.training:
                self.code_usage.mul_(0.99).add_(hard_assignments.sum(0), alpha=0.01)
                
                # Force redistribution if too few codes are active
                usage_probs = self.code_usage / self.code_usage.sum()
                active_codes = (usage_probs > self.min_usage_threshold).sum().item()
                
                if active_codes < self.num_embeddings // 4:  # If using less than 25% of codes
                    # Reset rarely used codes to be perturbations of most used codes
                    used_indices = torch.topk(self.code_usage, k=active_codes).indices
                    unused_indices = torch.topk(-self.code_usage, k=self.num_embeddings - active_codes).indices
                    
                    # Get mean and std of used embeddings
                    used_embeddings = self.embed.weight.data[used_indices]
                    mean_embedding = used_embeddings.mean(0)
                    std_embedding = used_embeddings.std(0)
                    
                    # Reset unused embeddings with noise
                    noise = torch.randn_like(self.embed.weight.data[unused_indices]) * 0.1
                    new_embeddings = mean_embedding.unsqueeze(0) + noise * std_embedding.unsqueeze(0)
                    new_embeddings = F.normalize(new_embeddings, dim=1)
                    
                    self.embed.weight.data[unused_indices] = new_embeddings
                    self.code_usage[unused_indices] = self.code_usage.mean()
        
        # Total loss with increased weights for diversity and entropy
        total_loss = (self.commitment_cost * commitment_loss + 
                     self.diversity_weight * diversity_loss + 
                     self.entropy_weight * entropy_loss)
        
        # Straight-through estimator
        quantized = flat_input + (quantized - flat_input).detach()
        quantized = quantized.view(input_shape)
        
        return quantized, total_loss, encoding_indices

    def get_codebook_metrics(self):
        with torch.no_grad():
            # Normalize codebook
            codebook_norm = F.normalize(self.embed.weight, dim=1)
            
            # Compute pairwise similarities
            similarity_matrix = torch.matmul(codebook_norm, codebook_norm.t())
            
            # Get metrics
            avg_similarity = similarity_matrix.mean().item()
            max_similarity = similarity_matrix.max().item()
            
            # Usage statistics
            usage_probs = self.code_usage / self.code_usage.sum()
            entropy = -(usage_probs * torch.log(usage_probs + 1e-10)).sum().item()
            active_codes = (usage_probs > 0.01).sum().item()
            
            return {
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'codebook_entropy': entropy,
                'active_codes': active_codes
            }



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
        
        # EMA buffers
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
        
        # Normalize vectors for cosine similarity
        latent_normalized = F.normalize(flat_input, dim=-1)
        codebook_normalized = F.normalize(self.codebook.weight, dim=-1)
        cosine_sim = torch.matmul(latent_normalized, codebook_normalized.t())
        
        # Quantization distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True) 
            - 2 * torch.matmul(flat_input, self.codebook.weight.t())
            + torch.sum(self.codebook.weight ** 2, dim=1)
        )
        
        # Softmax with temperature for soft assignment
        scaled_distances = distances / max(self.temperature.item(), 1e-5)
        soft_assign = F.softmax(-scaled_distances, dim=-1)
        
        # Hard assignment indices
        indices = soft_assign.argmax(dim=-1)
        hard_assign = F.one_hot(indices, self.num_tokens).float()
        assign = hard_assign + soft_assign - soft_assign.detach()
        
        # Update usage count
        self.usage_count[indices] += 1
        
        # Average cosine similarity for analysis
        selected_cosine_sim = cosine_sim[torch.arange(len(indices)), indices].mean()
        
        # EMA Update
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
        
        # Calculate distances for clustering
        codebook_distances = torch.cdist(self.codebook.weight, self.codebook.weight)
        mask = ~torch.eye(codebook_distances.shape[0], dtype=bool, device=codebook_distances.device)
        masked_distances = codebook_distances[mask]
        avg_euclidean = masked_distances.mean()
        min_euclidean = masked_distances.min()
        
        # Quantize
        quantized = torch.matmul(assign, self.codebook.weight)
        quantized = quantized.view(batch_size, seq_len, latent_dim)
        
        # Losses
        commitment_loss = self.beta * F.mse_loss(latent.detach(), quantized)
        codebook_loss = F.mse_loss(latent, quantized.detach())
        
        # Straight-through estimator
        quantized = latent + (quantized - latent).detach()
        
        # Perplexity
        avg_probs = torch.mean(hard_assign, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return (quantized, indices, commitment_loss, codebook_loss, 
                perplexity, selected_cosine_sim, avg_euclidean, min_euclidean)

class Seq2SeqTransformerPatchBased(nn.Module):
    def __init__(self, image_size, patch_size, action_dim, embed_dim, n_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, num_tokens, latent_dim, beta, max_seq_len=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Linear embedding for patches
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(1, self.num_patches, kernel_size=patch_size, stride=patch_size),  # Convert patches to embeddings
            nn.Flatten(2),  # Flatten spatial dimensions
            nn.Linear(self.num_patches, embed_dim)  # Project flattened patches to embed_dim
        )

        # Action embedding
        self.action_embedding = nn.Embedding(action_dim, embed_dim)

        # Positional embeddings for patches and sequences
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # CLS token
        self.patch_positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim)) 

        self.sequence_positional_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Vector quantizer
        self.vq_layer = VectorQuantizer(num_tokens, embed_dim, beta)

        # Decoder for patch reconstruction
        self.patch_reconstruction_decoder = nn.Sequential(
            nn.Linear(embed_dim, self.num_patches * embed_dim),
            nn.ReLU(),
            nn.Unflatten(1, (embed_dim, self.num_patches)),
            nn.ConvTranspose2d(embed_dim, 1, kernel_size=patch_size, stride=patch_size)
        )

    def generate_square_subsequent_mask(self, size):
        """Generate a causal mask for a sequence."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mask = torch.triu(torch.ones(size, size, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, states, actions, targets, teacher_forcing_ratio=0.5):
        batch_size, seq_len, channels, height, width = states.shape

        # Extract patch embeddings
        patch_embeddings = torch.stack([
            self.patch_embedding(states[:, t].float())
            for t in range(seq_len)
        ], dim=1)  # Shape: (batch_size, seq_len, num_patches, embed_dim)

        # Flatten patches and prepend CLS token
        patch_embeddings = patch_embeddings.view(batch_size, seq_len, self.num_patches, -1)
        cls_tokens = self.cls_token.expand(batch_size, seq_len, -1, -1)  # Expand CLS token
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=2)  # Prepend CLS token

        # Add positional embeddings
        embeddings += self.patch_positional_embedding[:, :embeddings.size(2), :]

        # Encode actions
        action_embeddings = self.action_embedding(actions)  # Shape: (batch_size, seq_len, embed_dim)

        # Add action embeddings to the sequence embeddings
        embeddings[:, :, 0, :] += action_embeddings  # Add actions only to the CLS token
        batch_size, seq_len, num_patches, embed_dim = embeddings.shape
        embeddings = embeddings.view(batch_size * seq_len, num_patches, embed_dim)

        # Transformer encoder
        encoder_out = self.encoder(embeddings)  # Shape: (batch_size, seq_len, num_patches + 1, embed_dim)
        encoder_out = encoder_out.view(batch_size, seq_len, num_patches, embed_dim)

        # Extract CLS token
        cls_token_out = encoder_out[:, :, 0, :]  # CLS token is the first token

        # Quantize CLS token
        quantized, indices, commitment_loss, codebook_loss, perplexity, cosine_sim, avg_euclidean, min_euclidean = self.vq_layer(cls_token_out)


        # Initialize decoder outputs list
        decoder_outs = []

        # Initialize decoder input as zeros
        decoder_input = torch.zeros(batch_size, 1, self.embed_dim, device=states.device)

        for t in range(seq_len):
            # Transformer Decoder
            seq_len = decoder_input.size(1)
            decoder_mask = self.generate_square_subsequent_mask(seq_len).to(states.device)
            decoder_out = self.decoder(decoder_input, quantized, tgt_mask=decoder_mask)

            # Reconstruct patches
            print(decoder_out.shape, quantized.shape)
            rec_patches = self.patch_reconstruction_decoder(decoder_out.squeeze(1))
            decoder_outs.append(rec_patches)

            # Teacher forcing logic
            if t < seq_len - 1:  # Don't need input for the last timestep
                if torch.rand(1).item() < teacher_forcing_ratio:
                    # Use target frame as next input
                    decoder_input = self.patch_embedding(targets[:, t]).unsqueeze(1)
                else:
                    # Use own prediction as next input
                    decoder_input = self.patch_embedding(rec_patches).unsqueeze(1)

        # Stack reconstructed frames
        reconstructed_states = torch.stack(decoder_outs, dim=1)

        recon_loss = F.mse_loss(reconstructed_states, targets)

        # Total Loss
        total_loss = recon_loss + commitment_loss + codebook_loss

        return reconstructed_states, total_loss, {
            "recon_loss": recon_loss.item(),
            "commitment_loss": commitment_loss.item(),
            "codebook_loss": codebook_loss.item(),
            "perplexity": perplexity.item(),
            "avg_euclidean": avg_euclidean.item(),
            "min_euclidean": min_euclidean.item(),
        }



class AtariDataset(Dataset):
    def __init__(self, env_name, context_len=8, frameskip=4):
        from utils import OfflineEnvAtari
        # self.env = gym.make(env_name)
        dataset = OfflineEnvAtari(path='/home/ubuntu/.d4rl/datasets/Seaquest/1/1').get_dataset()
        self.frames = dataset['observations']  # Keep only reference
        self.actions = dataset['actions']
        self.terminals = dataset['terminals']
        self.context_len = context_len
        self.frameskip = frameskip
        
        # Store only indices instead of actual frames
        self.sequences = []
        
        episode_starts = [0] + (np.where(self.terminals)[0] + 1).tolist()
        episode_starts = episode_starts[:-1]  

        for start_idx in episode_starts:
            end_idx = start_idx + np.where(self.terminals[start_idx:])[0][0] + 1 if start_idx < len(self.terminals) - 1 else len(self.frames)
            
            for i in range(start_idx, end_idx - context_len * frameskip):
                frame_indices = [i + j * frameskip for j in range(context_len)]
                target_indices = frame_indices[1:] + [min(i + context_len * frameskip, end_idx - 1)]
                self.sequences.append({
                    "frame_indices": frame_indices,
                    "action_indices": frame_indices,
                    "target_indices": target_indices
                })

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        frames = torch.tensor(self.frames[seq["frame_indices"]], dtype=torch.uint8)/255.
        actions = torch.tensor(self.actions[seq["action_indices"]], dtype=torch.long)
        targets = torch.tensor(self.frames[seq["target_indices"]], dtype=torch.uint8)/255.
        return {"frames": frames, "actions": actions, "targets": targets}

    def __len__(self):
        return len(self.sequences)



class Seq2SeqTransformer(nn.Module):
    def __init__(self, image_size, action_dim, embed_dim, n_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, num_tokens, latent_dim, beta, max_seq_len=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        
        # Calculate the output size after convolutions
        conv1_size = (image_size - 8 + 2 * 1) // 4 + 1  # After first conv
        conv2_size = (conv1_size - 4 + 2 * 1) // 2 + 1  # After second conv
        conv3_size = (conv2_size - 3 + 2 * 1) // 1 + 1  # After third conv
        self.conv_output_size = conv3_size

        # Encoder for image input
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * conv3_size * conv3_size, embed_dim)
        )
        
        # Action embedding
        self.action_embedding = nn.Embedding(action_dim, embed_dim)

        # Positional embeddings
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Vector quantizer
        self.vq_layer = VectorQuantizer(num_tokens, embed_dim, beta)

        # Decoder for reconstruction
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(embed_dim, 128 * conv3_size * conv3_size),
            nn.ReLU(),
            nn.Unflatten(1, (128, conv3_size, conv3_size)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(84, 84), mode='bilinear', align_corners=False), 
            nn.Tanh()
        )

    def generate_square_subsequent_mask(self, size):
        """Generate a causal mask for a sequence."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mask = torch.triu(torch.ones(size, size, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, states, actions, targets, teacher_forcing_ratio=0.5):
        batch_size, seq_len, channels, height, width = states.shape

        # Causal masks
        encoder_mask = self.generate_square_subsequent_mask(seq_len).to(states.device)
        decoder_mask = self.generate_square_subsequent_mask(seq_len).to(states.device)

        # Encode image frames
        encoded_frames = torch.stack([self.image_encoder(states[:, t].float()) for t in range(seq_len)], dim=1)

        # Encode actions
        encoded_actions = self.action_embedding(actions)

        # Combine encoded frames and actions
        seq_embeddings = encoded_frames + encoded_actions
        seq_embeddings += self.positional_embedding[:, :seq_len, :]  # Add positional embeddings

        # Transformer Encoder
        encoder_out = self.encoder(seq_embeddings, mask=encoder_mask)

        # Quantize the encoder output
        quantized, indices, commitment_loss, codebook_loss, perplexity, cosine_sim, avg_euclidean, min_euclidean = self.vq_layer(encoder_out)

        # Initialize decoder outputs list
        decoder_outs = []
        
        # Initialize decoder input as zeros
        decoder_input = torch.zeros(batch_size, 1, self.embed_dim, device=states.device)

        for t in range(seq_len):
            # Transformer Decoder
            seq_len = decoder_input.size(1)
            decoder_mask = self.generate_square_subsequent_mask(seq_len).to(states.device)
            decoder_out = self.decoder(decoder_input, quantized, tgt_mask=decoder_mask)

            
            # Reconstruct the frame
            rec_frame = self.reconstruction_decoder(decoder_out.squeeze(1))
            decoder_outs.append(rec_frame)

            # Teacher forcing logic
            if t < seq_len - 1:  # Don't need input for the last timestep
                if torch.rand(1).item() < teacher_forcing_ratio:
                    # Use target frame as next input
                    decoder_input = self.image_encoder(targets[:, t]).unsqueeze(1)
                else:
                    # Use own prediction as next input
                    decoder_input = self.image_encoder(rec_frame).unsqueeze(1)

        # Stack reconstructed frames
        reconstructed_states = torch.stack(decoder_outs, dim=1)

        recon_loss = F.mse_loss(reconstructed_states, targets)

        # Total Loss
        total_loss = recon_loss + commitment_loss + codebook_loss

        return reconstructed_states, total_loss, {
            "recon_loss": recon_loss.item(),
            "commitment_loss": commitment_loss.item(),
            "codebook_loss": codebook_loss.item(),
            "perplexity": perplexity.item(),
            "avg_euclidean": avg_euclidean.item(),
            "min_euclidean": min_euclidean.item(),
        }

def train_model(model, dataset, epochs, batch_size, lr, teacher_forcing_ratio, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(dataloader):
            states = batch["frames"].to(device).float()
            actions = batch["actions"].to(device)
            targets = batch["targets"].to(device).float()

            reconstructed_states, loss, stats = model(states, actions, targets, teacher_forcing_ratio)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


            
            print("loss: ", loss.item())
            for k, v in stats.items():
                print(f"{k}: {v:.4f}")

            if i > 0 and i % 1000 == 0:
                torch.save(model.state_dict(), f"weights/model_atari.pt")
                save_frames(states, reconstructed_states, epoch, i)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

        teacher_forcing_ratio *= 0.95  # Decay teacher forcing ratio




def save_frames(states, predicted_frames, epoch, batch_idx, max_examples=5, max_frames=30, folder_prefix="frames"):
    import os
    import cv2
    os.makedirs(f'{folder_prefix}_original', exist_ok=True)
    os.makedirs(f'{folder_prefix}_predicted', exist_ok=True)

    for example_idx in range(min(max_examples, states.size(0))):  # Save up to max_examples examples per batch
        for frame_idx in range(min(max_frames, states.size(1))):  # Save up to max_frames per sequence
            original_frame = (states[example_idx, frame_idx, 0].cpu().numpy() * 255).astype('uint8')
            predicted_frame = (predicted_frames[example_idx, frame_idx, 0].detach().cpu().numpy() * 255).astype('uint8')

            # Save original frame
            cv2.imwrite(f'{folder_prefix}_original/epoch_{epoch}_batch_{batch_idx}_example_{example_idx}_frame_{frame_idx}.png', original_frame)
            
            # Save predicted frame
            cv2.imwrite(f'{folder_prefix}_predicted/epoch_{epoch}_batch_{batch_idx}_example_{example_idx}_frame_{frame_idx}.png', predicted_frame)



def main():
    env_name = "seaquest-mixed-v0"
    context_len = 40
    frameskip = 4
    image_size = 84  # Explicit image size
    action_dim = 18
    embed_dim = 32
    n_heads = 8
    num_encoder_layers = 4
    num_decoder_layers = 4
    dim_feedforward = 64
    num_tokens = 64
    latent_dim = 64
    beta = 1.0
    epochs = 10
    batch_size = 4
    lr = 1e-4
    teacher_forcing_ratio = 1.0

    dataset = AtariDataset(env_name, context_len, frameskip)
    # model = Seq2SeqTransformerPatchBased(
    #     image_size=image_size,
    #     patch_size=7,
    #     action_dim=action_dim,
    #     embed_dim=embed_dim,
    #     n_heads=n_heads,
    #     num_encoder_layers=num_encoder_layers,
    #     num_decoder_layers=num_decoder_layers,
    #     dim_feedforward=dim_feedforward,
    #     num_tokens=num_tokens,
    #     latent_dim=latent_dim,
    #     beta=beta,
    #     max_seq_len=40,
    # )
    model = Seq2SeqTransformer(
        image_size=image_size,
        action_dim=action_dim,
        embed_dim=embed_dim,
        n_heads=n_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        num_tokens=num_tokens,
        latent_dim=latent_dim,
        beta=beta
    )
    # model.load_state_dict(torch.load("weights/model_atari.pt"))
    train_model(model, dataset, epochs, batch_size, lr, teacher_forcing_ratio, device="cuda")

if __name__ == "__main__":
    main()
