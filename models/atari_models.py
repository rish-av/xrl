import torch
import torch.nn as nn
import wandb
import argparse
from torch.utils.tensorboard import SummaryWriter
import cv2
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import gym
from utils import OfflineEnvAtari, BehaviorAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Atari Model with Custom Model Name in WandB")
    
    # Model hyperparameters
    parser.add_argument('--max_seq_len', type=int, default=60, help="Maximum sequence length")
    parser.add_argument('--embed_dim', type=int, default=128, help="Embedding dimension")
    parser.add_argument('--nhead', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of transformer layers")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--num_embeddings', type=int, default=128, help="Number of embeddings for VQ")

    # Logging and experiment tracking
    parser.add_argument('--log', action='store_true', help="Enable logging with WandB")
    parser.add_argument('--wandb_project', type=str, default="atari-xrl", help="WandB project name")
    parser.add_argument('--wandb_entity', type=str, default="mail-rishav9", help="WandB entity name")
    parser.add_argument('--wandb_run_name', type=str, default="xrl_atari", help="Base name for WandB run")
    parser.add_argument('--patch_size', type=int, default=4, help="Patch size for image transformer")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument('--load_checkpoint', type=str, default=None, help="Path to load model checkpoint")
    parser.add_argument('--epochs', type=int, default=2000, help="Number of training epochs")
    parser.add_argument('--scheduler_step_size', type=int, default=50, help="Step size for learning rate scheduler")
    parser.add_argument('--frame_skip', type=int, default=4, help="Number of frames to skip in dataset")

    return parser.parse_args()



def generate_run_name(args):
    """
    Generates a unique run name by combining key hyperparameters.
    Args:
        args: Parsed arguments from argparse.

    Returns:
        str: A descriptive run name.
    """
    run_name_parts = [
        args.wandb_run_name,  # Base run name
        f"seq{args.max_seq_len}",  # Sequence length
        f"embed{args.embed_dim}",  # Embedding dimension
        f"heads{args.nhead}",  # Attention heads
        f"layers{args.num_layers}",  # Transformer layers
        f"batch{args.batch_size}",  # Batch size
        f"lr{args.learning_rate:.0e}"  # Learning rate (scientific notation)
        f"vq{args.num_embeddings}"  # Number of embeddings for VQ
        f"skip{args.frame_skip}" # Frame skip
    ]
    
    # Filter out any empty parts and join with underscores
    run_name = "_".join(map(str, run_name_parts))
    return run_name

args = parse_args()



if args.log:
    run_name = generate_run_name(args)
    wandb.init(project='atari_xrl_2', entity=args.wandb_entity, name=run_name, config=vars(args))




class MiniGridDataset(Dataset):
    def __init__(self, env_name, max_seq_len=30, transform=None, frame_skip=1, action_dim=6):
        self.dataset = gym.make(env_name).get_dataset()
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.frame_skip = frame_skip
        self.action_dim = action_dim
        self.total_frames = len(self.dataset['observations'])
        self.valid_starts = self._get_valid_start_indices()

    def _get_valid_start_indices(self):
        valid_starts = []
        for i in range(0, self.total_frames - self.max_seq_len * self.frame_skip, self.frame_skip):
            terminals = self.dataset['terminals'][i:i + self.max_seq_len * self.frame_skip:self.frame_skip]
            if not np.any(terminals):
                valid_starts.append(i)
        return valid_starts

    def _one_hot_encode(self, actions):
        one_hot = np.zeros((len(actions), self.action_dim), dtype=np.float32)
        for idx, action in enumerate(actions):
            print(action)
            one_hot[idx, int(action)] = 1.0
        return one_hot

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]
        indices = range(start_idx, start_idx + self.max_seq_len * self.frame_skip, self.frame_skip)
        states = [self.dataset['observations'][i] for i in indices]
        actions = [int(self.dataset['actions'][i]) for i in indices]  # Ensure actions are integers
        target_states = [self.dataset['observations'][i + 1] for i in indices]
        states = np.array(states)
        target_states = np.array(target_states)
        actions = np.array(actions, dtype=int)  # Explicit integer conversion
        states = torch.FloatTensor(states).permute(0, 3, 1, 2)
        actions = torch.LongTensor(actions)  # Ensure LongTensor receives integers
        target_states = torch.FloatTensor(target_states).permute(0, 3, 1, 2)
        return states, actions, target_states



class AtariGrayscaleDataset(Dataset):
    def __init__(self, dataset_path, max_seq_len=30, transform=None, frame_skip=4):
        self.dataset = OfflineEnvAtari(stack=True, path=dataset_path).get_dataset()  # Load metadata
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.frame_skip = frame_skip

        self.total_frames = len(self.dataset['observations'])  # Total number of frames
        self.valid_starts = self._get_valid_start_indices()

    def _get_valid_start_indices(self):
        valid_starts = []
        for i in range(0, self.total_frames - self.max_seq_len * self.frame_skip, self.frame_skip):
            # Check for terminal flags if available
            if 'terminals' in self.dataset:
                terminals = self.dataset['terminals'][i:i+self.max_seq_len*self.frame_skip:self.frame_skip]
                if not np.any(terminals):  # No terminal flag in sequence
                    valid_starts.append(i)
            else:
                valid_starts.append(i)  # Assume all starts are valid if no terminal flags
        return valid_starts

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]
        indices = range(start_idx, start_idx + self.max_seq_len * self.frame_skip, self.frame_skip)

        states = [self.dataset['observations'][i] for i in indices]
        actions = [self.dataset['actions'][i] for i in indices]
        target_states = [self.dataset['observations'][i+1] for i in indices]

        # Normalize frames
        states = np.array(states) / 255.0
        target_states = np.array(target_states) / 255.0

        # Apply transform if provided
        if self.transform:
            states = self.transform(states)
            target_states = self.transform(target_states)

        # Convert to tensors
        states = torch.FloatTensor(states) # Add channel dimension
        actions = torch.LongTensor(actions)
        target_states = torch.FloatTensor(target_states) # Add channel dimension

        return states, actions, target_states

def diversity_loss(embeddings):
    similarity = torch.matmul(embeddings, embeddings.t())
    identity = torch.eye(embeddings.shape[0], device=embeddings.device)
    return F.mse_loss(similarity, identity)

def entropy_regularization(quantized_indices, num_embeddings):
    # Calculate usage probability for each code
    code_probs = F.one_hot(quantized_indices, num_embeddings).float().mean(0)
    # Add small epsilon to avoid log(0)
    entropy = -(code_probs + 1e-8).log() * code_probs
    return -entropy.sum()  # Negative because we want to maximize entropy



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
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.orthogonal_(self.embed.weight.data)
        self.register_buffer('code_usage', torch.zeros(num_embeddings))
        
    def reinit_unused_codes(self):
        total_usage = self.code_usage.sum()
        usage_prob = self.code_usage / total_usage
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




class AtariPatchSeq2SeqTransformerVQ(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_heads, num_layers, action_dim, num_embeddings, max_seq_len=1024):
        super(AtariPatchSeq2SeqTransformerVQ, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        # Patch embedding layer: converts image patches to embeddings
        self.patch_embed = nn.Sequential(
            nn.Conv2d(4, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.LayerNorm([embed_dim, img_size[0] // patch_size, img_size[1] // patch_size])
        )
        
        # Linear projection for patches
        patch_dim = embed_dim * (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.patch_projection = nn.Linear(patch_dim, embed_dim)
        
        # Action embedding
        self.action_embedding = nn.Embedding(action_dim, embed_dim)
        
        # Positional embeddings (for both patches and time)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, self.num_patches + 1, embed_dim))
        
        # Special [CLS] token for state-level representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        
        # Transformer layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=4 * embed_dim, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=4 * embed_dim, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        
        # Vector Quantizer
        self.vq = EnhancedVectorQuantizer(num_embeddings, embed_dim)
        
        # State-level aggregation
        self.state_aggregation = nn.Sequential(
            nn.Linear(embed_dim * (self.num_patches + 1), embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Frame reconstruction
        self.reconstruct = nn.Sequential(
            nn.Linear(embed_dim, 4 * patch_size * patch_size),  # Output channel=4, size=patch_size
            nn.ReLU(),
            nn.Unflatten(1, (4, patch_size, patch_size)),  # Directly reshape to target patch size
            nn.Sigmoid()
        )

    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask.to(device)

    def forward(self, states, actions, target_states=None):
        B, T, C, H, W = states.shape
        
        # Extract patches and embed them
        states_reshaped = states.view(-1, C, H, W)
        patch_embeddings = self.patch_embed(states_reshaped)  # [B*T, E, H', W']
        
        # Reshape to [B, T, num_patches, embed_dim]
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)
        patch_embeddings = patch_embeddings.view(B, T, self.num_patches, -1)
        
        # Add CLS token to each timestep
        cls_tokens = self.cls_token.expand(B, T, -1, -1)
        patch_embeddings = torch.cat([cls_tokens, patch_embeddings], dim=2)
        
        # Add positional embeddings
        patch_embeddings = patch_embeddings + self.pos_embed[:, :T, :, :]
        
        # Embed actions and combine
        action_embeddings = self.action_embedding(actions)  # [B, T, embed_dim]
        action_embeddings = action_embeddings.unsqueeze(2).expand(-1, -1, self.num_patches + 1, -1)
        combined_embeddings = patch_embeddings + action_embeddings
        
        # Reshape for transformer: [B, T*(num_patches+1), embed_dim]
        seq_embeddings = combined_embeddings.view(B, T * (self.num_patches + 1), -1)
        
        # Generate causal mask
        causal_mask = self.generate_causal_mask(seq_embeddings.size(1), states.device)
        
        # Forward through encoder
        encoder_output = self.encoder(seq_embeddings, mask=causal_mask)
        
        # State-level aggregation before quantization
        encoder_output = encoder_output.view(B, T, self.num_patches + 1, -1)
        state_repr = self.state_aggregation(encoder_output.flatten(2))  # [B, T, embed_dim]
        
        # Apply Vector Quantizer on state-level representations
        quantized, vq_loss, encoding_indices = self.vq(state_repr)
        
        if target_states is not None:
            # Process target states similarly
            target_embeddings = self.patch_embed(target_states.view(-1, C, H, W))
            target_embeddings = target_embeddings.flatten(2).transpose(1, 2)
            target_embeddings = target_embeddings.view(B, T, self.num_patches, -1)
            target_cls = self.cls_token.expand(B, T, -1, -1)
            target_embeddings = torch.cat([target_cls, target_embeddings], dim=2)
            target_embeddings = target_embeddings + self.pos_embed[:, :T, :, :]
            
            # Shift target sequence by one timestep
            target_embeddings = target_embeddings[:, :-1]  # [B, T-1, num_patches+1, embed_dim]
            target_embeddings = target_embeddings.view(B, (T-1) * (self.num_patches + 1), -1)
            
            # Prepare memory from quantized embeddings
            memory = quantized.unsqueeze(2).expand(-1, -1, self.num_patches + 1, -1)  # [B, T, num_patches+1, embed_dim]
            memory = memory.reshape(B, T * (self.num_patches + 1), -1)  # [B, T*(num_patches+1), embed_dim]
            
            # Decoder forward pass
            tgt_mask = self.generate_causal_mask(target_embeddings.size(1), states.device)
            outputs = self.decoder(target_embeddings, memory, tgt_mask=tgt_mask)
        else:
            outputs = quantized.unsqueeze(2).expand(-1, -1, self.num_patches + 1, -1)
            outputs = outputs.reshape(B, T * (self.num_patches + 1), -1)
            
        # Reconstruct frames
        outputs = outputs.view(B * (T if target_states is None else T-1), self.num_patches + 1, self.embed_dim)
        outputs = outputs[:, 1:, :]  # Remove CLS token
        
        # Process each patch independently
        patch_reconstructions = []
        for i in range(self.num_patches):
            patch_emb = outputs[:, i, :]  # [B*T, embed_dim]
            patch_reconstruction = self.reconstruct(patch_emb)  # [B*T, 4, patch_size, patch_size]
            patch_reconstructions.append(patch_reconstruction)
            
        # Combine patches
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        
        predicted_frames = []
        for i in range(len(patch_reconstructions)):
            y = (i // w_patches) * self.patch_size
            x = (i % w_patches) * self.patch_size
            predicted_frames.append((y, x, patch_reconstructions[i]))
            
        # Create final output
        final_frames = torch.zeros(B * (T if target_states is None else T-1), C, H, W, device=states.device)
        for y, x, patch in predicted_frames:
            final_frames[:, :, y:y+self.patch_size, x:x+self.patch_size] = patch
            
        final_frames = final_frames.view(B, -1, C, H, W)
        
        return final_frames, vq_loss, encoding_indices




class AtariSeq2SeqTransformerVQ(nn.Module):
    def __init__(self, img_size, embed_dim, num_heads, num_layers, action_dim, num_embeddings, max_seq_len=1024):
        super(AtariSeq2SeqTransformerVQ, self).__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim

        # State encoder: extracts spatial embeddings from frames
        self.state_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, embed_dim),  # Match embedding dimension
            nn.Tanh()
        )

        # Action embedding: converts discrete actions to embeddings
        self.action_embedding = nn.Embedding(action_dim, embed_dim)

        # Positional embeddings
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        # Transformer encoder-decoder architecture
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=4 * embed_dim, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=4 * embed_dim, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # Vector Quantizer
        self.vq = EnhancedVectorQuantizer(num_embeddings, embed_dim)

        # Frame reconstruction head
        self.reconstruct = nn.Sequential(
            nn.Linear(embed_dim, 3136),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=8, stride=4, padding=0), nn.Sigmoid()  # Normalize output
        )

    def generate_causal_mask(self, seq_len, device):
        """
        Generate a causal mask to ensure predictions depend only on past and current inputs.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask.to(device)

    def forward(self, states, actions, target_states=None, sampling_probability=0.9):
        B, T, C, H, W = states.shape

        # Encode states into embeddings
        state_embeddings = self.state_encoder(states.view(-1, C, H, W))
        state_embeddings = state_embeddings.view(B, T, -1)

        # Embed actions
        action_embeddings = self.action_embedding(actions)

        # Combine state and action embeddings
        seq_embeddings = state_embeddings + action_embeddings
        seq_embeddings += self.positional_embedding[:, :T, :]

        # Forward through encoder
        encoder_output = self.encoder(seq_embeddings)
        
        # Apply Vector Quantizer
        quantized, vq_loss, encoding_indices = self.vq(encoder_output)

        predicted_frames = []
        decoder_inputs = torch.zeros((B, T, self.embed_dim), device=states.device)

        if target_states is not None:
            # Training mode with teacher forcing
            target_embeddings = self.state_encoder(target_states.view(-1, C, H, W))
            target_embeddings = target_embeddings.view(B, T, -1)

            for t in range(T - 1):
                # if t == 0 or torch.rand(1).item() > sampling_probability:
                decoder_inputs[:, t, :] = target_embeddings[:, t, :]
                # else:
                #     prev_embedding = self.state_encoder(prev_output.view(-1, C, H, W))
                #     decoder_inputs[:, t, :] = prev_embedding

                # Add positional embeddings
                curr_decoder_input = decoder_inputs[:, :t+1, :] + self.positional_embedding[:, :t+1, :]
                
                # Forward through decoder
                output = self.decoder(
                    curr_decoder_input,
                    quantized[:, :t+1, :],
                    tgt_mask=self.generate_causal_mask(t+1, states.device)
                )

                # Reconstruct frame from the last output
                prev_output = self.reconstruct(output[:, -1])
                predicted_frames.append(prev_output)

        else:
            # Inference mode
            for t in range(T - 1):
                if t == 0:
                    decoder_inputs[:, 0, :] = state_embeddings[:, 0, :]
                else:
                    prev_embedding = self.state_encoder(prev_output.view(-1, C, H, W))
                    decoder_inputs[:, t, :] = prev_embedding

                # Add positional embeddings
                curr_decoder_input = decoder_inputs[:, :t+1, :] + self.positional_embedding[:, :t+1, :]
                
                # Forward through decoder
                output = self.decoder(
                    curr_decoder_input,
                    quantized[:, :t+1, :],
                    tgt_mask=self.generate_causal_mask(t+1, states.device)
                )

                # Reconstruct frame from the last output
                prev_output = self.reconstruct(output[:, -1])
                predicted_frames.append(prev_output)

        predicted_frames = torch.stack(predicted_frames, dim=1)
        return predicted_frames, vq_loss, encoding_indices



def save_frames(states, predicted_frames, epoch, batch_idx, max_examples=5, max_frames=30, folder_prefix="frames"):
    """
    Save original and predicted frames for visualization.
    """
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


def save_model_checkpoint(model, epoch, save_dir="checkpoints"):
    """
    Save the model checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved at {checkpoint_path}")





def train_model(model, dataset, epochs=2000, batch_size=4, lr=1e-4, scheduler_step_size=50, scheduler_gamma=0.9, save_interval=5, collate_fn=None):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            states, actions, target_states = batch
            states = states.to(device)
            actions = actions.to(device).long()
            target_states = target_states.to(device)

            # Forward pass
            predicted_frames, vq_loss, indices = model(states, actions, target_states)
            reconstruction_loss = criterion(predicted_frames, target_states[:, 1:])
            loss = reconstruction_loss + vq_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate batch loss
            epoch_loss += loss.item()

            if args.log:
                wandb.log({
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "loss": loss.item(),
                    "reconstruction_loss": reconstruction_loss.item(),
                    "vq_loss": vq_loss.item(),
                    "unique_indices": torch.unique(indices).size(0)
                })

            # Logging loss for each batch
            if indices is not None:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Recon Loss: {reconstruction_loss.item():.4f}, VQ Loss: {vq_loss.item()}, Unique Indices: {len(torch.unique(indices))} ")
                metrics = model.vq.get_codebook_metrics()
                print(f"Active codes: {metrics['active_codes']}")
                print(f"Codebook entropy: {metrics['codebook_entropy']:.2f}")
                print(f"Average similarity: {metrics['avg_similarity']:.3f}")
            else:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}],  Recon Loss: {reconstruction_loss.item():.4f}, VQ Loss: {vq_loss.item}")


            if isinstance(model, AtariPatchSeq2SeqTransformerVQ) or isinstance(model, AtariSeq2SeqTransformerVQ):
                model.vq.reinit_unused_codes()
            # Save intermediate predictions every 40 batches

            if args.log:
                metrics = model.vq.get_codebook_metrics()
                wandb.log({
                    "active_codes": metrics["active_codes"],
                    "codebook_entropy": metrics["codebook_entropy"],
                    "avg_similarity": metrics["avg_similarity"]
                    })

            # if batch_idx % 100 == 0:
            #     metrics = model.vq.get_codebook_metrics()
            #     print(f"Active codes: {metrics['active_codes']}")
            #     print(f"Codebook entropy: {metrics['codebook_entropy']:.2f}")
            #     print(f"Average similarity: {metrics['avg_similarity']:.3f}")
            

        # Average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}")
        
        # Scheduler step
        scheduler.step()

        # Save the model checkpoint every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            save_model_checkpoint(model, epoch + 1)
            print(f"Model checkpoint saved at epoch {epoch + 1}")





# embed_dim = 32
# num_heads = 4
# num_layers = 2
# action_dim = 6  
# num_embeddings = 32
# max_seq_len = 30

# # Initialize the model
# model = MiniGridSeq2SeqTransformerVQ(
#     embed_dim=embed_dim,
#     num_heads=num_heads,
#     num_layers=num_layers,
#     action_dim=action_dim,
#     num_embeddings=num_embeddings,
#     max_seq_len=max_seq_len
# ).to("cuda")


# dataset = MiniGridDataset('minigrid-fourrooms-v0')
# train_model(model, dataset, epochs=2000, batch_size=32, lr=1e-4, scheduler_step_size=50, scheduler_gamma=0.9, save_interval=50, collate_fn=collate_fn)


# model = AtariSeq2SeqTransformerVQ(
#     img_size=(84, 84),
#     # patch_size=4,
#     embed_dim=args.embed_dim,
#     num_heads=args.nhead,
#     num_layers=args.num_layers,
#     action_dim=18,
#     num_embeddings=args.num_embeddings,
#     max_seq_len=args.max_seq_len,
# ).to("cuda")


# if args.load_checkpoint is not None:
#     model.load_state_dict(torch.load(args.load_checkpoint))
#     print(f"Model loaded from {args.load_checkpoint}")



# dataset = AtariGrayscaleDataset( dataset_path='/home/ubuntu/.d4rl/datasets/Seaquest/1/1', max_seq_len=args.max_seq_len, frame_skip=args.frame_skip)
# print(f"Dataset loaded with {len(dataset)} sequences.")

# # Define training parameters
# epochs = args.epochs
# learning_rate = args.learning_rate
# batch_size = args.batch_size
# scheduler_step_size = args.scheduler_step_size
# scheduler_gamma = 0.9
# train_model(model, dataset, epochs=epochs, batch_size=batch_size, lr=learning_rate, scheduler_step_size=scheduler_step_size, scheduler_gamma=scheduler_gamma)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# dataset = AtariGrayscaleDataset(dataset_path='/home/rishav/scratch/d4rl_dataset/Seaquest/1/10', max_seq_len=60, frame_skip=args.frame_skip)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
# analyzer = BehaviorAnalyzer(model, device)
# codes, transitions = analyzer.extract_behaviors(dataloader, num_samples=1000)
# print(f"Extracted {len(codes)} behavior codes and {len(transitions)} transitions.")
# transition_matrix = analyzer.analyze_transitions(transitions)
# print("Transition matrix calculated.")
# analyzer.visualize_behaviors(codes, save_path="behavior_embeddings.png")
# print("Behavior embeddings visualization saved as 'behavior_embeddings.png'.")
# analyzer.visualize_transitions(transition_matrix, save_path="transition_matrix.png")
# print("Transition matrix visualization saved as 'transition_matrix.png'.")