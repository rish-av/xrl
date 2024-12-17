import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import OfflineEnvAtari
import numpy as np
from torch.utils.data import Dataset, DataLoader

from torch.optim.lr_scheduler import StepLR
import wandb
import random
import argparse
import os
import cv2
from sklearn.cluster import KMeans





class AtariGrayscaleDataset(Dataset):
    def __init__(self, env_name, max_seq_len=50, image_size=(84, 84), frame_skip=4):
        # Initialize the dataset
        self.data = OfflineEnvAtari(stack=False, path='/home/rishav/scratch/d4rl_dataset/Seaquest/1/10').get_dataset()
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.frame_skip = frame_skip

        # Extract relevant data
        self.frames = self.data['observations']  # Grayscale image frames
        self.actions = self.data['actions']
        self.terminals = self.data['terminals']

        # Split trajectories
        self.trajectories = self._split_trajectories()

    def _split_trajectories(self):
        """
        Splits the dataset into trajectories based on terminal flags.
        Applies frame-skipping by selecting every `frame_skip`-th frame.
        """
        trajectories = []
        trajectory = {'frames': [], 'actions': []}

        for i in range(len(self.frames)):
            if i % self.frame_skip == 0:  # Apply frame-skip
                trajectory['frames'].append(self.frames[i])
                trajectory['actions'].append(self.actions[i])

            if self.terminals[i]:  # End of trajectory
                if len(trajectory['frames']) > self.max_seq_len:  # Ignore trajectories with single frames
                    trajectories.append(trajectory)
                trajectory = {'frames': [], 'actions': []}

        return trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        """
        Samples a random sequence from the trajectory.
        """
        trajectory = self.trajectories[idx]
        frames = trajectory['frames']
        actions = trajectory['actions']

        # Sample a random sequence from the trajectory
        traj_len = len(frames) - 1  # Exclude the last frame since it has no "next frame"
        seq_len = min(self.max_seq_len, traj_len)
        start_idx = np.random.randint(0, traj_len - seq_len + 1)

        frame_seq = frames[start_idx:start_idx + seq_len]
        action_seq = actions[start_idx:start_idx + seq_len]
        next_frame_seq = frames[start_idx + 1:start_idx + seq_len + 1]  # Next frames are shifted by 1

        # Normalize and convert images to tensor
        frame_seq = torch.tensor(frame_seq, dtype=torch.float32) / 255.0  # Normalize pixel values
        next_frame_seq = torch.tensor(next_frame_seq, dtype=torch.float32) / 255.0

        return frame_seq, torch.tensor(action_seq, dtype=torch.float32), next_frame_seq


class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, epsilon=1e-5, commitment_cost=1.0):
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

        # Affine parameters for reparameterization
        self.affine_mean = nn.Parameter(torch.zeros(embedding_dim))
        self.affine_std = nn.Parameter(torch.ones(embedding_dim))

    def replace_dead_codes(self, usage_threshold=10):
        """Replace underutilized embeddings."""
        underutilized = self.cluster_size < usage_threshold
        if underutilized.any():
            with torch.no_grad():
                self.embedding.data[underutilized] = torch.randn_like(self.embedding[underutilized]) * 0.1
                self.cluster_size.data[underutilized] = usage_threshold

    def initialize_codebook_with_kmeans(self, inputs):
        flat_input = inputs.view(-1, self.embedding_dim).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_embeddings, random_state=0)
        kmeans.fit(flat_input)
        centroids = kmeans.cluster_centers_
        self.embedding.data.copy_(torch.tensor(centroids, dtype=torch.float32))



    def forward(self, x):
        # Flatten input to (batch_size * seq_len, embedding_dim)
        flat_x = x.reshape(-1, self.embedding_dim)

        # Apply affine reparameterization
        embedding = self.affine_mean + self.affine_std * self.embedding

        # Compute distances and get encoding indices
        distances = torch.cdist(flat_x, embedding)
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = embedding[encoding_indices].view(x.shape)

        # Compute losses
        codebook_loss = F.mse_loss(quantized.detach(), x)
        commitment_loss = F.mse_loss(quantized, x.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # EMA updates for the codebook
        if self.training:
            encoding_one_hot = F.one_hot(encoding_indices, self.num_embeddings).type_as(flat_x)
            new_cluster_size = encoding_one_hot.sum(dim=0)
            self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

            dw = encoding_one_hot.t() @ flat_x
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Normalize the embeddings
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            self.embedding.data = self.ema_w / cluster_size.unsqueeze(1)

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        return quantized, vq_loss, encoding_indices



class CausalGrayscaleImageTransformerWithVQ(nn.Module):
    def __init__(self, action_dim, embed_dim, nhead, num_layers, num_embeddings, image_size=(84, 84), patch_size=7):
        super(CausalGrayscaleImageTransformerWithVQ, self).__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        # CLS and EOS tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # Learnable CLS token per image
        self.eos_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # Learnable EOS token for the sequence

        # Linear layer to embed patches
        self.patch_embedding = nn.Linear(patch_size * patch_size, embed_dim)

        # Action embedding
        self.action_embedding = nn.Embedding(action_dim, embed_dim)

        # Positional embedding
        self.positional_embedding = nn.Embedding(self.num_patches_per_frame + 1, embed_dim)  # Includes CLS

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Vector Quantizer
        self.vq = EMAVectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embed_dim)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Reconstruction layers
        self.reconstruct = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Upsample(size=image_size, mode='bilinear', align_corners=False),
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def generate_causal_mask(self, seq_len, device):
        """
        Generate a causal mask to ensure predictions depend only on past and current inputs.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask.to(device)

    def extract_patches(self, frames):
        """
        Extract non-overlapping patches from the input frames.
        """
        batch_size, seq_len, _, height, width = frames.size()
        patch_size = self.patch_size
        patches = F.unfold(
            frames.view(batch_size * seq_len, 1, height, width),  # Combine batch and time dimensions
            kernel_size=patch_size,
            stride=patch_size
        )  # Output shape: [B*T, patch_size*patch_size, num_patches_per_frame]
        patches = patches.transpose(1, 2)  # [B*T, num_patches_per_frame, patch_size*patch_size]
        return patches.view(batch_size, seq_len, self.num_patches_per_frame, -1)  # [B, T, P, patch_size*patch_size]

    def forward(self, frames, actions, targets=None):
        """
        Forward pass with optional teacher forcing.
        frames: [B, T, 1, H, W] (input frames)
        actions: [B, T] (action indices)
        targets: [B, T, 1, H, W] (ground truth frames for teacher forcing, optional)
        """
        batch_size, seq_len, _, _, _ = frames.size()

        # Extract patches and embed them
        patches = self.extract_patches(frames)  # [B, T, P, patch_size*patch_size]
        patch_embeds = self.patch_embedding(patches)  # [B, T, P, embed_dim]

        # Embed actions
        action_embeds = self.action_embedding(actions)  # [B, T, embed_dim]
        action_embeds = action_embeds.unsqueeze(2).expand(-1, -1, self.num_patches_per_frame, -1)  # [B, T, P, embed_dim]

        # Positional embeddings
        positions = torch.arange(self.num_patches_per_frame).to(frames.device)
        position_embeds = self.positional_embedding(positions)  # [P, embed_dim]
        position_embeds = position_embeds.unsqueeze(0).unsqueeze(0)  # [1, 1, P, embed_dim]

        # Add CLS token per image
        cls_token = self.cls_token.expand(batch_size, seq_len, 1, self.embed_dim)  # [B, T, 1, embed_dim]
        encoder_input = torch.cat([cls_token, patch_embeds], dim=2)  # [B, T, P+1, embed_dim]

        # Flatten for encoder
        encoder_input = encoder_input.view(batch_size * seq_len, self.num_patches_per_frame + 1, self.embed_dim)

        # Encoder output
        encoder_output = self.encoder(encoder_input)  # [B*T, P+1, embed_dim]

        # Extract CLS token (first token)
        cls_token_output = encoder_output[:, 0, :]  # [B*T, embed_dim]

        # Quantize CLS token only
        quantized, vq_loss, encoding_indices = self.vq(cls_token_output)  # Quantized CLS token and VQ loss

        # Prepare decoder input
        if targets is not None:
            # Teacher forcing: Use ground truth patches
            target_patches = self.extract_patches(targets)  # [B, T, P, patch_size*patch_size]
            decoder_input = self.patch_embedding(target_patches).view(
                batch_size * seq_len, self.num_patches_per_frame, self.embed_dim
            )  # [B*T, P, embed_dim]
        else:
            # Inference: Use quantized CLS token as a global representation
            decoder_input = quantized.unsqueeze(1).expand(-1, self.num_patches_per_frame, -1)  # [B*T, P, embed_dim]

        # Generate causal mask for the decoder
        tgt_mask = self.generate_causal_mask(decoder_input.size(1), frames.device)

        # Decoder output
        decoder_output = self.decoder(decoder_input, quantized.unsqueeze(1), tgt_mask=tgt_mask)  # [B*T, P, embed_dim]

        # Reshape decoder output for reconstruction
        decoder_output = decoder_output.view(
            batch_size, seq_len, self.num_patches_per_frame, self.embed_dim
        ).permute(0, 1, 3, 2)  # [B, T, embed_dim, num_patches]
        decoder_output = decoder_output.view(
            batch_size * seq_len, self.embed_dim, self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size
        )  # [B*T, embed_dim, H_patches, W_patches]

        # Reconstruct full frames
        next_frames = self.reconstruct(decoder_output)  # [B*T, 1, H, W]
        next_frames = next_frames.view(batch_size, seq_len, 1, *self.image_size)  # [B, T, 1, H, W]
        return next_frames, vq_loss, encoding_indices




class CausalGrayscaleImageTransformerWithVQDT(nn.Module):
    def __init__(self, action_dim, embed_dim, nhead, num_layers, num_embeddings, image_size=(84, 84), patch_size=7):
        super(CausalGrayscaleImageTransformerWithVQDT, self).__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        # CLS and EOS tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # Learnable CLS token per image
        self.eos_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # Learnable EOS token for the sequence

        # Linear layer to embed patches
        self.patch_embedding = nn.Linear(patch_size * patch_size, embed_dim)

        # Action embedding
        self.action_embedding = nn.Embedding(action_dim, embed_dim)

        # Positional embedding
        self.positional_embedding = nn.Embedding(1024, embed_dim)  # Large enough for long sequences

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Vector Quantizer
        self.vq = EMAVectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embed_dim)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Reconstruction layers
        self.reconstruct = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Upsample(size=image_size, mode='bilinear', align_corners=False),
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def generate_causal_mask(self, seq_len, device):
        """
        Generate a causal mask to ensure predictions depend only on past and current inputs.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask.to(device)

    def extract_patches(self, frames):
        """
        Extract non-overlapping patches from the input frames.
        """
        batch_size, seq_len, _, height, width = frames.size()
        patch_size = self.patch_size
        patches = F.unfold(
            frames.view(batch_size * seq_len, 1, height, width),  # Combine batch and time dimensions
            kernel_size=patch_size,
            stride=patch_size
        )  # Output shape: [B*T, patch_size*patch_size, num_patches_per_frame]
        patches = patches.transpose(1, 2)  # [B*T, num_patches_per_frame, patch_size*patch_size]
        return patches.view(batch_size, seq_len, self.num_patches_per_frame, -1)  # [B, T, P, patch_size*patch_size]

    def forward(self, frames, actions, targets=None):
        """
        Forward pass with optional teacher forcing.
        frames: [B, T, 1, H, W] (input frames)
        actions: [B, T] (action indices)
        targets: [B, T, 1, H, W] (ground truth frames for teacher forcing, optional)
        """
        batch_size, seq_len, _, _, _ = frames.size()

        # Extract patches and embed them
        patches = self.extract_patches(frames)  # [B, T, P, patch_size*patch_size]
        patch_embeds = self.patch_embedding(patches)  # [B, T, P, embed_dim]

        # CLS token for global state embedding
        cls_token = self.cls_token.expand(batch_size, seq_len, 1, self.embed_dim)  # [B, T, 1, embed_dim]
        state_tokens = torch.cat([cls_token, patch_embeds.mean(dim=2, keepdim=True)], dim=2)  # [B, T, 1+P, embed_dim]
        state_tokens = state_tokens[:, :, 0, :]  # Use CLS token for state embedding [B, T, embed_dim]

        # Embed actions
        action_tokens = self.action_embedding(actions)  # [B, T, embed_dim]

        # Combine state and action tokens like Decision Transformer
        combined_tokens = torch.stack([state_tokens, action_tokens], dim=2)  # [B, T, 2, embed_dim]
        combined_tokens = combined_tokens.view(batch_size, -1, self.embed_dim)  # [B, T*2, embed_dim]

        # Add positional embeddings
        positions = torch.arange(combined_tokens.size(1), device=frames.device).unsqueeze(0)  # [1, T*2]
        combined_tokens += self.positional_embedding(positions)  # Add positional embeddings

        # Flatten for encoder
        encoder_output = self.encoder(combined_tokens)  # [B, T*2, embed_dim]

        # Quantize CLS token
        cls_output = encoder_output[:, 0::2]  # Select CLS tokens (state representations) [B, T, embed_dim]
        quantized, vq_loss, encoding_indices = self.vq(cls_output)  # Quantized output and VQ loss


        quantized = quantized.view(batch_size, seq_len, -1)  # [B, T, embed_dim]

        # Prepare decoder input
        if targets is not None:
            # Teacher forcing: Use ground truth patches
            target_patches = self.extract_patches(targets)  # [B, T, P, patch_size*patch_size]
            decoder_input = self.patch_embedding(target_patches).view(
                batch_size * seq_len, self.num_patches_per_frame, self.embed_dim
            )  # [B*T, P, embed_dim]
        else:
            # Inference: Use quantized CLS token
            decoder_input = quantized.expand(-1, self.num_patches_per_frame, -1)  # [B*T, P, embed_dim]

        # Reshape decoder_input to match batch size and sequence structure
        decoder_input = decoder_input.view(batch_size, seq_len * self.num_patches_per_frame, self.embed_dim)  # [B, T*P, embed_dim]

        # Generate causal mask for the decoder
        tgt_mask = self.generate_causal_mask(decoder_input.size(1), frames.device)

        # Transformer decoder forward pass
        decoder_output = self.decoder(decoder_input, quantized, tgt_mask=tgt_mask)

        # Reshape decoder output for reconstruction
        decoder_output = decoder_output.view(
            batch_size, seq_len, self.num_patches_per_frame, self.embed_dim
        ).permute(0, 1, 3, 2)  # [B, T, embed_dim, num_patches]
        decoder_output = decoder_output.view(
            batch_size * seq_len, self.embed_dim, self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size
        )  # [B*T, embed_dim, H_patches, W_patches]

        # Reconstruct full frames
        next_frames = self.reconstruct(decoder_output)  # [B*T, 1, H, W]
        next_frames = next_frames.view(batch_size, seq_len, 1, *self.image_size)  # [B, T, 1, H, W]
        return next_frames, vq_loss, encoding_indices



def train_model_patch_crop(
    model, dataloader, optimizer_task, optimizer_vq, criterion, scheduler, epochs=1000, resize_dim=(84, 84)
):
    model.train()

    if hasattr(model.vq, "initialize_codebook_with_kmeans"):
        print("Initializing codebook with KMeans...")
        with torch.no_grad():
            for frames, actions, _ in dataloader:
                frames = frames.to("cuda")
                patches = model.extract_patches(frames)  # Extract patches
                flat_patches = patches.reshape(-1, patches.size(-1))  # Flatten for initialization
                model.vq.initialize_codebook_with_kmeans(flat_patches)

    for epoch in range(epochs):
        total_loss = 0
        for i, (frames, actions, next_frames) in enumerate(dataloader):
            frames, actions, next_frames = frames.to("cuda"), actions.to("cuda").long(), next_frames.to("cuda")



            

            # Crop and resize frames
            # frames = frames[:, :, :, 20:-20, :]  # Remove 20 pixels from the top and bottom
            # next_frames = next_frames[:, :, :, 20:-20, :]  # Remove 20 pixels from the top and bottom

            # Ensure frames are reshaped for F.interpolate
            # batch_size, seq_len, channels, cropped_height, cropped_width = frames.size()
            # frames = frames.view(-1, channels, cropped_height, cropped_width)  # [B*T, C, H, W]
            # next_frames = next_frames.view(-1, channels, cropped_height, cropped_width)

            # # Resize frames to the desired dimension
            # frames = F.interpolate(frames, size=resize_dim, mode="bilinear", align_corners=False)  # [B*T, C, H', W']
            # next_frames = F.interpolate(next_frames, size=resize_dim, mode="bilinear", align_corners=False)

            # # Reshape back to original dimensions
            # frames = frames.view(batch_size, seq_len, channels, *resize_dim)  # [B, T, C, H', W']
            # next_frames = next_frames.view(batch_size, seq_len, channels, *resize_dim)

            # Zero gradients
            optimizer_task.zero_grad()
            optimizer_vq.zero_grad()

            # Forward pass
            preds, vq_loss, indices = model(frames, actions, targets=next_frames)

            # Losses
            task_loss = criterion(preds, next_frames)
            total_loss = task_loss + vq_loss

            # Backward pass
            vq_loss.backward(retain_graph=True)
            optimizer_vq.step()

            task_loss.backward()
            optimizer_task.step()

            # Replace dead codes
            if i % 15 == 0:
                print("Replacing dead codes...")
                model.vq.replace_dead_codes(usage_threshold=10)

            # Save frames, predictions, and reconstructions every 20 batches
            if i % 20 == 0:
                for batch_idx in range(min(5, frames.size(0))):  # Save up to 5 examples per batch
                    for frame_idx in range(min(30, frames.size(1))):  # Save up to 30 frames per sequence
                        os.makedirs('frames_ste', exist_ok=True)
                        os.makedirs('preds_ste', exist_ok=True)
                        
                        # Save cropped and resized frames
                        original_frame = frames[batch_idx][frame_idx][0].cpu().numpy() * 255
                        cv2.imwrite(f'frames_ste/{batch_idx}_{frame_idx}.png', original_frame)
                        
                        # Save cropped and resized predicted frames
                        predicted_frame = preds[batch_idx][frame_idx][0].detach().cpu().numpy() * 255
                        cv2.imwrite(f'preds_ste/{batch_idx}_{frame_idx}.png', predicted_frame)

            # Print batch loss
           
            if len(torch.unique(indices)) < 5:
                with torch.no_grad():
                    print("Reinitializing codebook with KMeans...")
                    patches = model.extract_patches(frames)
                    flat_patches = patches.reshape(-1, patches.size(-1))
                    model.vq.initialize_codebook_with_kmeans(flat_patches)

            print(
                f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(dataloader)}, "
                f"Task Loss: {task_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}, "
                f"Unique Indices: {torch.unique(indices)}"
            )


        if epoch % 10 == 0:
            scheduler.step()

        # Print epoch loss
        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss.item():.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]}")

        # Save model checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Atari Model with Custom Model Name in WandB")
    
    # Model hyperparameters
    parser.add_argument('--max_seq_len', type=int, default=30, help="Maximum sequence length")
    parser.add_argument('--embed_dim', type=int, default=16, help="Embedding dimension")
    parser.add_argument('--nhead', type=int, default=4, help="Number of attention heads")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of transformer layers")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--num_embeddings', type=int, default=64, help="Number of embeddings for VQ")

    # Logging and experiment tracking
    parser.add_argument('--log', action='store_true', help="Enable logging with WandB")
    parser.add_argument('--wandb_project', type=str, default="atari-xrl", help="WandB project name")
    parser.add_argument('--wandb_entity', type=str, default="mail-rishav9", help="WandB entity name")
    parser.add_argument('--wandb_run_name', type=str, default="xrl_atari", help="Base name for WandB run")
    parser.add_argument('--patch_size', type=int, default=4, help="Patch size for image transformer")

    return parser.parse_args()



if __name__=='__main__':
    image_size = (84, 84)
    action_dim = 18
    
    # Parse arguments
    env_name = 'SeaquestNoFrameskip-v4'
    
    args = parse_args()


    max_seq_len = args.max_seq_len  # Sequence length
    embed_dim = args.embed_dim  # Embedding dimension
    nhead = args.nhead  # Number of attention heads
    num_layers = args.num_layers  # Number of transformer layers
    batch_size = args.batch_size  # Batch size
    num_embeddings = args.num_embeddings  # Number of VQ-VAE embeddings
    patch_size = args.patch_size  # Patch size for image transformer

    # Dataset and DataLoader
    dataset = AtariGrayscaleDataset(env_name, max_seq_len=max_seq_len, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = CausalGrayscaleImageTransformerWithVQDT(action_dim, embed_dim, nhead, num_layers, num_embeddings, image_size=image_size, patch_size=patch_size).to("cuda")
    lr_task = 1e-4  # Learning rate for the encoder-decoder
    lr_vq = 1e-4    # Learning rate for the vector quantizer

    # Separate parameters for the task and VQ
    task_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.reconstruct.parameters())
    vq_params = list(model.vq.parameters())

    # Initialize optimizers
    optimizer_task = torch.optim.Adam(task_params, lr=lr_task)
    optimizer_vq = torch.optim.Adam(vq_params, lr=lr_vq)

    scheduler = StepLR(optimizer_task, step_size=8, gamma=0.1)
    criterion = nn.MSELoss()

    run_name = (
            f"{args.wandb_run_name}_{model.__class__.__name__}_"
            f"Embed{args.embed_dim}_Layers{args.num_layers}_Heads{args.nhead}_"
            f"SeqLen{args.max_seq_len}_Batch{args.batch_size}_Embeddings{args.num_embeddings}_"
            f"patch_size{args.patch_size}_"
            f"Run{random.randint(1, 1000)}"
        )
        
    if args.log:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)
        )

    

    # Train the model
    train_model_patch_crop(model, dataloader, optimizer_task, optimizer_vq, criterion, scheduler, epochs=1000)
    torch.save(model.state_dict(), 'model_atari_ste.pth')