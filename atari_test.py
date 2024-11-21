import d4rl_atari  # Ensure you have the D4RL Atari package installed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils import OfflineEnvAtari
import cv2
import os
import random
import wandb






class AtariGrayscaleDataset(Dataset):
    def __init__(self, env_name, max_seq_len=50, image_size=(84, 84)):
        # self.env = d4rl_atari.make(env_name)
        self.data = OfflineEnvAtari(stack=False, path='/home/rishav/scratch/d4rl_dataset/Seaquest/1/10').get_dataset()
        self.max_seq_len = max_seq_len
        self.image_size = image_size

        # Extract relevant data
        self.frames = self.data['observations']  # Grayscale image frames
        self.actions = self.data['actions']
        self.terminals = self.data['terminals']

        # Split trajectories
        self.trajectories = self._split_trajectories()

    def _split_trajectories(self):
        """
        Splits the dataset into trajectories based on terminal flags.
        """
        trajectories = []
        trajectory = {'frames': [], 'actions': []}

        for i in range(len(self.frames)):
            trajectory['frames'].append(self.frames[i])
            trajectory['actions'].append(self.actions[i])

            if self.terminals[i]:  # End of trajectory
                if len(trajectory['frames']) > self.max_seq_len:  # Ignore trajectories with single frames
                    trajectories.append(trajectory)
                trajectory = {'frames': [], 'actions': []}

        # if trajectory['frames']:  # Handle leftover trajectory
        #     trajectories.append(trajectory)

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
        frame_seq = torch.tensor(frame_seq, dtype=torch.float32) / 255.0  # Add channel dimension
        next_frame_seq = torch.tensor(next_frame_seq, dtype=torch.float32)/ 255.0

        return frame_seq, torch.tensor(action_seq, dtype=torch.float32), next_frame_seq






import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2SeqTransformer(nn.Module):
    def __init__(self, action_dim, embed_dim, nhead, num_encoder_layers, num_decoder_layers, max_seq_len, patch_size=7, image_size=(84, 84)):
        super(Seq2SeqTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches_per_frame = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        # Patch Embedding for Image States
        self.patch_embedding = nn.Linear(patch_size * patch_size, embed_dim)

        # Action Embedding
        self.action_embedding = nn.Embedding(action_dim, embed_dim)

        # Positional Embedding
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Reconstruction Layer (Patch to Image)
        self.reconstruct = nn.Linear(embed_dim, patch_size * patch_size)  

    def generate_causal_mask(self, seq_len, device):
        """
        Generate a causal mask to prevent attending to future tokens.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask.to(device)

    def extract_patches(self, frames):
        """
        Extract non-overlapping patches from image frames.
        """
        batch_size, seq_len, _, height, width = frames.size()
        patches = F.unfold(
            frames.view(batch_size * seq_len, 1, height, width),  # Combine batch and time dimensions
            kernel_size=self.patch_size,
            stride=self.patch_size
        )  # [B*T, patch_dim, num_patches]
        patches = patches.transpose(1, 2)  # [B*T, num_patches, patch_dim]
        return patches.view(batch_size, seq_len, self.num_patches_per_frame, -1)  # [B, T, P, patch_dim]

    def forward(self, states, actions, targets=None):
        """
        Forward pass for seq2seq model.
        states: [B, T, 1, H, W]  (image states)
        actions: [B, T]          (discrete action indices)
        targets: [B, T, 1, H, W] (target image states)
        """
        batch_size, seq_len, _, _, _ = states.size()

        # Extract patches from images
        patches = self.extract_patches(states)  # [B, T, P, patch_dim]
        state_embeds = self.patch_embedding(patches)  # [B, T, P, embed_dim]

        # Action Embeddings
        action_embeds = self.action_embedding(actions).unsqueeze(2).expand(-1, -1, self.num_patches_per_frame, -1)  # [B, T, P, embed_dim]

        # Positional Embeddings
        positions = torch.arange(seq_len).to(states.device)
        position_embeds = self.positional_embedding(positions).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.num_patches_per_frame, -1)  # [B, T, P, embed_dim]

        # Combine embeddings
        encoder_input = state_embeds + action_embeds + position_embeds  # [B, T, P, embed_dim]
        encoder_input = encoder_input.view(batch_size, seq_len * self.num_patches_per_frame, -1)  # Flatten patches

        # Generate causal mask for the encoder
        causal_mask = self.generate_causal_mask(encoder_input.size(1), states.device)

        # Transformer Encoder
        encoder_output = self.encoder(encoder_input, mask=causal_mask)  # [B, T*P, embed_dim]

        # Prepare decoder inputs
        if targets is not None:  # Training (teacher forcing)
            target_patches = self.extract_patches(targets)  # [B, T, P, patch_dim]
            decoder_input = self.patch_embedding(target_patches).view(batch_size, seq_len * self.num_patches_per_frame, -1)  # [B, T*P, embed_dim]
        else:  # Inference
            decoder_input = encoder_output[:, :-1, :]  # Use all except the last token

        tgt_seq_len = decoder_input.size(1)
        decoder_mask = self.generate_causal_mask(tgt_seq_len, states.device)
        decoder_output = self.decoder(decoder_input, encoder_output, tgt_mask=decoder_mask)  # [B, T*P, embed_dim]
        recon_patches = decoder_output.view(batch_size, seq_len, self.num_patches_per_frame, -1)   # [B, T, P, embed_dim]
        # print(recon_patches.view(batch_size * seq_len, -1).shape)
        recon_patches = recon_patches.view(batch_size * seq_len, self.num_patches_per_frame, -1)  # [B*T, P, embed_dim]
        recon_frames = self.reconstruct(recon_patches)  # Flatten for Linear layer
        recon_frames = recon_frames.view(batch_size, seq_len, 1, *self.image_size)  # [B, T, 1, H, W]

        return recon_frames



import torch.nn.functional as F

class CausalGrayscaleImageTransformerPatchTFTokens(nn.Module):
    def __init__(self, action_dim, embed_dim, nhead, num_layers, image_size=(84, 84), patch_size=7):
        super(CausalGrayscaleImageTransformerPatchTFTokens, self).__init__()
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

        # Prepare decoder input
        if targets is not None:
            # Teacher forcing: Use ground truth patches
            target_patches = self.extract_patches(targets)  # [B, T, P, patch_size*patch_size]
            decoder_input = self.patch_embedding(target_patches).view(
                batch_size * seq_len, self.num_patches_per_frame, self.embed_dim
            )  # [B*T, P, embed_dim]
        else:
            # Inference: Use EOS token as input
            decoder_input = self.eos_token.expand(batch_size, 1, self.embed_dim)  # [B, 1, embed_dim]

        # Generate causal mask for the decoder
        tgt_mask = self.generate_causal_mask(decoder_input.size(1), frames.device)

        # Decoder output
        decoder_output = self.decoder(decoder_input, encoder_output, tgt_mask=tgt_mask)  # [B*T, P, embed_dim]

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
        return next_frames




class CausalGrayscaleImageTransformerPatchTF(nn.Module):
    def __init__(self, action_dim, embed_dim, nhead, num_layers, image_size=(84, 84), patch_size=7):
        super(CausalGrayscaleImageTransformerPatchTF, self).__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        # Linear layer to embed patches
        self.patch_embedding = nn.Linear(patch_size * patch_size, embed_dim)

        # Action embedding
        self.action_embedding = nn.Embedding(action_dim, embed_dim)

        # Positional embedding
        self.positional_embedding = nn.Embedding(self.num_patches_per_frame, embed_dim)

        # Transformer encoder-decoder
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )

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

        # Extract patches from each frame
        patches = self.extract_patches(frames)  # [B, T, P, patch_size*patch_size]

        # Embed patches
        patch_embeds = self.patch_embedding(patches)  # [B, T, P, embed_dim]

        # Embed actions
        action_embeds = self.action_embedding(actions)  # [B, T, embed_dim]
        action_embeds = action_embeds.unsqueeze(2).expand(-1, -1, self.num_patches_per_frame, -1)  # [B, T, P, embed_dim]

        # Add positional embeddings
        positions = torch.arange(self.num_patches_per_frame).to(frames.device)  # [P]
        position_embeds = self.positional_embedding(positions)  # [P, embed_dim]
        position_embeds = position_embeds.unsqueeze(0).unsqueeze(0)  # [1, 1, P, embed_dim]

        # Combine embeddings for the encoder input
        encoder_input = patch_embeds + action_embeds + position_embeds  # [B, T, P, embed_dim]
        encoder_input = encoder_input.view(batch_size * seq_len, self.num_patches_per_frame, self.embed_dim)  # Flatten sequence

        # Generate causal mask for the encoder
        causal_mask = self.generate_causal_mask(self.num_patches_per_frame, frames.device)

        # Encoder output
        encoder_output = self.transformer.encoder(encoder_input, mask=causal_mask)  # [B*T, P, embed_dim]

        # Prepare decoder input (teacher forcing or inference)
        if targets is not None:
            # Teacher forcing: Use ground truth targets
            target_patches = self.extract_patches(targets)  # [B, T, P, patch_size*patch_size]
            decoder_input = self.patch_embedding(target_patches).view(
                batch_size * seq_len, self.num_patches_per_frame, self.embed_dim
            )  # [B*T, P, embed_dim]
        else:
            # Inference: Use encoder output
            decoder_input = encoder_output

        # Transformer decoder
        decoder_output = self.transformer.decoder(decoder_input, encoder_output)  # [B*T, P, embed_dim]

        # Reshape decoder output for spatial reconstruction
        decoder_output = decoder_output.view(
            batch_size, seq_len, self.num_patches_per_frame, self.embed_dim
        ).permute(0, 1, 3, 2)  # [B, T, embed_dim, num_patches]
        decoder_output = decoder_output.view(
            batch_size * seq_len, self.embed_dim, self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size
        )  # [B*T, embed_dim, H_patches, W_patches]

        # Reconstruct full frames
        next_frames = self.reconstruct(decoder_output)  # [B*T, 1, H, W]
        next_frames = next_frames.view(batch_size, seq_len, 1, *self.image_size)  # [B, T, 1, H, W]
        return next_frames


class CausalGrayscaleImageTransformerPatch(nn.Module):
    def __init__(self, action_dim, embed_dim, nhead, num_layers, max_seq_len, image_size=(84, 84), patch_size=7):
        super(CausalGrayscaleImageTransformerPatch, self).__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        # Linear layer to embed patches
        self.patch_embedding = nn.Linear(patch_size * patch_size, embed_dim)

        # Action embedding
        self.action_embedding = nn.Embedding(action_dim, embed_dim)

        # Positional embedding
        self.positional_embedding = nn.Embedding(self.num_patches_per_frame, embed_dim)

        # Transformer encoder with causal masking
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )

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

    def forward(self, frames, actions):
        batch_size, seq_len, _, _, _ = frames.size()

        # Extract patches from each frame
        patches = self.extract_patches(frames)  # [B, T, P, patch_size*patch_size]

        # Embed patches
        patch_embeds = self.patch_embedding(patches)  # [B, T, P, embed_dim]

        # Embed actions
        
        action_embeds = self.action_embedding(actions)  # [B, T, embed_dim]
        action_embeds = action_embeds.unsqueeze(2).expand(-1, -1, self.num_patches_per_frame, -1)  # [B, T, P, embed_dim]

        # Add positional embeddings
        positions = torch.arange(self.num_patches_per_frame).to(frames.device)  # [P]
        position_embeds = self.positional_embedding(positions)  # [P, embed_dim]
        position_embeds = position_embeds.unsqueeze(0).unsqueeze(0)  # [1, 1, P, embed_dim]

        # Combine embeddings
        transformer_input = patch_embeds + action_embeds + position_embeds  # [B, T, P, embed_dim]
        transformer_input = transformer_input.view(batch_size * seq_len, self.num_patches_per_frame, self.embed_dim)  # Flatten sequence

        # Generate causal mask
        causal_mask = self.generate_causal_mask(self.num_patches_per_frame, frames.device)

        # Transformer processing with causal mask
        transformer_out = self.transformer(transformer_input, transformer_input, tgt_mask=causal_mask)  # [B*T, P, embed_dim]

        # Reshape transformer output for spatial reconstruction
        transformer_out = transformer_out.view(
            batch_size, seq_len, self.num_patches_per_frame, self.embed_dim
        ).permute(0, 1, 3, 2)  # [B, T, embed_dim, num_patches]
        transformer_out = transformer_out.view(
            batch_size * seq_len, self.embed_dim, self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size
        )  # [B*T, embed_dim, H_patches, W_patches]

        # Reconstruct full frames
        next_frames = self.reconstruct(transformer_out)  # [B*T, 1, H, W]
        next_frames = next_frames.view(batch_size, seq_len, 1, *self.image_size)  # [B, T, C, H, W]
        return next_frames




class CausalGrayscaleImageTransformer(nn.Module):
    def __init__(self, action_dim, embed_dim, nhead, num_layers, max_seq_len, image_size=(84, 84)):
        super(CausalGrayscaleImageTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size

        # CNN-based image embedding for grayscale images
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, embed_dim)
        )

        # Action embedding
        self.action_embedding = nn.Embedding(action_dim, embed_dim)

        # Positional embedding
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer encoder with causal masking
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )

        # Reconstruction layers: convs and upsampling
        self.reconstruct = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Ensure output is in range [0, 1] for image reconstruction
            nn.Upsample(size=image_size, mode='bilinear', align_corners=False)
        )

    def generate_causal_mask(self, seq_len, device):
        """
        Generate a causal mask to ensure predictions depend only on past and current inputs.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask.to(device)

    def forward(self, frames, actions):
        batch_size, seq_len, _, _, _ = frames.size()

        # CNN embedding for frames
        frame_embeds = torch.stack([self.cnn(frames[:, t]) for t in range(seq_len)], dim=1)

        # Action embedding 
        action_embeds = self.action_embedding(actions)

        # Add positional embeddings
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len).to(frames.device)
        position_embeds = self.positional_embedding(positions)

        # Combine embeddings
        transformer_input = frame_embeds + action_embeds + position_embeds

        # Generate causal mask
        causal_mask = self.generate_causal_mask(seq_len, frames.device)

        # Transformer processing with causal mask
        transformer_out = self.transformer(transformer_input, transformer_input, tgt_mask=causal_mask)

        # Reconstruction
        transformer_out = transformer_out.view(batch_size * seq_len, -1, 1, 1)  # Reshape to [B * T, D, 1, 1]
        next_frames = self.reconstruct(transformer_out)  # Apply reconstruction layers
        next_frames = next_frames.view(batch_size, seq_len, 1, *self.image_size)  # Reshape to [B, T, C, H, W]
        return next_frames



def train_model(model, dataloader, optimizer, criterion, epochs=1000):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (frames, actions, next_frames) in enumerate(dataloader):
            frames, actions, next_frames = frames.to("cuda"), actions.to("cuda").long(), next_frames.to("cuda")
            optimizer.zero_grad()
            preds = model(frames, actions)
            loss = criterion(preds, next_frames)

            if args.log:
                wandb.log({"loss": loss.item()})
            

            #save frames and preds every 50 batch
            if i % 20 == 0:
                for i in range(5):
                    for j in range(30):
                        #mkdir if not exists

                        os.makedirs('frames1', exist_ok=True)
                        os.makedirs('preds1', exist_ok=True)
                        cv2.imwrite(f'frames1/{i}_{j}.png', frames[i][j][0].cpu().numpy()*255)
                        cv2.imwrite(f'preds1/{i}_{j}.png', preds[i][j][0].detach().cpu().numpy()*255)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")



from models.vector_quantizers import EMAVectorQuantizer
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

        # Quantize encoder output
        quantized, vq_loss, encoding_indices = self.vq(encoder_output)  # Quantized output and VQ loss

        # Prepare decoder input
        if targets is not None:
            # Teacher forcing: Use ground truth patches
            target_patches = self.extract_patches(targets)  # [B, T, P, patch_size*patch_size]
            decoder_input = self.patch_embedding(target_patches).view(
                batch_size * seq_len, self.num_patches_per_frame, self.embed_dim
            )  # [B*T, P, embed_dim]
        else:
            # Inference: Use CLS tokens
            decoder_input = quantized

        # Generate causal mask for the decoder
        tgt_mask = self.generate_causal_mask(decoder_input.size(1), frames.device)

        # Decoder output
        decoder_output = self.decoder(decoder_input, quantized, tgt_mask=tgt_mask)  # [B*T, P, embed_dim]

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






def train_model_patch_crop(model, dataloader, optimizer, criterion, scheduler, epochs=1000, resize_dim=(84, 84)):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        total_loss = 0
        for i, (frames, actions, next_frames) in enumerate(dataloader):
            # Move data to the appropriate device
            frames, actions, next_frames = frames.to("cuda"), actions.to("cuda").long(), next_frames.to("cuda")
            
            # Crop 20 pixels from top and bottom
            frames = frames[:, :, :, 20:-20, :]  # Crop top and bottom for input frames
            next_frames = next_frames[:, :, :, 20:-20, :]  # Crop top and bottom for target frames

            # Flatten batch and sequence dimensions for resizing
            batch_size, seq_len, _, height, width = frames.size()
            frames = frames.view(batch_size * seq_len, 1, height, width)  # [B*T, 1, H, W]
            next_frames = next_frames.view(batch_size * seq_len, 1, height, width)

            # Resize frames to the desired dimension
            frames = F.interpolate(frames, size=resize_dim, mode='bilinear', align_corners=False)  # [B*T, 1, H', W']
            next_frames = F.interpolate(next_frames, size=resize_dim, mode='bilinear', align_corners=False)

            # Reshape back to original dimensions
            frames = frames.view(batch_size, seq_len, 1, *resize_dim)  # [B, T, 1, H', W']
            next_frames = next_frames.view(batch_size, seq_len, 1, *resize_dim)

            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass with teacher forcing
            if isinstance(model, CausalGrayscaleImageTransformerWithVQ):
                preds, vq_loss, indices = model(frames, actions, targets=next_frames)
                loss = criterion(preds, next_frames) + vq_loss
            else:
                preds = model(frames, actions, targets=next_frames)
                loss = criterion(preds, next_frames)
            
            # Compute loss
            
            
            # Log loss if required
            if args.log:
                import wandb
                if isinstance(model, CausalGrayscaleImageTransformerWithVQ):
                    wandb.log({"loss": loss.item(), "vq_loss": vq_loss.item(), "unique_indices": len(torch.unique(indices))})
                else:
                    wandb.log({"loss": loss.item()})
            
            # Save frames and predictions every 20 batches
            if i % 20 == 0:
                for batch_idx in range(min(5, frames.size(0))):  # Save up to 5 examples per batch
                    for frame_idx in range(min(30, frames.size(1))):  # Save up to 30 frames per sequence
                        os.makedirs('frames5', exist_ok=True)
                        os.makedirs('preds5', exist_ok=True)
                        
                        # Save cropped and resized frames
                        original_frame = frames[batch_idx][frame_idx][0].cpu().numpy() * 255
                        cv2.imwrite(f'frames5/{batch_idx}_{frame_idx}.png', original_frame)
                        
                        # Save cropped and resized predicted frames
                        predicted_frame = preds[batch_idx][frame_idx][0].detach().cpu().numpy() * 255
                        cv2.imwrite(f'preds5/{batch_idx}_{frame_idx}.png', predicted_frame)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Print batch loss
            if isinstance(model, CausalGrayscaleImageTransformerWithVQ):
                print(f"Batch {i+1}, Loss: {loss.item()} VQ Loss: {vq_loss.item()} unique indices: {torch.unique(indices)}")
            else:
                print(f"Batch {i+1}, Loss: {loss.item()}")
        
        # Adjust learning rate using the scheduler
        scheduler.step()
        
        # Print epoch loss
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]}")

from torch.optim.lr_scheduler import StepLR


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Train Atari Model with Custom Model Name in WandB")
    
    # Model hyperparameters
    parser.add_argument('--max_seq_len', type=int, default=30, help="Maximum sequence length")
    parser.add_argument('--embed_dim', type=int, default=16, help="Embedding dimension")
    parser.add_argument('--nhead', type=int, default=4, help="Number of attention heads")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of transformer layers")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--num_embeddings', type=int, default=64, help="Number of embeddings for VQ")

    # Logging and experiment tracking
    parser.add_argument('--log', action='store_true', help="Enable logging with WandB")
    parser.add_argument('--wandb_project', type=str, default="atari-xrl", help="WandB project name")
    parser.add_argument('--wandb_entity', type=str, default="mail-rishav9", help="WandB entity name")
    parser.add_argument('--wandb_run_name', type=str, default="xrl_atari", help="Base name for WandB run")

    return parser.parse_args()


# Example Usage
if __name__ == "__main__":
    # Dataset parameters
    env_name = "breakout-expert-v0"

    image_size = (84, 84)
    action_dim = 18
    
    # Parse arguments
    
    args = parse_args()


    max_seq_len = args.max_seq_len  # Sequence length
    embed_dim = args.embed_dim  # Embedding dimension
    nhead = args.nhead  # Number of attention heads
    num_layers = args.num_layers  # Number of transformer layers
    batch_size = args.batch_size  # Batch size
    num_embeddings = args.num_embeddings  # Number of VQ-VAE embeddings

    # Dataset and DataLoader
    dataset = AtariGrayscaleDataset(env_name, max_seq_len=max_seq_len, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = CausalGrayscaleImageTransformerWithVQ(action_dim, embed_dim, nhead, num_layers, num_embeddings, image_size=image_size, patch_size=4).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    criterion = nn.MSELoss()

    run_name = (
            f"{args.wandb_run_name}_{model.__class__.__name__}_"
            f"Embed{args.embed_dim}_Layers{args.num_layers}_Heads{args.nhead}_"
            f"SeqLen{args.max_seq_len}_Batch{args.batch_size}_Embeddings{args.num_embeddings}_"
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
    train_model_patch_crop(model, dataloader, optimizer, criterion, scheduler, epochs=1000)
