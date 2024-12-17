import torch
import torch.nn as nn
import wandb
import argparse
from torch.utils.tensorboard import SummaryWriter
import cv2
import os


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
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument('--load_checkpoint', type=str, default=None, help="Path to load model checkpoint")

    return parser.parse_args()

args = parse_args()

if args.log:
    wandb.init(
        project="atari-seq2seq-transformer",
        name="seq2seq-training",
        entity="mail-rishav9",
        config={
            "learning_rate": 1e-4,
            "batch_size": 4,
            "epochs": 2000,
            "embed_dim": 128,
            "num_heads": 8,
            "num_layers": 6,
            "max_seq_len": 30,
        },
        sync_tensorboard=True,  # Synchronize TensorBoard logs with wandb
    )

writer = SummaryWriter(log_dir="tensorboard_logs")



import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchBasedAtariTransformerWithOneActionPerState(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_heads, num_layers, action_dim, max_seq_len=1024):
        super(PatchBasedAtariTransformerWithOneActionPerState, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_size * patch_size, embed_dim)

        # Action embedding (one per frame)
        self.action_embedding = nn.Embedding(action_dim, embed_dim)

        # Positional embeddings (for patches and temporal sequence)
        self.patch_pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        # Transformer encoder-decoder architecture
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=4 * embed_dim, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=4 * embed_dim, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # Reconstruction head
        self.reconstruct = nn.Sequential(
            nn.Linear(self.embed_dim, self.patch_size * self.patch_size),  # Map embeddings to patch pixels
            nn.Unflatten(2, (self.patch_size, self.patch_size)),  # Reshape each patch
            nn.Unflatten(1, (int(self.img_size[0] / self.patch_size), int(self.img_size[1] / self.patch_size))),
            # nn.Flatten(1, 2),

            # nn.ConvTranspose2d(
            #     in_channels=1,
            #     out_channels=1,
            #     kernel_size=self.patch_size,
            #     stride=self.patch_size,
            #     padding=0,
            # )  # Combine patches into full frames
        )


    def generate_causal_mask(self, seq_len, device):
        """
        Generate a causal mask to ensure predictions depend only on past and current inputs.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask.to(device)

    def extract_patches(self, frames):
        """
        Extract non-overlapping patches from input frames.
        """
        B, T, _, H, W = frames.shape
        patches = F.unfold(frames.view(B * T, 1, H, W), kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2)  # [B*T, num_patches, patch_size * patch_size]
        return patches.view(B, T, self.num_patches, -1)  # [B, T, num_patches, patch_size * patch_size]

    def forward(self, states, actions, target_states=None):
        B, T, _, H, W = states.shape

        # Patch embedding and temporal processing
        state_patches = self.extract_patches(states)
        state_embeddings = self.patch_embedding(state_patches)
        state_embeddings += self.patch_pos_embedding

        # Flatten and add positional embeddings
        state_embeddings = state_embeddings.view(B, T * self.num_patches, -1)
        temporal_pos_embeddings = self.temporal_pos_embedding[:, :T, :].repeat_interleave(self.num_patches, dim=1)
        state_embeddings += temporal_pos_embeddings

        # Add action embeddings
        action_embeddings = self.action_embedding(actions).unsqueeze(2).repeat(1, 1, self.num_patches, 1)
        action_embeddings = action_embeddings.view(B, T * self.num_patches, -1)
        state_embeddings += action_embeddings

        # Transformer encoder
        causal_mask = self.generate_causal_mask(state_embeddings.size(1), states.device)
        encoder_output = self.encoder(state_embeddings, mask=causal_mask)

        # Transformer decoder
        if target_states is not None:
            target_patches = self.extract_patches(target_states)
            target_embeddings = self.patch_embedding(target_patches) + self.patch_pos_embedding
            target_embeddings = target_embeddings.view(B, T * self.num_patches, -1)
            target_embeddings = target_embeddings
            target_embeddings += temporal_pos_embeddings
            tgt_mask = self.generate_causal_mask(target_embeddings.size(1), states.device)
            outputs = self.decoder(target_embeddings, encoder_output, tgt_mask=tgt_mask)
        else:
            outputs = encoder_output

        # Reconstruction
        predicted_patches = outputs.view(B, T, self.num_patches, self.embed_dim)

# Reconstruct patches into frames
        predicted_patches = predicted_patches.view(-1, self.num_patches, self.embed_dim)  # Flatten B*T for processing
        # predicted_patches = predicted_patches.permute(0, 2, 1)  # [B*T, embed_dim, num_patches]
        predicted_frames = self.reconstruct(predicted_patches)
        predicted_frames = predicted_frames.view(B, T, 1, H, W)   # [B, T, 1, H, W]
        return predicted_frames





class AtariSeq2SeqTransformer(nn.Module):
    def __init__(self, img_size, embed_dim, num_heads, num_layers, action_dim, max_seq_len=1024):
        super(AtariSeq2SeqTransformer, self).__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim

        # State encoder: extracts spatial embeddings from frames
        self.state_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
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

        # Frame reconstruction head
        self.reconstruct = nn.Sequential(
            nn.Linear(embed_dim, 3136),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4, padding=0), nn.Sigmoid()  # Normalize output
        )

    def generate_causal_mask(self, seq_len, device):
        """
        Generate a causal mask to ensure predictions depend only on past and current inputs.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask.to(device)

    def forward(self, states, actions, target_states=None):
        """
        states: [B, T, 1, H, W] - input grayscale frames
        actions: [B, T] - input actions
        target_states: [B, T, 1, H, W] - ground truth future frames
        """
        B, T, _, H, W = states.shape

        # Encode states into embeddings
        state_embeddings = self.state_encoder(states.view(-1, 1, H, W))  # [B*T, embed_dim]
        state_embeddings = state_embeddings.view(B, T, -1)  # [B, T, embed_dim]

        # Embed actions
        action_embeddings = self.action_embedding(actions)  # [B, T, embed_dim]

        # Combine state and action embeddings
        seq_embeddings = state_embeddings + action_embeddings  # Element-wise addition
        seq_embeddings += self.positional_embedding[:, :T, :]  # Add positional embeddings

        # Generate causal mask for the encoder
        causal_mask = self.generate_causal_mask(seq_embeddings.size(1), states.device)

        # Forward through encoder
        encoder_output = self.encoder(seq_embeddings, mask=causal_mask)  # [B, T, embed_dim]

        # Prepare target input for the decoder
        if target_states is not None:
            # Teacher forcing: use shifted target frames
            target_embeddings = self.state_encoder(target_states.view(-1, 1, H, W))
            target_embeddings = target_embeddings.view(B, T, -1)  # [B, T, embed_dim]
            target_embeddings = target_embeddings[:, :-1, :]  # Shift by one time step
            target_embeddings = target_embeddings + self.positional_embedding[:, :target_embeddings.size(1), :]


            # Causal mask for the decoder
            tgt_mask = self.generate_causal_mask(target_embeddings.size(1), states.device)

            # Decoder forward pass
            outputs = self.decoder(target_embeddings, encoder_output, tgt_mask=tgt_mask)  # [B, T-1, embed_dim]
        else:
            # During inference, autoregressively predict frames
            outputs = encoder_output

        # Reconstruct frames from Transformer output embeddings
        frame_embeddings = outputs  # [B, T-1, embed_dim] (or [B, T, embed_dim] during inference)
        predicted_frames = self.reconstruct(frame_embeddings.reshape(-1, self.embed_dim))  # [B*(T-1), 1, H, W]
        predicted_frames = predicted_frames.view(B, -1, 1, H, W)  # [B, T-1, 1, H, W]

        return predicted_frames

from utils import OfflineEnvAtari, AtariGrayscaleDataset


model = AtariSeq2SeqTransformer(
    img_size=(84, 84),
    embed_dim=128,
    num_heads=8,
    num_layers=6,
    action_dim=18,
).to("cuda")


if args.load_checkpoint is not None:
    model.load_state_dict(torch.load(args.load_checkpoint))
    print(f"Model loaded from {args.load_checkpoint}")

data = AtariGrayscaleDataset('Breakout-v0',max_seq_len=30)
dataloader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
criteria = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheculer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)


for epoch in range(2000):
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        states, actions, target_states = batch
        states = states.to("cuda")
        actions = actions.to("cuda").long()
        target_states = target_states.to("cuda")

        # Forward pass
        predicted_frames = model(states, actions, target_states)
        loss = criteria(predicted_frames, target_states[:,1:])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate batch loss
        epoch_loss += loss.item()

        # Logging loss for each batch
        print(f"Epoch [{epoch + 1}], Batch [{batch_idx + 1}], Loss: {loss.item():.4f}")
        writer.add_scalar("Loss/Batch", loss.item(), epoch * len(dataloader) + batch_idx)

        if batch_idx % 40 == 0:
            #save images
            for batch_idx in range(min(5, states.size(0))):  # Save up to 5 examples per batch
                    for frame_idx in range(min(29, states.size(1))):  # Save up to 30 frames per sequence
                        os.makedirs('frames_no_vq_p', exist_ok=True)
                        os.makedirs('preds_no_vq_p', exist_ok=True)
                        
                        # Save cropped and resized frames
                        original_frame = states[batch_idx][frame_idx][0].cpu().numpy() * 255
                        cv2.imwrite(f'frames_no_vq_p/{batch_idx}_{frame_idx}.png', original_frame)
                        
                        # Save cropped and resized predicted frames
                        predicted_frame = predicted_frames[batch_idx][frame_idx][0].detach().cpu().numpy() * 255
                        cv2.imwrite(f'preds_no_vq_p/{batch_idx}_{frame_idx}.png', predicted_frame)

    # Average epoch loss
    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}], Average Loss: {avg_epoch_loss:.4f}")

    # Log average loss for the epoch
    writer.add_scalar("Loss/Epoch", avg_epoch_loss, epoch + 1)

    # Save the model checkpoint every 50 epochs


    if (epoch + 1) % 50 == 0:
        #name of the model
        name = f"model_epoch_{model.__class__}_{epoch + 1}.pth"
        checkpoint_path = f"model_epoch_{name}_{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")
        scheculer.step()
        # wandb.save(checkpoint_path)

# Close TensorBoard writer
writer.close()
# wandb.finish()
        

# Initialize Model
# Initialize model



