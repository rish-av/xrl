import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import OfflineEnvAtari
import random
import math
import numpy as np
from models.vector_quantizers import EMAVectorQuantizer
import cv2
from utils import get_args_atari


args = get_args_atari()

if args.log:
    import wandb
    print("Logging to wandb")
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, entity=args.wandb_entity, config=args)
    wandb.config.update(args)


class AtariSequenceDataset(Dataset):
    def __init__(self, dataset, sequence_length=10, transform=None):
        self.sequence_length = sequence_length
        self.transform = transform
        self.trajectories = self.split_into_trajectories(dataset)

    def split_into_trajectories(self, dataset):
        observations = dataset['observations']
        actions = dataset['actions']
        terminals = dataset['terminals']
        trajectories = []
        start_idx = 0
        for idx in range(len(terminals)):
            if terminals[idx]:
                end_idx = idx + 1
                traj = {
                    'observations': observations[start_idx:end_idx],
                    'actions': actions[start_idx:end_idx]
                }
                if len(traj['observations']) > self.sequence_length:
                    trajectories.append(traj)
                start_idx = end_idx
        return trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        states = traj['observations']
        actions = traj['actions']

        # Ensure the trajectory is long enough for sampling
        if len(states) <= self.sequence_length:
            raise ValueError("Trajectory too short for the desired sequence length")

        # Select a random starting index within the trajectory
        start_idx = np.random.randint(0, len(states) - self.sequence_length)

        # Extract the sequence and the target sequence
        seq_states = states[start_idx:start_idx + self.sequence_length]
        seq_actions = actions[start_idx:start_idx + self.sequence_length]
        target_states = states[start_idx + 1:start_idx + self.sequence_length + 1]

        # Apply transform if specified
        if self.transform:
            seq_states = torch.stack([self.transform(state.transpose(1, 2, 0)) for state in seq_states])  # Shape: (seq_len, C, H, W)
            target_states = torch.stack([self.transform(state.transpose(1, 2, 0)) for state in target_states])  # Shape: (seq_len, C, H, W)
        else:
            seq_states = torch.from_numpy(seq_states)
            target_states = torch.from_numpy(target_states)

        seq_actions = torch.tensor(seq_actions).long()

        return seq_states, seq_actions, target_states

class D4RLSequenceDataset(Dataset):
    def __init__(self, env_name, seq_length, image_size=84, transform=None):
        self.dataset = OfflineEnvAtari(stack=False, path='/home/rishav/scratch/d4rl_dataset/Seaquest/1/10').get_dataset()
        self.seq_length = seq_length
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.sequences = self._extract_sequences()

    def _extract_sequences(self):
        sequences = []
        trajectory_start = 0

        for i in range(len(self.dataset['terminals'])):
            if self.dataset['terminals'][i] or i == len(self.dataset['terminals']) - 1:
                trajectory_end = i + 1
                trajectory = [
                    (self.dataset['observations'][j], self.dataset['actions'][j])
                    for j in range(trajectory_start, trajectory_end)
                ]
                trajectory_start = trajectory_end

                if len(trajectory) >= self.seq_length + 1:
                    for _ in range(5):
                        start_idx = random.randint(0, len(trajectory) - self.seq_length - 1)
                        sequence = trajectory[start_idx:start_idx + self.seq_length + 1]
                        sequences.append(sequence)

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        states = []
        actions = []
        target_states = []

        for i in range(self.seq_length):
            state, action = sequence[i]
            next_state, _ = sequence[i + 1]

            state_tensor = self.transform(state.transpose(1,2,0))  # Ensure shape is [C, H, W] after transform
            next_state_tensor = self.transform(next_state.transpose(1,2,0))

            states.append(state_tensor)
            actions.append(action)
            target_states.append(next_state_tensor)

        states = torch.stack(states)  # Shape: [seq_len, C, H, W]
        actions = torch.tensor(actions, dtype=torch.long)  # Shape: [seq_len]
        target_states = torch.stack(target_states)  # Shape: [seq_len, C, H, W]
        
        return states, actions, target_states  # Output with seq_len as the first dimension




class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Seq2SeqTransformer(nn.Module):
    def __init__(self, image_size, action_dim, embed_dim, n_heads, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super(Seq2SeqTransformer, self).__init__()
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(embed_dim)

        self.vqae = EMAVectorQuantizer(128, embed_dim, commitment_cost=1.0)
        
        # Image encoder to convert images to embeddings
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (image_size // 8) * (image_size // 8), embed_dim)
        )

        # Action embedding layer
        self.action_embedding = nn.Embedding(action_dim, embed_dim)

        # Transformer Encoder for (state, action) pairs
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer Decoder for future states
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Transpose Conv layers to decode embeddings back into images
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(embed_dim, 128 * (image_size // 8) * (image_size // 8)),
            nn.Unflatten(1, (128, image_size // 8, image_size // 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False)
        )

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, states, actions, tgt_states):
        batch_size, seq_len, C, H, W = states.size()
        
        # Generate causal masks for encoder and decoder
        encoder_mask = self.generate_square_subsequent_mask(seq_len).to(states.device)
        decoder_mask = self.generate_square_subsequent_mask(seq_len).to(states.device)
        
        # Encode the input states
        encoded_states = torch.stack([self.image_encoder(states[:, i]) for i in range(seq_len)], dim=1)  # [batch_size, seq_len, embed_dim]
        
        # Embed actions and combine with states
        encoded_actions = self.action_embedding(actions)  # [batch_size, seq_len, embed_dim]
        encoder_input = encoded_states + encoded_actions
        
        # Add positional encoding to encoder input
        encoder_input = self.positional_encoding(encoder_input)
        
        # Transformer Encoder with mask for causality
        eos_rep = self.encoder(encoder_input, mask=encoder_mask)  # [batch_size, seq_len, embed_dim]
        eos_rep = eos_rep[:, -1, :]  # End of sequence representation for the decoder
        
        # Quantize representation
        eos_rep, vq_loss, encoding_indices = self.vqae(eos_rep)
        
        # Prepare the target sequence embeddings for the decoder with teacher forcing
        # Shift the target states by one position to the right
        tgt_input = torch.zeros_like(tgt_states)  # Initialize with zeros for the <START> token
        tgt_input[:, 1:] = tgt_states[:, :-1]  # Shifted target states
        
        # Encode the shifted target states
        tgt_encoded_states = torch.stack([self.image_encoder(tgt_input[:, i]) for i in range(seq_len)], dim=1)  # [batch_size, seq_len, embed_dim]
        
        # Add positional encoding to target states
        tgt_encoded_states = self.positional_encoding(tgt_encoded_states)
        
        # Transformer Decoder with causal mask
        decoder_output = self.decoder(tgt_encoded_states, eos_rep.unsqueeze(1), tgt_mask=decoder_mask)  # [batch_size, seq_len, embed_dim]
        
        # Reconstruct images from the transformer decoder output
        reconstructed_states = torch.stack([self.reconstruction_decoder(decoder_output[:, i]) for i in range(seq_len)], dim=1)
        
        return reconstructed_states, vq_loss, encoding_indices  


# Define the loss function for image reconstruction
def reconstruction_loss(reconstructed_states, target_states):
    return nn.MSELoss()(reconstructed_states, target_states)

def recover_image(image):
    # Shift the values from [-0.5, 0.5] to [0, 1]
    image_normalized = image + 0.5  # Shift the range to [0, 1]
    # Scale to the range [0, 255]
    image_scaled = np.clip(image_normalized * 255, 0, 255).astype(np.uint8)  # Scale to [0, 255]
    return image_scaled


# Example training loop
import torch.nn.functional as F

def train_model(model, dataloader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for states, actions, target_states in dataloader:
            # Move data to GPU
            states, actions, target_states = (
                states.to('cuda').float(),
                actions.to('cuda'),
                target_states.to('cuda').float(),
            )
            top_crop = 18  # Pixels to crop from the top
            bottom_crop = 20  # Pixels to crop from the bottom
            states = states[:, :, :, top_crop:-bottom_crop, :]
            target_states = target_states[:, :, :, top_crop:-bottom_crop, :]
            
            states = F.interpolate(states.view(-1, *states.shape[2:]), size=(84, 84), mode='bilinear').view(*states.shape[:2], -1, 84, 84)
            target_states = F.interpolate(target_states.view(-1, *target_states.shape[2:]), size=(84, 84), mode='bilinear').view(*target_states.shape[:2], -1, 84, 84)
            
            optimizer.zero_grad()
            reconstructed_states, vq_loss, encoding_idx = model(states, actions, target_states)
            
            # Compute loss
            recon_loss = reconstruction_loss(reconstructed_states, target_states)
            loss = recon_loss

            if args.log:
                wandb.log({'reconstruction_loss': recon_loss.item(), 'vq_loss': vq_loss.item(), 'total_loss': loss.item(), "uniue_encoding_idx": torch.unique(encoding_idx)})

            loss.backward()
            optimizer.step()
            
            # Update total loss
            total_loss += loss.item()
            print(loss.item(), torch.unique(encoding_idx))
        
        # Log epoch loss
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader)}')

        if args.log:
            if epoch % 10 == 0:
                seq_0 = states[0,:]
                tar_0 = target_states[0,:]
                rec_0 = reconstructed_states[0,:]

                seq_len = seq_0.shape[0]
                for i in range(seq_len-1):

                    #log to wandb these images
                    # wandb.log({f'originals/original_{i}': [wandb.Image(recover_image(seq_0[i].permute(1,2,0).detach().cpu().numpy().squeeze()))]})
                    # wandb.log({f'reconstructed/reconstructed_{i}': [wandb.Image(recover_image(rec_0[i].permute(1,2,0).detach().cpu().numpy().squeeze()))]})
                    # wandb.log({f'targets/target_{i}': [wandb.Image(recover_image(tar_0[i].permute(1,2,0).detach().cpu().numpy().squeeze()))]})
                    cv2.imwrite(f'originals1/original_{i}.png', recover_image(seq_0[i].permute(1,2,0).detach().cpu().numpy().squeeze()))
                    cv2.imwrite(f'reconstructed1/reconstructed_{i}.png', recover_image(rec_0[i].permute(1,2,0).detach().cpu().numpy().squeeze()))
                    cv2.imwrite(f'targets1/target_{i}.png', recover_image(tar_0[i].permute(1,2,0).detach().cpu().numpy().squeeze()))
# Hyperparameters
image_size = 84  # Assuming images are 84x84
action_dim = 18  # Number of discrete actions in Atari
embed_dim = 128  # Embedding dimension
n_heads = 4  # Number of attention heads
num_encoder_layers = 4  # Number of encoder layers
num_decoder_layers = 4  # Number of decoder layers
dim_feedforward = 128  # Dimension of feedforward layers in transformer

# Instantiate the model
model = Seq2SeqTransformer(image_size, action_dim, embed_dim, n_heads, num_encoder_layers, num_decoder_layers, dim_feedforward).to('cuda')
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Assuming `dataloader` is a DataLoader that provides (states, actions, target_states)

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
dataset_org = OfflineEnvAtari(stack=True, path='/home/rishav/scratch/d4rl_dataset/Seaquest/1/10').get_dataset()
dataset = AtariSequenceDataset(dataset_org, 40, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(len(dataset))

iterator = iter(dataloader)

states, actions, target_states = next(iterator)
states = states.permute(1,0,2,3,4)
actions = actions.permute(1,0)
target_states = target_states.permute(1,0,2,3,4)
seq_0 = states[:,0]
tar_0 = target_states[:,0]

# for i in range(10):
#     cv2.imwrite(f'coriginal_{i}.png', recover_image(seq_0[i].permute(1,2,0).detach().cpu().numpy().squeeze()))
#     cv2.imwrite(f'ctarget_{i}.png', recover_image(tar_0[i].permute(1,2,0).detach().cpu().numpy().squeeze()))

train_model(model, dataloader, optimizer, num_epochs=10000)
