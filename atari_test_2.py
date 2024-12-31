import d4rl_atari
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse

# ------------------------------------------------------------------
#  1) Dataset class for Atari with frameskip
# ------------------------------------------------------------------
class AtariD4RLDataset(Dataset):
    def __init__(self, env_name, context_len=8, frameskip=4, transform=None):
        super().__init__()
        self.env = gym.make(env_name)
        dataset = self.env.get_dataset()
        self.frames = dataset['observations']
        self.actions = dataset['actions']

        self.context_len = context_len
        self.frameskip = frameskip
        self.transform = transform

        self.data = []
        max_index = len(self.frames) - context_len * frameskip
        for i in range(max_index):
            frame_seq = []
            action_seq = []

            for step in range(context_len):
                idx = i + step * frameskip
                frame_seq.append(self.frames[idx])
                action_seq.append(self.actions[idx])

            target_seq = frame_seq[1:] + [self.frames[i + context_len * frameskip]]

            self.data.append({
                "states": np.array(frame_seq),
                "actions": np.array(action_seq),
                "targets": np.array(target_seq)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        frames = item["states"]
        actions = item["actions"]
        targets = item["targets"]

        if self.transform is not None:
            frames = np.stack([self.transform(f) for f in frames], axis=0)
            targets = np.stack([self.transform(t) for t in targets], axis=0)

        frames = torch.tensor(frames, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)

        return {
            "states": frames,
            "actions": actions,
            "targets": targets
        }

# ------------------------------------------------------------------
#  2) Model classes with probabilistic teacher forcing
# ------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0).to(x.device)
        return x

class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embed = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embed.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.ema_w = nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))
        self.ema_w.data.normal_()

    def forward(self, inputs):
        flat_inputs = inputs.view(-1, self.embedding_dim)

        distances = (
            flat_inputs.pow(2).sum(1, keepdim=True)
            - 2 * flat_inputs @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(dim=1, keepdim=True).t()
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(inputs.dtype)

        quantized = self.embed(encoding_indices).view_as(inputs)

        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encodings, 0)
            dw = encodings.t() @ flat_inputs
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)

            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n

            updated_embed = self.ema_w / self.ema_cluster_size.unsqueeze(1)
            self.embed.weight.data.copy_(updated_embed)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices

class Seq2SeqTransformer(nn.Module):
    def __init__(self, image_size, action_dim, embed_dim, n_heads, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super(Seq2SeqTransformer, self).__init__()

        self.positional_encoding = PositionalEncoding(embed_dim)
        self.vqae = EMAVectorQuantizer(num_embeddings=128, embedding_dim=embed_dim, commitment_cost=1.0)

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

        self.action_embedding = nn.Embedding(action_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(embed_dim, 128 * (image_size // 8) * (image_size // 8)),
            nn.Unflatten(1, (128, image_size // 8, image_size // 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, states, actions, tgt_states, teacher_forcing_ratio=0.5):
        B, T, C, H, W = states.size()

        encoder_mask = self.generate_square_subsequent_mask(T).to(states.device)
        decoder_mask = self.generate_square_subsequent_mask(T).to(states.device)

        encoder_input = []
        for t in range(T):
            emb_frame = self.image_encoder(states[:, t])
            emb_action = self.action_embedding(actions[:, t])
            encoder_input.append(emb_frame + emb_action)
        encoder_input = torch.stack(encoder_input, dim=1)
        encoder_input = self.positional_encoding(encoder_input)

        encoder_out = self.encoder(encoder_input, mask=encoder_mask)
        eos_rep = encoder_out[:, -1, :]

        eos_rep_quant, vq_loss, encoding_indices = self.vqae(eos_rep)

        decoder_input = []
        for t in range(T):
            if t == 0:
                zero_frame = torch.zeros_like(tgt_states[:, t])
                emb = self.image_encoder(zero_frame)
            else:
                if torch.rand(1).item() < teacher_forcing_ratio:
                    emb = self.image_encoder(tgt_states[:, t - 1])
                else:
                    emb = self.image_encoder(reconstructed_states[:, t - 1].detach())
            decoder_input.append(emb)

        decoder_input = torch.stack(decoder_input, dim=1)
        decoder_input = self.positional_encoding(decoder_input)

        memory = eos_rep_quant.unsqueeze(1).expand(-1, T, -1)

        decoder_out = self.decoder(
            decoder_input,
            memory,
            tgt_mask=decoder_mask
        )

        reconstructed_states = []
        for t in range(T):
            dec_t = decoder_out[:, t]
            rec_frame = self.reconstruction_decoder(dec_t)
            reconstructed_states.append(rec_frame)
        reconstructed_states = torch.stack(reconstructed_states, dim=1)

        return reconstructed_states, vq_loss, encoding_indices

# ------------------------------------------------------------------
#  3) Training function and argument parser
# ------------------------------------------------------------------
def train_model(args):
    dataset = AtariD4RLDataset(
        env_name=args.env_name,
        context_len=args.context_len,
        frameskip=args.frameskip
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = Seq2SeqTransformer(
        image_size=args.image_size,
        action_dim=args.action_dim,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    teacher_forcing_ratio = args.teacher_forcing_ratio

    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(dataloader):
            states = batch["states"].cuda()
            actions = batch["actions"].cuda()
            targets = batch["targets"].cuda()

            reconstructed, vq_loss, _ = model(states, actions, targets, teacher_forcing_ratio)
            recon_loss = F.mse_loss(reconstructed, targets)

            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.log_interval == 0:
                print(f"Epoch={epoch}, Batch={i}, Total Loss={loss.item():.4f}, "
                      f"Recon={recon_loss.item():.4f}, VQ={vq_loss.item():.4f}")

        teacher_forcing_ratio *= args.teacher_forcing_decay

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="Breakout-v0", help='Atari environment name')
    parser.add_argument('--context_len', type=int, default=8, help='Length of context sequence')
    parser.add_argument('--frameskip', type=int, default=4, help='Frameskip value')
    parser.add_argument('--image_size', type=int, default=84, help='Input image size')
    parser.add_argument('--action_dim', type=int, default=18, help='Number of discrete actions')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Dimension of feedforward layer')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.9, help='Initial teacher forcing ratio')
    parser.add_argument('--teacher_forcing_decay', type=float, default=0.9, help='Decay rate for teacher forcing ratio')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    return parser.parse_args()

# ------------------------------------------------------------------
#  4) Main entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    train_model(args)
