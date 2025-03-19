import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True) 
            - 2 * torch.matmul(flat_input, self.codebook.weight.t())
            + torch.sum(self.codebook.weight ** 2, dim=1)
        )
        scaled_distances = distances / max(self.temperature.item(), 1e-5)
        soft_assign = F.softmax(-scaled_distances, dim=-1)
        indices = soft_assign.argmax(dim=-1)
        hard_assign = F.one_hot(indices, self.num_tokens).float()
        assign = hard_assign + soft_assign - soft_assign.detach()
        self.usage_count[indices] += 1
        
        # EMA update
        if self.training:
            cluster_size = hard_assign.sum(0)
            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * cluster_size.detach()
            embed_sum = torch.matmul(hard_assign.t(), flat_input)
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * embed_sum.detach()
            
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
        
        # Loss
        commitment_loss = self.beta * F.mse_loss(latent.detach(), quantized)
        codebook_loss = F.mse_loss(latent, quantized.detach())
        
        # Straight-through estimator
        quantized = latent + (quantized - latent).detach()

        return quantized, codebook_loss + commitment_loss, indices


    def get_codebook_metrics(self):
        # Calculate usage probability for each code
        code_probs = self.usage_count / self.usage_count.sum()
        entropy = -(code_probs * torch.log(code_probs + 1e-10)).sum()

        # Calculate average Euclidean distance between codebook vectors
        codebook_distances = torch.cdist(self.codebook.weight, self.codebook.weight)
        mask = ~torch.eye(codebook_distances.shape[0], dtype=bool, device=codebook_distances.device)
        masked_distances = codebook_distances[mask]

        #calculate the perplexity
        avg_probs = code_probs
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return {
            'entropy': entropy.item(),
            'avg_euclidean': masked_distances.mean().item(),
            'min_euclidean': masked_distances.min().item(),
            'perplexity': perplexity.item()
        }


class CausalTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout):
        super(CausalTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention with causal masking
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feedforward network
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src




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

        # Transformer encoder with causal layers
        self.encoder_layers = nn.ModuleList([
            CausalTransformerEncoderLayer(embed_dim, num_heads, 4 * embed_dim, dropout=0.1)
            for _ in range(num_layers)
        ])

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=4 * embed_dim, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embed_dim, beta=0.25)

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

    def forward(self, states, actions, target_states=None, sampling_probability=0.2):
        B, T, C, H, W = states.shape

        # Encode states into embeddings
        state_embeddings = self.state_encoder(states.view(-1, C, H, W))
        state_embeddings = state_embeddings.view(B, T, -1)

        # Embed actions
        action_embeddings = self.action_embedding(actions)

        # Combine state and action embeddings
        seq_embeddings = state_embeddings + action_embeddings
        seq_embeddings += self.positional_embedding[:, :T, :]

        # Generate causal mask
        causal_mask = self.generate_causal_mask(T, states.device)

        # Forward through encoder with causal masking
        for layer in self.encoder_layers:
            seq_embeddings = layer(seq_embeddings, src_mask=causal_mask)

        encoder_output = seq_embeddings  # Final encoder output

        # Apply Vector Quantizer
        quantized, vq_loss, encoding_indices = self.vq(encoder_output)

        # Decoder remains unchanged
        predicted_frames = []
        decoder_inputs = torch.zeros((B, T, self.embed_dim), device=states.device)

        if target_states is not None:
            # Training mode with teacher forcing
            target_embeddings = self.state_encoder(target_states.view(-1, C, H, W))
            target_embeddings = target_embeddings.view(B, T, -1)

            for t in range(T - 1):
                if t == 0 or torch.rand(1).item() < sampling_probability:
                    decoder_inputs[:, t, :] = target_embeddings[:, t, :]
                else:
                    prev_embedding = self.state_encoder(prev_output.view(-1, C, H, W))
                    decoder_inputs[:, t, :] = prev_embedding

                # Add positional embeddings
                curr_decoder_input = decoder_inputs[:, :t+1, :] + self.positional_embedding[:, :t+1, :]

                # Forward through decoder
                output = self.decoder(
                    curr_decoder_input,
                    quantized[:, :t+1, :],
                    tgt_mask=self.generate_causal_mask(t + 1, states.device)
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
                    tgt_mask=self.generate_causal_mask(t + 1, states.device)
                )

                # Reconstruct frame from the last output
                prev_output = self.reconstruct(output[:, -1])
                predicted_frames.append(prev_output)

        predicted_frames = torch.stack(predicted_frames, dim=1)
        return predicted_frames, vq_loss, encoding_indices
