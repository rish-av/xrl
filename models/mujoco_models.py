import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import warnings
np.warnings = warnings
import cv2
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


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
        scaled_distances = distances / max(self.temperature.item(), 1e-5)
        
        # Softmax with temperature
        soft_assign = F.softmax(-scaled_distances, dim=-1)
        
        # Get indices and hard assignment
        indices = soft_assign.argmax(dim=-1)
        hard_assign = F.one_hot(indices, self.num_tokens).float()
        assign = hard_assign + soft_assign - soft_assign.detach()
        
        # Update usage count
        self.usage_count[indices] += 1
        
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
        
        # Calculate Euclidean distances between codebook vectors
        codebook_distances = torch.cdist(self.codebook.weight, self.codebook.weight)
        mask = ~torch.eye(codebook_distances.shape[0], dtype=bool, device=codebook_distances.device)
        masked_distances = codebook_distances[mask]
        avg_euclidean = masked_distances.mean()
        min_euclidean = masked_distances.min()
        
        # Quantize
        quantized = torch.matmul(assign, self.codebook.weight)
        quantized = quantized.view(batch_size, seq_len, latent_dim)
        
        # Loss
        commitment_loss = self.beta * F.mse_loss(latent.detach(), quantized)
        codebook_loss = F.mse_loss(latent, quantized.detach())
        
        # Straight-through estimator
        quantized = latent + (quantized - latent).detach()
        
        # Calculate perplexity
        avg_probs = torch.mean(hard_assign, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return (quantized, indices, commitment_loss, codebook_loss, 
                perplexity, selected_cosine_sim, avg_euclidean, min_euclidean)


class Seq2SeqMujocoTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, num_tokens, hidden_size, n_layers, n_heads, context_len, beta,
                 temp_init=1.0, temp_min=0.1, anneal_rate=0.00003, ema_decay=0.99):
        super().__init__()

        # Embedding and normalization layers
        self.state_embedding = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_size)
        )
        self.action_embedding = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, hidden_size)
        )
        self.pos_embedding = nn.Embedding(context_len, hidden_size)

        # Causal masks
        self.register_buffer(
            "encoder_mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        )
        self.register_buffer(
            "decoder_mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        )

        # Transformer encoder with norm_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size * 4, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Latent projection
        self.to_latent = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, latent_dim)
        )

        # Vector Quantizer
        self.vector_quantizer = VectorQuantizer(
            num_tokens, latent_dim, beta, 
        )

        # Decoder layers
        self.from_latent = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_size)
        )

        # Transformer decoder with norm_first=True
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size * 4, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_state = nn.Linear(hidden_size, state_dim)

        self.model_config = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "latent_dim": latent_dim,
            "num_tokens": num_tokens,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "context_len": context_len,
            "beta": beta
        }

    def encode(self, states, actions):
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)
        combined_emb = state_emb + action_emb

        # Add positional embeddings
        position_ids = torch.arange(states.size(1), device=states.device).unsqueeze(0)
        position_emb = self.pos_embedding(position_ids)
        combined_emb += position_emb

        # Encoder with causal mask
        encoded = self.encoder(combined_emb, mask=self.encoder_mask)
        return self.to_latent(encoded)

    def decode(self, quantized, targets, teacher_forcing_ratio=0.5):
        batch_size, seq_len = targets.shape[0], targets.shape[1]
        
        # Transform quantized vectors to decoder space
        decoder_memory = self.from_latent(quantized)
        
        current_input = torch.zeros_like(targets[:, 0]).unsqueeze(1)
        outputs = []
        tf_count = 0
        
        for t in range(seq_len):
            # Create causal mask for current timestep
            # Change mask shape to match requirements
            tgt_mask = torch.ones((t+1, t+1), device=current_input.device).triu_(1).bool()
            memory_mask = torch.ones((t+1, seq_len), device=current_input.device).bool()
            
            decoder_input = self.state_embedding(current_input)
            decoder_output = self.decoder(
                decoder_input, 
                decoder_memory,
                tgt_mask=tgt_mask,             # For self-attention
                memory_mask=None  # For cross-attention, we can leave it as None since we want to attend to all memory
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
        
        # Calculate total loss
        total_loss = reconstruction_loss + commitment_loss + codebook_loss
        
        print(f"loss: {reconstruction_loss.item():.4f}, commitment: {commitment_loss.item():.4f}, "
            f"codebook: {codebook_loss.item():.4f}, perplexity: {perplexity.item():.4f}, "
            f"temperature: {self.vector_quantizer.temperature:.4f}, "
            f"cosine_similarity: {cosine_sim.item():.4f}, "
            f"avg_euclidean: {avg_euclidean.item():.4f}, "
            f"min_euclidean: {min_euclidean.item():.4f}")
                
        return (predicted_states, total_loss, reconstruction_loss, commitment_loss, 
                codebook_loss, perplexity, cosine_sim, avg_euclidean, min_euclidean)
    




def bifurcate_segments(lists, states, actions):
    segments_dict = defaultdict(list)
    state_action_segs = defaultdict(list)

    for lst, state, action in zip(lists, states, actions):
        current_segment = []
        current_segment_state_action = []

        for i, val in enumerate(lst):
            if not current_segment or val == current_segment[-1]:
                current_segment.append(val)
                current_segment_state_action.append((state[i], action[i]))
            else:
                segments_dict[current_segment[0]].append(current_segment)
                current_segment = [val]

                state_action_segs[current_segment[0]].append(current_segment_state_action)
                current_segment_state_action = [(state[i], action[i])]

        
        if current_segment:
            segments_dict[current_segment[0]].append(current_segment)
            state_action_segs[current_segment[0]].append(current_segment_state_action)
        
        print(f"Segments dict: {segments_dict.keys()}")


    return dict(segments_dict), dict(state_action_segs)





class MLPPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(MLPPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.net(state)

class BCDataset(Dataset):
    def __init__(self, segments):
        self.states = []
        self.actions = []
        
        for segment in segments:
            for (s, a) in segment:
                self.states.append(s)
                self.actions.append(a)

        self.states = torch.FloatTensor(self.states)
        self.actions = torch.FloatTensor(self.actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

def train_bc_with_eval(data_dict, epochs=50, batch_size=256, lr=3e-4, hidden_dim=256, train_split=0.9):
    policy_dict = {}

    for key, segments in data_dict.items():
        dataset = BCDataset(segments)
        total_size = len(dataset)
        train_size = int(train_split * total_size)
        test_size = total_size - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        state_dim = dataset.states.shape[1]
        action_dim = dataset.actions.shape[1]

        policy = MLPPolicy(state_dim, action_dim, hidden_dim)
        optimizer = optim.Adam(policy.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            policy.train()
            total_train_loss = 0
            for states, actions in train_loader:
                pred_actions = policy(states)
                loss = criterion(pred_actions, actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Evaluation on Test Set
            policy.eval()
            total_test_loss = 0
            with torch.no_grad():
                for states, actions in test_loader:
                    pred_actions = policy(states)
                    loss = criterion(pred_actions, actions)
                    total_test_loss += loss.item()

            avg_test_loss = total_test_loss / len(test_loader)

            print(f"Key {key} - Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        policy_dict[key] = policy

    import os
    #save the models
    for key, policy in policy_dict.items():
        os.makedirs("bc_models", exist_ok=True)
        torch.save(policy.state_dict(), f"bc_models/{key}.pt")

    return policy_dict


