import torch
import torch.nn as nn
import os



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class TransformerEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_heads, num_layers, max_len):
        super(TransformerEncoder, self).__init__()
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        self.eos_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len + 1, hidden_dim))
        
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers
        )
        self.num_heads = num_heads
    
    def generate_causal_mask(self, batch_size, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)
        causal_mask = mask.unsqueeze(0).expand(batch_size+1, -1, -1)
        causal_mask = causal_mask.repeat(self.num_heads, 1, 1)
        return causal_mask
    
    def forward(self, states, actions):
        batch_size, seq_len = states.size(0), states.size(1)
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)
        embeddings = state_emb + action_emb
        eos_token_expanded = self.eos_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([embeddings, eos_token_expanded], dim=1)
        embeddings = embeddings + self.positional_encoding[:, :seq_len + 1, :]
        causal_mask = self.generate_causal_mask(batch_size, seq_len)
        encoded_trajectory = self.transformer(embeddings, embeddings, src_mask=causal_mask)
        eos_representation = encoded_trajectory[:, -1, :]
        return eos_representation

# Vector Quantizer Layer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.commitment_cost = commitment_cost
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(self, inputs):
        flat_inputs = inputs.view(-1, self.embedding_dim)
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) + (self.embedding.weight ** 2).sum(dim=1) \
                    - 2 * torch.matmul(flat_inputs, self.embedding.weight.t())
        
        # Find nearest embedding vector for each input
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings).to(inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view_as(inputs)
        
        # Commitment loss
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices


# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, state_dim, num_heads, num_layers):
        super(TransformerDecoder, self).__init__()
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc_target_embed = nn.Linear(state_dim, hidden_dim)  # Project target states to hidden dim
        self.fc_state = nn.Linear(hidden_dim, state_dim)  # Project hidden dim back to state dimension
    
    def forward(self, quantized_eos_embedding, target_states):
        target_embeddings = self.fc_target_embed(target_states)
        # Use quantized EOS embedding as the encoder memory
        decoded_output = self.transformer(quantized_eos_embedding.unsqueeze(0), target_embeddings)
        predicted_states = self.fc_state(decoded_output)
        
        return predicted_states


class TrajectoryTransformerVQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_heads, num_layers, max_len, num_embeddings):
        super(TrajectoryTransformerVQ, self).__init__()
        self.encoder = TransformerEncoder(state_dim, action_dim, hidden_dim, num_heads, num_layers, max_len)
        self.quantizer = VectorQuantizer(num_embeddings, hidden_dim)
        self.decoder = TransformerDecoder(hidden_dim, state_dim, num_heads, num_layers)
    
    def forward(self, states, actions, target_states=None):
        encoded_trajectory = self.encoder(states, actions)  # Encoded EOS token is the summary
        quantized, quantization_loss, idxs = self.quantizer(encoded_trajectory)
        predicted_states = self.decoder(quantized, target_states)
        
        return predicted_states, quantization_loss
