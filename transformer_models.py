import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_heads, num_layers, max_len):
        super(TransformerEncoder, self).__init__()
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers
        )
    
    def generate_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def forward(self, states, actions):
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)
        embeddings = state_emb + action_emb + self.positional_encoding[:, :states.size(1), :]
        causal_mask = self.generate_causal_mask(embeddings.size(1)).to(embeddings.device)
        encoded_trajectory = self.transformer(embeddings, embeddings, src_mask=causal_mask)
        
        return encoded_trajectory


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
    def __init__(self, hidden_dim, state_dim, action_dim, num_heads, num_layers, max_len):
        super(TransformerDecoder, self).__init__()
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc_state = nn.Linear(hidden_dim, state_dim)
        self.fc_action = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, quantized_embeddings, target_states, target_actions):
        target_embeddings = self.fc_state(target_states) + self.fc_action(target_actions)
        decoded_output = self.transformer(quantized_embeddings, target_embeddings)
        predicted_states = self.fc_state(decoded_output)
        predicted_actions = self.fc_action(decoded_output)
        
        return predicted_states, predicted_actions


class TrajectoryTransformerVQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_heads, num_layers, max_len, num_embeddings):
        super(TrajectoryTransformerVQ, self).__init__()
        self.encoder = TransformerEncoder(state_dim, action_dim, hidden_dim, num_heads, num_layers, max_len)
        self.quantizer = VectorQuantizer(num_embeddings, hidden_dim)
        self.decoder = TransformerDecoder(hidden_dim, state_dim, action_dim, num_heads, num_layers, max_len)
    
    def forward(self, states, actions, target_states=None, target_actions=None):
        encoded_trajectory = self.encoder(states, actions)
        quantized, quantization_loss, _ = self.quantizer(encoded_trajectory)
        predicted_states, predicted_actions = self.decoder(quantized, target_states, target_actions)
        
        return predicted_states, predicted_actions, quantization_loss
