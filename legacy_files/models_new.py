import gym
import d4rl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
import torch.nn.functional as F

#transformer encoder
import torch
import torch.nn as nn
import math


def generate_causal_mask(size):
    mask = torch.tril(torch.ones(size, size))  # Lower triangular matrix
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Positional Encoding Module
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]

# Segment Transformer Encoder with Positional Encoding
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_embeddings):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.eos_embedding = nn.Parameter(torch.randn(1, 1, model_dim))  # Learnable EOS embedding
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.vq_layer = EMAVectorQuantizer(num_embeddings, model_dim)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding(seq_len)
        
        # Append EOS embedding at the end of sequence
        eos_embedding_expanded = self.eos_embedding.expand(x.size(0), -1, -1)  # Match batch size
        x = torch.cat([x, eos_embedding_expanded], dim=1)  # Append EOS token to each sequence
        
        # Apply causal mask to encoder
        causal_mask = generate_causal_mask(x.size(1)).to(x.device)
        x = self.transformer_encoder(x.permute(1, 0, 2), mask=causal_mask).permute(1, 0, 2)
        
        # Apply EMA VQ layer to quantize encoder outputs and get VQ loss
        quantized, vq_loss, encidx = self.vq_layer(x)
        
        return quantized[:, -1, :], vq_loss, encidx 


#transformer decoder
class TransformerDecoder(nn.Module):
    def __init__(self, model_dim, output_dim, num_heads, num_layers):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(output_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, tgt, memory):
        seq_len = tgt.size(1)
        tgt = self.embedding(tgt) + self.positional_encoding(seq_len)
        
        # Apply causal mask to decoder
        causal_mask = generate_causal_mask(seq_len).to(tgt.device)
        tgt = self.transformer_decoder(tgt.permute(1, 0, 2), memory.unsqueeze(0), tgt_mask=causal_mask)
        
        return self.fc_out(tgt.permute(1, 0, 2))



# class SoftEMA_VQEmbedding(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim, initial_temperature=1.0, min_temperature=0.1, anneal_rate=0.99, decay=0.99, epsilon=1e-5, commitment_cost=0.25):
#         super(SoftEMA_VQEmbedding, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.num_embeddings = num_embeddings
#         self.temperature = initial_temperature
#         self.min_temperature = min_temperature
#         self.anneal_rate = anneal_rate
#         self.decay = decay
#         self.epsilon = epsilon
#         self.commitment_cost = commitment_cost

#         self.embedding = nn.Embedding(num_embeddings, embedding_dim)
#         nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        
#         self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
#         self.register_buffer("ema_embedding_avg", self.embedding.weight.clone())

#     def anneal_temperature(self):
#         self.temperature = max(self.min_temperature, self.temperature * self.anneal_rate)
        
#     def forward(self, inputs):
#         flat_input = inputs.view(-1, self.embedding_dim)
#         distances = (
#             torch.sum(flat_input ** 2, dim=1, keepdim=True) 
#             + torch.sum(self.embedding.weight ** 2, dim=1) 
#             - 2 * torch.matmul(flat_input, self.embedding.weight.t())
#         )
        
#         soft_assignments = F.softmax(-distances / self.temperature, dim=1)
#         quantized = torch.matmul(soft_assignments, self.embedding.weight).view(inputs.shape)
        
#         if self.training:
#             encoding_probs = soft_assignments.sum(0)
#             updated_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * encoding_probs
#             updated_embedding_avg = self.decay * self.ema_embedding_avg + (1 - self.decay) * torch.matmul(soft_assignments.t(), flat_input)

#             n = updated_cluster_size.sum()
#             normalized_cluster_size = (updated_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n

#             self.ema_cluster_size = updated_cluster_size
#             self.ema_embedding_avg = updated_embedding_avg

#             self.embedding.weight.data = self.ema_embedding_avg / normalized_cluster_size.unsqueeze(1)

#         reconstruction_loss = torch.mean((quantized - inputs) ** 2)
#         commitment_loss = torch.mean((quantized.detach() - inputs) ** 2)
#         vq_loss = reconstruction_loss + self.commitment_cost * commitment_loss

#         return quantized, vq_loss, soft_assignments

#     def initialize_codebook_with_kmeans(self, inputs):
       
#         flat_input = inputs.view(-1, self.embedding_dim).detach().cpu().numpy()
#         kmeans = KMeans(n_clusters=self.num_embeddings, random_state=0)
#         kmeans.fit(flat_input)
#         centroids = kmeans.cluster_centers_
#         self.embedding.weight.data.copy_(torch.tensor(centroids, dtype=torch.float32))
    
class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, epsilon=1e-5, commitment_cost=0.25):
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
    def initialize_codebook_with_kmeans(self, inputs):
       
        flat_input = inputs.view(-1, self.embedding_dim).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_embeddings, random_state=0)
        kmeans.fit(flat_input)
        centroids = kmeans.cluster_centers_
        self.embedding.weight.data.copy_(torch.tensor(centroids, dtype=torch.float32))


    def reset_unused_embeddings(self, usage_threshold=10):
        underutilized_embeddings = self.ema_cluster_size < usage_threshold
        num_underutilized = underutilized_embeddings.sum().item()
        if num_underutilized > 0:
            print(f"Resetting {num_underutilized} underutilized embeddings.")
        with torch.no_grad():
            self.embedding.weight[underutilized_embeddings] = torch.randn_like(
                self.embedding.weight[underutilized_embeddings]
            ) * 0.1  # Small random values
        self.ema_cluster_size[underutilized_embeddings] = usage_threshold

    def forward(self, x):
        # Flatten input to (batch_size * seq_len, embedding_dim)
        flat_x = x.reshape(-1, self.embedding_dim)
        
        # Compute distances and get encoding indices
        distances = torch.cdist(flat_x, self.embedding)
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding[encoding_indices].view(x.shape)
        
        # Compute VQ Loss (Codebook loss + Commitment loss)
        codebook_loss = F.mse_loss(quantized.detach(), x)  # Codebook loss
        commitment_loss = F.mse_loss(quantized, x.detach())  # Commitment loss
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # EMA updates for embedding (codebook)
        if self.training:
            encoding_one_hot = F.one_hot(encoding_indices, self.num_embeddings).type_as(flat_x)
            new_cluster_size = encoding_one_hot.sum(dim=0)
            self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)
            dw = encoding_one_hot.t() @ flat_x
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            # Normalize to prevent embedding drift
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            self.embedding.data = self.ema_w / cluster_size.unsqueeze(1)
        
        # Straight-through gradient
        quantized = x + (quantized - x).detach()
        
        return quantized, vq_loss, encoding_indices  # Return quantized output and VQ loss

class BexRL(nn.Module):
    def __init__(self, input_dim, model_dim, output_dim, num_heads, num_encoder_layers, num_decoder_layers, num_embeddings):
        super(BexRL, self).__init__()
        self.encoder = TransformerEncoder(input_dim, model_dim, num_heads, num_encoder_layers, num_embeddings)
        self.decoder = TransformerDecoder(model_dim, output_dim, num_heads, num_decoder_layers)
    
    def forward(self, src, tgt):
        memory, vq_loss, encidx = self.encoder(src)  # Only the final quantized state and VQ loss from encoder
        output = self.decoder(tgt, memory)  # Decode future states from the quantized encoder output
        return output, vq_loss, encidx  


    


# class BeXRLSoftVQ(nn.Module):
#     def __init__(self, feature_dim, model_dim, num_heads, num_layers, num_embeddings, segment_length, initial_temperature=1.0, min_temperature=0.1, anneal_rate=0.99):
#         super(BeXRLSoftVQ, self).__init__()
#         self.segment_length = segment_length
#         self.encoder = TransformerDecoder(feature_dim, model_dim, num_heads, num_layers)
#         self.vq_layer = SoftEMA_VQEmbedding(num_embeddings, model_dim, initial_temperature, min_temperature, anneal_rate)
#         self.decoder = TransformerDecoder(model_dim, feature_dim, num_heads, num_layers)

#     def initialize_vq_with_kmeans(self, x):
#         segments = x.view(-1, self.segment_length, x.size(-1))
#         encoded_segments = self.encoder(segments)
#         end_of_sequence_repr = encoded_segments[:, -1, :].detach()
#         self.vq_layer.initialize_codebook_with_kmeans(end_of_sequence_repr)

#     def forward(self, x):
#         segments = x.view(-1, self.segment_length, x.size(-1))
#         encoded_segments = self.encoder(segments)
#         end_of_sequence_repr = encoded_segments[:, -1, :]

#         quantized_segments, vq_loss, soft_assignments = self.vq_layer(end_of_sequence_repr)
        
#         reconstructed_segments = self.decoder(quantized_segments.unsqueeze(0), encoded_segments)
#         reconstructed_sequence = reconstructed_segments.view(-1, reconstructed_segments.size(-1))

#         # Anneal the temperature for the soft assignments
#         # self.vq_layer.anneal_temperature()
        
#         return reconstructed_sequence, vq_loss, soft_assignments
