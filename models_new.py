import gym
import d4rl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans

#transformer encoder
class SegmentTransformerEncoder(nn.Module):
    def __init__(self, feature_dim, model_dim, num_heads, num_layers):
        super(SegmentTransformerEncoder, self).__init__()
        self.embedding = nn.Linear(feature_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def generate_causal_mask(self, size):
        mask = torch.tril(torch.ones(size, size)).to(self.device)  # Lower triangular matrix
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        x = self.embedding(x) 
        x = x.permute(1, 0, 2)  

        # Generate causal mask
        causal_mask = self.generate_causal_mask(x.size(0))  # Shape: [segment_length, segment_length]
        encoded_segments = self.transformer_encoder(x, is_causal=True, mask=causal_mask)  
        return encoded_segments.permute(1, 0, 2)


#transformer decoder
class TransformerDecoder(nn.Module):
    def __init__(self, model_dim, feature_dim, num_heads, num_layers):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, feature_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def generate_causal_mask(self, size):
        mask = torch.tril(torch.ones(size, size)).to(self.device)  # Lower triangular matrix
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, memory, target_seq):
        target_seq = target_seq.permute(1, 0, 2) 
        
        # Generate causal mask
        causal_mask = self.generate_causal_mask(target_seq.size(0))  # Shape: [seq_len, seq_len]

        # Apply the transformer decoder with causal mask
        decoded_output = self.transformer_decoder(target_seq, memory, tgt_mask=causal_mask)
        return self.output_layer(decoded_output.permute(1, 0, 2))


#vector quantizer with EMA
class EMA_VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(EMA_VQEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embedding_avg', self.embedding.weight.clone())

    def initialize_codebook_with_kmeans(self, inputs):
       
        flat_input = inputs.view(-1, self.embedding_dim).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_embeddings, random_state=0)
        kmeans.fit(flat_input)
        centroids = kmeans.cluster_centers_
        self.embedding.weight.data.copy_(torch.tensor(centroids, dtype=torch.float32))

    def forward(self, inputs):
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight ** 2, dim=1) 
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        if self.training:
            encoding_one_hot = encodings
            updated_cluster_size = torch.sum(encoding_one_hot, dim=0)
            updated_embedding_avg = torch.matmul(encoding_one_hot.t(), flat_input)

            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * updated_cluster_size
            self.ema_embedding_avg = self.decay * self.ema_embedding_avg + (1 - self.decay) * updated_embedding_avg
            n = self.ema_cluster_size.sum()
            cluster_size = ((self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon)) * n

            self.embedding.weight.data = self.ema_embedding_avg / cluster_size.unsqueeze(1)

        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        return quantized, vq_loss, encoding_indices.view(inputs.size(0), -1)
    


class BeXRL(nn.Module):
    def __init__(self, feature_dim, model_dim, num_heads, num_layers, num_embeddings, segment_length):
        super(BeXRL, self).__init__()
        self.segment_length = segment_length
        self.encoder = SegmentTransformerEncoder(feature_dim, model_dim, num_heads, num_layers)
        self.vq_layer = EMA_VQEmbedding(num_embeddings, model_dim)
        self.decoder = TransformerDecoder(model_dim, feature_dim, num_heads, num_layers)

    def initialize_vq_with_kmeans(self, x):
        segments = x.view(-1, self.segment_length, x.size(-1))
        encoded_segments = self.encoder(segments)
        end_of_sequence_repr = encoded_segments[:, -1, :].detach()
        self.vq_layer.initialize_codebook_with_kmeans(end_of_sequence_repr)

    def forward(self, x):
        segments = x.view(-1, self.segment_length, x.size(-1))
        encoded_segments = self.encoder(segments)
        end_of_sequence_repr = encoded_segments[:, -1, :]

        quantized_segments, vq_loss, encoding_indices = self.vq_layer(end_of_sequence_repr)
        
        reconstructed_segments = self.decoder(quantized_segments.unsqueeze(0), encoded_segments)
        reconstructed_sequence = reconstructed_segments.view(-1, reconstructed_segments.size(-1))
        return reconstructed_sequence, vq_loss, encoding_indices