import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

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
        
        return quantized, vq_loss, encoding_indices  