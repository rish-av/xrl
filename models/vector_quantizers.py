import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class EnhancedEMAVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, 
                 commitment_cost=0.25, decay=0.99, epsilon=1e-5,
                 diversity_weight=1.0, entropy_weight=1.0, temperature=0.1):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.diversity_weight = diversity_weight
        self.temperature = temperature
        self.entropy_weight = entropy_weight

        # Codebook embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.init_embeddings()
        
        # Buffers for EMA updates
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        
        # Usage tracking
        self.register_buffer('usage_count', torch.zeros(num_embeddings))
        self.register_buffer('last_reset_epoch', torch.zeros(1))

    def init_embeddings(self):
        # Initialize embeddings using k-means++ style initialization
        std = 1.0 / self.num_embeddings
        nn.init.uniform_(self.embedding.weight, -std, std)
        self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=1)

    def reset_dead_codes(self, inputs, min_usage=0.01):
        """Reset rarely used codes using samples from high-usage regions"""
        with torch.no_grad():
            # Calculate usage distribution
            total_usage = self.usage_count.sum()
            usage_prob = self.usage_count / total_usage
            
            # Identify dead codes
            dead_codes = usage_prob < min_usage
            if not dead_codes.any():
                return
            
            # Sample new embeddings from inputs near popular codes
            flat_inputs = inputs.reshape(-1, self.embedding_dim)
            distances = self.compute_distances(flat_inputs)
            encoding_indices = torch.argmin(distances, dim=1)
            
            # For each dead code
            for idx in torch.where(dead_codes)[0]:
                # Find popular code regions
                popular_codes = torch.where(usage_prob > torch.quantile(usage_prob, 0.75))[0]
                samples_from_popular = flat_inputs[torch.isin(encoding_indices, popular_codes)]
                
                if len(samples_from_popular) > 0:
                    # Sample and add noise
                    new_embedding = samples_from_popular[torch.randint(0, len(samples_from_popular), (1,))]
                    noise = torch.randn_like(new_embedding) * 0.1
                    self.embedding.weight.data[idx] = new_embedding + noise
                    
            # Reset usage counts for dead codes
            self.usage_count[dead_codes] = 0

    def compute_distances(self, flat_input):
        # Normalize inputs and embeddings
        flat_input_norm = F.normalize(flat_input, dim=1)
        codebook_norm = F.normalize(self.embedding.weight, dim=1)
        similarity = torch.clamp(
            torch.matmul(flat_input_norm, codebook_norm.t()),
            min=-1.0,
            max=1.0
        )
        distances = 2 - 2 * similarity  # transformed cosine distance
        return distances

    def forward(self, inputs, compute_losses=True):
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # Compute distances using normalized vectors
        distances = self.compute_distances(flat_input)
        
        # Compute soft assignments (for diversity loss)
        soft_assign = F.softmax(-distances / self.temperature, dim=1)
        
        # Hard assignments for quantization
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Update usage tracking
        if self.training:
            if self.usage_count.sum() > 1e6:
                self.usage_count.mul_(0.5)  # Scale down instead of zeroing
            self.usage_count.add_(encodings.sum(0))
        
        # Quantized vectors
        quantized = self.embedding(encoding_indices).view(input_shape)
        
        # Compute losses
        loss = 0
        if compute_losses:
            # Commitment loss
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            commitment_loss = q_latent_loss + self.commitment_cost * e_latent_loss
            
            # Diversity loss (minimize codebook vector similarities)
            codebook_norm = F.normalize(self.embedding.weight, dim=1)
            similarity_matrix = torch.matmul(codebook_norm, codebook_norm.t())
            diversity_loss = (similarity_matrix.sum() - self.num_embeddings) / (self.num_embeddings * (self.num_embeddings - 1))
            
            # Entropy loss (encourage uniform usage of codebook)
            avg_probs = soft_assign.mean(0)
            entropy_loss = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
            
            loss = commitment_loss + 0.05 * diversity_loss + 0.1 * entropy_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # EMA updates
        if self.training:
            self.ema_update(flat_input, encodings)
            
            # Periodically reset dead codes
            if (self.usage_count.sum() - self.last_reset_epoch) > 10000:
                self.reset_dead_codes(inputs)
                self.last_reset_epoch = self.usage_count.sum()
            if self.usage_count.sum() > 1e6:
                self.usage_count.zero_()

        return quantized, loss, encoding_indices

    def ema_update(self, flat_input, encodings):
        # Update cluster size
        cluster_size = encodings.sum(0)
        self.ema_cluster_size = self.ema_cluster_size * self.decay + cluster_size * (1 - self.decay)
        
        # Update embeddings
        dw = torch.matmul(encodings.t(), flat_input)
        self.ema_w = self.ema_w * self.decay + dw * (1 - self.decay)
        
        # Normalize cluster size
        n = self.ema_cluster_size.sum()
        cluster_size = ((self.ema_cluster_size + self.epsilon)
                         / (n + self.num_embeddings * self.epsilon)) * n
        
        # Update embedding weights
        normalized_ema_w = self.ema_w / cluster_size.unsqueeze(1)
        self.embedding.weight.data = normalized_ema_w
    def get_codebook_metrics(self):
        """
        Get comprehensive metrics about codebook usage and quality
        """
        with torch.no_grad():
            # Normalize codebook for similarity metrics
            codebook_norm = F.normalize(self.embedding.weight, dim=1)
            
            # Compute pairwise similarities
            similarity_matrix = torch.matmul(codebook_norm, codebook_norm.t())
            
            # Mask out self-similarities
            mask = torch.eye(self.num_embeddings, device=similarity_matrix.device)
            masked_similarity = similarity_matrix * (1 - mask)
            
            # Compute usage statistics
            total_usage = self.usage_count.sum()
            if total_usage > 0:
                usage_probs = self.usage_count / total_usage
            else:
                usage_probs = torch.ones_like(self.usage_count) / self.num_embeddings
                
            # Calculate entropy using torch operations
            entropy = -(usage_probs * torch.log(usage_probs + 1e-10)).sum()
            perplexity = torch.exp(entropy)  # Keep as tensor until final metrics
            
            # Basic metrics - convert to Python floats at the end
            metrics = {
                'avg_similarity': masked_similarity.mean().item(),
                'max_similarity': masked_similarity.max().item(),
                'codebook_entropy': entropy.item(),
                'codebook_perplexity': perplexity.item(),
                'active_codes': (usage_probs > 0.01).sum().item(),
                'usage_uniformity': (perplexity / self.num_embeddings).item()
            }
            
            return metrics
            
    def visualize_codebook(self, save_path='cluster_viz.png'):  
        """
        Create visualizations of codebook structure and usage
        
        Args:
            save_path: Optional path to save visualizations
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA
        
        with torch.no_grad():
            # Get codebook vectors
            codebook = self.embedding.weight.detach().cpu().numpy()
            
            # Compute usage probabilities
            total_usage = self.usage_count.sum().item()
            usage_probs = (self.usage_count / total_usage).cpu().numpy()
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            
            # 1. PCA visualization of codebook vectors
            pca = PCA(n_components=2)
            codebook_2d = pca.fit_transform(codebook)
            
            scatter = axes[0, 0].scatter(
                codebook_2d[:, 0], 
                codebook_2d[:, 1], 
                c=usage_probs, 
                cmap='viridis',
                s=100
            )
            axes[0, 0].set_title('PCA Visualization of Codebook Vectors')
            plt.colorbar(scatter, ax=axes[0, 0], label='Usage Probability')
            
            # 2. Usage distribution
            axes[0, 1].bar(range(self.num_embeddings), usage_probs)
            axes[0, 1].set_title('Codebook Usage Distribution')
            axes[0, 1].set_xlabel('Codebook Index')
            axes[0, 1].set_ylabel('Usage Probability')
            
            # 3. Similarity matrix
            similarity_matrix = torch.matmul(
                F.normalize(self.embedding.weight, dim=1),
                F.normalize(self.embedding.weight, dim=1).t()
            ).cpu().numpy()
            
            sns.heatmap(
                similarity_matrix, 
                ax=axes[1, 0], 
                cmap='coolwarm',
                center=0
            )
            axes[1, 0].set_title('Codebook Similarity Matrix')
            
            # 4. Nearest neighbor distance distribution
            distances = []
            for i in range(self.num_embeddings):
                dist = torch.norm(self.embedding.weight - self.embedding.weight[i], dim=1)
                dist[i] = float('inf')
                distances.append(dist.min().item())
            
            axes[1, 1].hist(distances, bins=20)
            axes[1, 1].set_title('Nearest Neighbor Distance Distribution')
            axes[1, 1].set_xlabel('Distance')
            axes[1, 1].set_ylabel('Count')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()




class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, epsilon=1e-5, commitment_cost=1.0):
        super(EMAVectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon
        self.commitment_cost = commitment_cost

        # Codebook for vector quantization
        print(f"Initializing codebook with {num_embeddings} embeddings of size {embedding_dim}.")
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


class EMAVectorQuantizerNew(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, epsilon=1e-5, commitment_cost=1.0):
        super(EMAVectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon
        self.commitment_cost = commitment_cost

        # Codebook for vector quantization
        print(f"Initializing codebook with {num_embeddings} embeddings of size {embedding_dim}.")
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.cluster_size = nn.Parameter(torch.zeros(num_embeddings), requires_grad=False)
        self.ema_w = nn.Parameter(self.embedding.clone(), requires_grad=False)

        # Affine parameters
        self.affine_mean = nn.Parameter(torch.zeros(embedding_dim))
        self.affine_std = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x):
        """
        Forward pass for the vector quantizer.
        """
        # Flatten input to (batch_size * seq_len, embedding_dim)
        flat_x = x.reshape(-1, self.embedding_dim)

        # Apply affine reparameterization to the embedding
        embedding = self.affine_mean + self.affine_std * self.embedding

        # Compute distances and get encoding indices
        distances = torch.cdist(flat_x, embedding)
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = embedding[encoding_indices].view(x.shape)

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
    

class EnhancedVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, 
                 commitment_cost=0.25,
                 diversity_weight=0.5,  # Increased significantly
                 entropy_weight=0.5,    # Increased significantly
                 temperature=0.2,      # Lowered for harder assignments
                 min_usage_threshold=0.01):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.diversity_weight = diversity_weight
        self.entropy_weight = entropy_weight
        self.temperature = temperature
        self.min_usage_threshold = min_usage_threshold
        
        # Initialize embeddings with orthogonal vectors
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.orthogonal_(self.embed.weight.data)
        
        # Tracking buffers
        self.register_buffer('code_usage', torch.zeros(num_embeddings))
        
    def reinit_unused_codes(self):
        # Get usage statistics
        total_usage = self.code_usage.sum()
        usage_prob = self.code_usage / total_usage
        
        # Find unused codes
        unused_mask = usage_prob < self.min_usage_threshold
        
        if unused_mask.any():
            # Get statistics from used codes
            used_codes = ~unused_mask
            mean_embedding = self.embed.weight[used_codes].mean(0)
            std_embedding = self.embed.weight[used_codes].std(0)
            
            # Reinitialize unused codes with perturbation around used codes
            num_unused = unused_mask.sum()
            noise = torch.randn(num_unused, self.embedding_dim, device=self.embed.weight.device)
            new_embeddings = mean_embedding + noise * std_embedding
            
            # Normalize new embeddings
            new_embeddings = F.normalize(new_embeddings, dim=1)
            
            # Update unused codes
            self.embed.weight.data[unused_mask] = new_embeddings
            self.code_usage[unused_mask] = 0
    
    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # Normalize inputs and embeddings
        flat_input_norm = F.normalize(flat_input, dim=1)
        codebook_norm = F.normalize(self.embed.weight, dim=1)
        
        # Compute distances
        distances = torch.cdist(flat_input, self.embed.weight, p=2)
        
        # Add usage penalty to distances to encourage using rare codes
        if self.training:
            usage_probs = (self.code_usage + 1e-5) / (self.code_usage + 1e-5).sum()
            usage_penalty = 2.0 * torch.log(usage_probs + 1e-5)  # Higher penalty for frequently used codes
            distances = distances + usage_penalty.unsqueeze(0)
        
        # Get hard assignments
        encoding_indices = torch.argmin(distances, dim=1)
        hard_assignments = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Get quantized embeddings
        quantized = self.embed(encoding_indices)
        
        # Compute losses
        commitment_loss = F.mse_loss(quantized.detach(), flat_input)
        
        # Stronger diversity loss
        similarity_matrix = torch.matmul(codebook_norm, codebook_norm.t())
        off_diag = similarity_matrix - torch.eye(self.num_embeddings, device=similarity_matrix.device)
        diversity_loss = torch.pow(torch.clamp(torch.abs(off_diag), min=0.1), 2).mean()
        
        # Stronger entropy loss
        avg_probs = hard_assignments.mean(0)
        uniform_probs = torch.ones_like(avg_probs) / self.num_embeddings
        entropy_loss = F.kl_div(
            (avg_probs + 1e-10).log(),
            uniform_probs,
            reduction='sum',
            log_target=False
        )
        
        # Update usage statistics
        with torch.no_grad():
            if self.training:
                self.code_usage.mul_(0.99).add_(hard_assignments.sum(0), alpha=0.01)
                
                # Force redistribution if too few codes are active
                usage_probs = self.code_usage / self.code_usage.sum()
                active_codes = (usage_probs > self.min_usage_threshold).sum().item()
                
                if active_codes < self.num_embeddings // 4:  # If using less than 25% of codes
                    # Reset rarely used codes to be perturbations of most used codes
                    used_indices = torch.topk(self.code_usage, k=active_codes).indices
                    unused_indices = torch.topk(-self.code_usage, k=self.num_embeddings - active_codes).indices
                    
                    # Get mean and std of used embeddings
                    used_embeddings = self.embed.weight.data[used_indices]
                    mean_embedding = used_embeddings.mean(0)
                    std_embedding = used_embeddings.std(0)
                    
                    # Reset unused embeddings with noise
                    noise = torch.randn_like(self.embed.weight.data[unused_indices]) * 0.1
                    new_embeddings = mean_embedding.unsqueeze(0) + noise * std_embedding.unsqueeze(0)
                    new_embeddings = F.normalize(new_embeddings, dim=1)
                    
                    self.embed.weight.data[unused_indices] = new_embeddings
                    self.code_usage[unused_indices] = self.code_usage.mean()
        
        # Total loss with increased weights for diversity and entropy
        total_loss = (self.commitment_cost * commitment_loss + 
                     self.diversity_weight * diversity_loss + 
                     self.entropy_weight * entropy_loss)
        
        # Straight-through estimator
        quantized = flat_input + (quantized - flat_input).detach()
        quantized = quantized.view(input_shape)
        
        return quantized, total_loss, encoding_indices

    def get_codebook_metrics(self):
        with torch.no_grad():
            # Normalize codebook
            codebook_norm = F.normalize(self.embed.weight, dim=1)
            
            # Compute pairwise similarities
            similarity_matrix = torch.matmul(codebook_norm, codebook_norm.t())
            
            # Get metrics
            avg_similarity = similarity_matrix.mean().item()
            max_similarity = similarity_matrix.max().item()
            
            # Usage statistics
            usage_probs = self.code_usage / self.code_usage.sum()
            entropy = -(usage_probs * torch.log(usage_probs + 1e-10)).sum().item()
            active_codes = (usage_probs > 0.01).sum().item()
            
            return {
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'codebook_entropy': entropy,
                'active_codes': active_codes
            }