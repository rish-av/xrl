import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import argparse
import os
import gym
import d4rl
from torch.utils.data import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train Atari Model with Custom Model Name in WandB")
    
    # Model hyperparameters
    parser.add_argument('--max_seq_len', type=int, default=60, help="Maximum sequence length")
    parser.add_argument('--embed_dim', type=int, default=128, help="Embedding dimension")
    parser.add_argument('--nhead', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of transformer layers")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--num_embeddings', type=int, default=128, help="Number of embeddings for VQ")

    # Logging and experiment tracking
    parser.add_argument('--log', action='store_true', help="Enable logging with WandB")
    parser.add_argument('--wandb_project', type=str, default="atari-xrl", help="WandB project name")
    parser.add_argument('--wandb_entity', type=str, default="mail-rishav9", help="WandB entity name")
    parser.add_argument('--wandb_run_name', type=str, default="xrl_atari", help="Base name for WandB run")
    parser.add_argument('--patch_size', type=int, default=4, help="Patch size for image transformer")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument('--load_checkpoint', type=str, default=None, help="Path to load model checkpoint")
    parser.add_argument('--epochs', type=int, default=2000, help="Number of training epochs")
    parser.add_argument('--scheduler_step_size', type=int, default=50, help="Step size for learning rate scheduler")
    parser.add_argument('--frame_skip', type=int, default=4, help="Number of frames to skip in dataset")

    return parser.parse_args()


args = parse_args()

def diversity_loss(embeddings):
    similarity = torch.matmul(embeddings, embeddings.t())
    identity = torch.eye(embeddings.shape[0], device=embeddings.device)
    return F.mse_loss(similarity, identity)

def entropy_regularization(quantized_indices, num_embeddings):
    # Calculate usage probability for each code
    code_probs = F.one_hot(quantized_indices, num_embeddings).float().mean(0)
    # Add small epsilon to avoid log(0)
    entropy = -(code_probs + 1e-8).log() * code_probs
    return -entropy.sum()  # Negative because we want to maximize entropy


class MiniGridDataset(Dataset):
    def __init__(self, env_name, max_seq_len=30, transform=None, frame_skip=1, action_dim=6):
        self.dataset = gym.make(env_name).get_dataset()
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.frame_skip = frame_skip
        self.action_dim = action_dim
        self.total_frames = len(self.dataset['observations'])
        self.valid_starts = self._get_valid_start_indices()

    def _get_valid_start_indices(self):
        valid_starts = []
        for i in range(0, self.total_frames - self.max_seq_len * self.frame_skip, self.frame_skip):
            terminals = self.dataset['terminals'][i:i + self.max_seq_len * self.frame_skip:self.frame_skip]
            if not np.any(terminals):
                valid_starts.append(i)
        return valid_starts

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]
        indices = range(start_idx, start_idx + self.max_seq_len * self.frame_skip, self.frame_skip)
        states = [self.dataset['observations'][i] for i in indices]
        actions = [int(self.dataset['actions'][i]) for i in indices]  # Ensure actions are integers
        target_states = [self.dataset['observations'][i + 1] for i in indices]
        
        # Convert states and target_states to flattened format
        states = np.array(states).reshape(len(states), -1)  # Flatten each state
        target_states = np.array(target_states).reshape(len(target_states), -1)  # Flatten each state
        actions = np.array(actions, dtype=int)  # Explicit integer conversion
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)  # Ensure LongTensor receives integers
        target_states = torch.FloatTensor(target_states)
        return states, actions, target_states


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

        total_loss = commitment_loss + codebook_loss

        return quantized, total_loss, indices
    def get_codebook_metrics(self):
        with torch.no_grad():
            # Normalize codebook
            codebook_norm = F.normalize(self.codebook.weight, dim=1)
            
            # Compute pairwise similarities
            similarity_matrix = torch.matmul(codebook_norm, codebook_norm.t())
            
            # Get metrics
            avg_similarity = similarity_matrix.mean().item()
            max_similarity = similarity_matrix.max().item()
            
            # Usage statistics
            # usage_probs = self.code_usage / self.code_usage.sum()
            # entropy = -(usage_probs * torch.log(usage_probs + 1e-10)).sum().item()
            # active_codes = (usage_probs > 0.01).sum().item()
            
            return {
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
            }



class EnhancedVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, 
                 commitment_cost=0.25,
                 diversity_weight=2.0,  # Increased significantly
                 entropy_weight=2.0,    # Increased significantly
                 temperature=0.05,      # Lowered for harder assignments
                 min_usage_threshold=0.01):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.diversity_weight = diversity_weight
        self.entropy_weight = entropy_weight
        self.temperature = temperature
        self.min_usage_threshold = min_usage_threshold
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.orthogonal_(self.embed.weight.data)
        self.register_buffer('code_usage', torch.zeros(num_embeddings))
        
    def reinit_unused_codes(self):
        total_usage = self.code_usage.sum()
        usage_prob = self.code_usage / total_usage
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


class MiniGridSeq2SeqTransformerVQ(nn.Module):
    def __init__(self, state_dim, embed_dim, num_heads, num_layers, action_dim, num_embeddings, max_seq_len=20):
        super(MiniGridSeq2SeqTransformerVQ, self).__init__()
        self.embed_dim = embed_dim

        self.state_encoder = nn.Linear(state_dim, embed_dim)
        self.action_embedding = nn.Embedding(action_dim, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=4 * embed_dim, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=4 * embed_dim, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.vq = VectorQuantizer(num_embeddings, embed_dim, beta=1.0)

        self.reconstruct = nn.Linear(embed_dim, state_dim)

    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask.to(device)

    def forward(self, states, actions, target_states=None, sampling_probability=0.5, mask=None):
        B, T, state_dim = states.shape
        states_flattened = states.view(B, T, state_dim)

        state_embeddings = self.state_encoder(states_flattened)
        action_embeddings = self.action_embedding(actions)
        seq_embeddings = state_embeddings + action_embeddings
        seq_embeddings += self.positional_embedding[:, :T, :]

        encoder_output = self.encoder(seq_embeddings, src_key_padding_mask=mask)
        quantized, vq_loss, encoding_indices = self.vq(encoder_output)

        predicted_states = []
        decoder_inputs = torch.zeros((B, T, self.embed_dim), device=states.device)

        if target_states is not None:
            target_embeddings = self.state_encoder(target_states.view(B, T, state_dim))
            tf_count = 0
            for t in range(T - 1):
                if t == 0 or torch.rand(1).item() < sampling_probability:
                    decoder_inputs[:, t, :] = target_embeddings[:, 0, :]
                    tf_count += 1
                else:
                    prev_output = predicted_states[-1]
                    prev_embedding = self.state_encoder(prev_output)
                    decoder_inputs[:, t, :] = prev_embedding

                curr_decoder_input = decoder_inputs[:, :t+1, :] + self.positional_embedding[:, :t+1, :]
                output = self.decoder(
                    curr_decoder_input,
                    quantized[:, :t+1, :],
                    tgt_mask=self.generate_causal_mask(t + 1, states.device),
                    memory_key_padding_mask=mask
                )
                prev_output = self.reconstruct(output[:, -1])
                predicted_states.append(prev_output)
            print(f"Teacher forcing ratio: {tf_count / T}")
        else:
            for t in range(T - 1):
                if t == 0:
                    decoder_inputs[:, 0, :] = state_embeddings[:, 0, :]
                else:
                    prev_output = predicted_states[-1]
                    prev_embedding = self.state_encoder(prev_output)
                    decoder_inputs[:, t, :] = prev_embedding

                curr_decoder_input = decoder_inputs[:, :t+1, :] + self.positional_embedding[:, :t+1, :]
                output = self.decoder(
                    curr_decoder_input,
                    quantized[:, :t+1, :],
                    tgt_mask=self.generate_causal_mask(t + 1, states.device),
                    memory_key_padding_mask=mask
                )
                prev_output = self.reconstruct(output[:, -1])
                predicted_states.append(prev_output)

        predicted_states = torch.stack(predicted_states, dim=1)
        return predicted_states, vq_loss, encoding_indices


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    states, actions, target_states = zip(*batch)

    # Convert states and target_states to padded sequences
    states = [torch.FloatTensor(state) for state in states]
    target_states = [torch.FloatTensor(target) for target in target_states]

    padded_states = pad_sequence(states, batch_first=True, padding_value=0)
    padded_target_states = pad_sequence(target_states, batch_first=True, padding_value=0)

    # Convert actions to padded sequences (ensuring they are integers)
    actions = [np.array(action, dtype=int) for action in actions]
    padded_actions = pad_sequence(
        [torch.LongTensor(action) for action in actions], batch_first=True, padding_value=0
    )

    return padded_states, padded_actions, padded_target_states



def save_model_checkpoint(model, epoch, save_dir="checkpoints"):
    """
    Save the model checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"model_epoch_vq2_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved at {checkpoint_path}")



def train_model(model, dataset, epochs=2000, batch_size=4, lr=1e-4, scheduler_step_size=50, scheduler_gamma=0.9, save_interval=100, collate_fn=None):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    sampling_probability = 0.5

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            states, actions, target_states = batch
            states = states.to(device)
            actions = actions.to(device).long()
            target_states = target_states.to(device)

            # Forward pass
            predicted_frames, vq_loss, indices = model(states, actions, target_states, sampling_probability=sampling_probability)
            reconstruction_loss = criterion(predicted_frames, target_states[:, 1:])
            loss = reconstruction_loss + vq_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate batch loss
            epoch_loss += loss.item()

            if args.log:
                wandb.log({
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "loss": loss.item(),
                    "reconstruction_loss": reconstruction_loss.item(),
                    "vq_loss": vq_loss.item(),
                    "unique_indices": torch.unique(indices).size(0)
                })

            # Logging loss for each batch
            if indices is not None:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Recon Loss: {reconstruction_loss.item():.4f}, VQ Loss: {vq_loss.item()}, Unique Indices: {len(torch.unique(indices))} ")
                metrics = model.vq.get_codebook_metrics()
                # print(f"Active codes: {metrics['active_codes']}")
                # print(f"Codebook entropy: {metrics['codebook_entropy']:.2f}")
                print(f"Average similarity: {metrics['avg_similarity']:.3f}")
            else:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}],  Recon Loss: {reconstruction_loss.item():.4f}, VQ Loss: {vq_loss.item}")

            # Save intermediate predictions every 40 batches

            # if args.log:
            #     metrics = model.vq.get_codebook_metrics()
            #     wandb.log({
            #         "active_codes": metrics["active_codes"],
            #         "codebook_entropy": metrics["codebook_entropy"],
            #         "avg_similarity": metrics["avg_similarity"]
            #         })

            # if batch_idx % 100 == 0:
            #     metrics = model.vq.get_codebook_metrics()
            #     print(f"Active codes: {metrics['active_codes']}")
            #     print(f"Codebook entropy: {metrics['codebook_entropy']:.2f}")
            #     print(f"Average similarity: {metrics['avg_similarity']:.3f}")
            

        # Average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}")

        if epoch % 50 == 0:
            sampling_probability = sampling_probability * 0.9
        
        # if epoch % 5 == 0:
        #     save_frames(states, predicted_frames, epoch, batch_idx, folder_prefix=f"frames_{generate_run_name(args)}")

        # Scheduler step
        scheduler.step()

        # Save the model checkpoint every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            save_model_checkpoint(model, epoch + 1)
            print(f"Model checkpoint saved at epoch {epoch + 1}")



def render_minigrid(agent_pos, agent_orientation, goal_pos):

    import gym
    from gym_minigrid.envs import FourRoomsEnv
    env = FourRoomsEnv(goal_pos=goal_pos)
    obs = env.reset()
    env.agent_pos = agent_pos
    env.agent_dir = agent_orientation
    # env.goal_pos = goal_pos

    # Render the environment
    return env.render(mode='rgb_array')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def cluster_and_visualize_codebook(codebook_vectors, num_clusters=5, method='pca', random_state=42):
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    labels = kmeans.fit_predict(codebook_vectors)
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=random_state)
    reduced_vectors = reducer.fit_transform(codebook_vectors)
    plt.figure(figsize=(8, 6))
    for i in range(num_clusters):
        cluster_points = reduced_vectors[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='x', s=100, label='Centroids')
    plt.legend()
    plt.title('Cluster Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('cluster_visualization.png')



def get_codes(model, dataset, samples = 100):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    codes = []
    count = 0
    for batch in dataloader:
        states, actions, target_states = batch
        states = states.to('cuda')
        actions = actions.to('cuda').long()
        target_states = target_states.to('cuda')

        with torch.no_grad():
            model.eval()
            predicted_frames, vq_loss, indices = model(states, actions, target_states, sampling_probability=1.0)
            codes.append(indices.cpu().numpy())
        count += 1
        if count == samples:
            break
    return codes

import cv2

if __name__=='__main__':
    dataset = MiniGridDataset('minigrid-fourrooms-v0', max_seq_len=20, frame_skip=1)
    state_dim = 7 * 7 * 3
    num_actions = 6
    embed_dim = 32
    num_heads = 4
    num_embeddings = 64
    num_layers = 4

    model = MiniGridSeq2SeqTransformerVQ(state_dim, embed_dim, num_heads, num_layers, num_actions, num_embeddings)
    model.to('cuda')
    model.load_state_dict(torch.load('checkpoints/model_epoch_vq2_475.pth', weights_only=True))
    # cluster_and_visualize_codebook(model.vq.embed.weight.data.cpu().numpy(), num_clusters=5, method='pca')

    # codes = get_codes(model, dataset, samples = 100)
    # print(codes[:3])
    train_model(model, dataset, batch_size=512)