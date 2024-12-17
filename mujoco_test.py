import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import argparse
from datetime import datetime
import socket

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='Enable wandb logging')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--initial_tf', type=float, default=0.8)
    parser.add_argument('--tf_decay', type=float, default=0.95)
    return parser.parse_args()

class D4RLStateActionDataset(Dataset):
    def __init__(self, states, actions, context_len=20):
        self.context_len = context_len
        self.data = []

        for i in range(len(states) - context_len - 1):
            seq_states = states[i:i+context_len]
            seq_actions = actions[i:i+context_len]
            target_states = states[i+1:i+1+context_len]
            self.data.append({
                "states": seq_states,
                "actions": seq_actions,
                "targets": target_states
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.data[idx]["states"], dtype=torch.float32),
            "actions": torch.tensor(self.data[idx]["actions"], dtype=torch.float32),
            "targets": torch.tensor(self.data[idx]["targets"], dtype=torch.float32),
        }

def load_d4rl_data(env_name):
    env = gym.make(env_name)
    dataset = env.get_dataset()
    states = dataset['observations']
    actions = dataset['actions']
    return states, actions


class VectorQuantizer(nn.Module):
    def __init__(self, num_tokens, latent_dim, beta):
        
        # Keep existing initialization
        super(VectorQuantizer, self).__init__()
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        self.beta = beta
        self.codebook = nn.Embedding(num_tokens, latent_dim)
        self.register_buffer('usage_count', torch.zeros(num_tokens))
        nn.init.uniform_(self.codebook.weight, -1.0 / num_tokens, 1.0 / num_tokens)

    def forward(self, latent):
        batch_size, seq_len, latent_dim = latent.shape
        latent_flattened = latent.view(-1, latent_dim)

        # Calculate cosine similarity
        latent_normalized = F.normalize(latent_flattened, dim=-1)
        codebook_normalized = F.normalize(self.codebook.weight, dim=-1)
        cosine_sim = torch.matmul(latent_normalized, codebook_normalized.t())
        
        # Original distance calculation for quantization
        distances = (
            torch.sum(latent_flattened ** 2, dim=-1, keepdim=True)
            - 2 * torch.matmul(latent_flattened, self.codebook.weight.t())
            + torch.sum(self.codebook.weight ** 2, dim=-1)
        )
        
        indices = torch.argmin(distances, dim=-1)
        self.usage_count[indices] += 1

        # Get average cosine similarity with chosen codebook entries
        selected_cosine_sim = cosine_sim[torch.arange(len(indices)), indices].mean()
        
        # Rest of the forward pass
        quantized = self.codebook(indices).view(batch_size, seq_len, latent_dim)
        commitment_loss = self.beta * F.mse_loss(latent.detach(), quantized)
        codebook_loss = F.mse_loss(latent, quantized.detach())
        
        # Calculate diversity losses
        similarity_matrix = torch.matmul(codebook_normalized, codebook_normalized.t())
        mask = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device)
        masked_sim = similarity_matrix * (1 - mask)
        
        # Euclidean distances between codebook vectors
        codebook_distances = torch.cdist(self.codebook.weight, self.codebook.weight)
        masked_distances = codebook_distances * (1 - mask)
        avg_euclidean = masked_distances[masked_distances > 0].mean()
        min_euclidean = masked_distances[masked_distances > 0].min()
        
        # Penalize high similarities and small distances
        similarity_penalty = torch.exp(torch.clamp(masked_sim, min=-10, max=10)).mean()
        min_distance = 2.0  # Increased target minimum separation
        distance_penalty = torch.relu(min_distance - masked_distances).mean()
        
        # Combined diversity loss with increased weights
        diversity_loss = similarity_penalty + 0.2 * distance_penalty
        
        quantized = latent + (quantized - latent).detach()

        one_hot = F.one_hot(indices, self.num_tokens).float()
        avg_probs = torch.mean(one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return (quantized, indices, commitment_loss, 
                codebook_loss + 1.0 * diversity_loss, # Increased weight
                perplexity, selected_cosine_sim, 
                avg_euclidean, min_euclidean)


class VQVAE_TeacherForcing(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, num_tokens, hidden_size, n_layers, n_heads, context_len, beta):
        super(VQVAE_TeacherForcing, self).__init__()
        self.state_embedding = nn.Linear(state_dim, hidden_size)
        self.action_embedding = nn.Linear(action_dim, hidden_size)
        self.pos_embedding = nn.Embedding(context_len, hidden_size)
        self.context_len = context_len

        # Layer norms for pre-norm architecture
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.decoder_norm = nn.LayerNorm(hidden_size)
        self.final_norm = nn.LayerNorm(hidden_size)

        # Create causal mask for encoder
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        )

        # Pre-norm encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            norm_first=True  # Enable pre-norm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Projection layers
        self.to_latent = nn.Linear(hidden_size, latent_dim)
        self.vector_quantizer = VectorQuantizer(num_tokens, latent_dim, beta)
        self.from_latent = nn.Linear(latent_dim, hidden_size)

        # Pre-norm decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, 
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            norm_first=True  # Enable pre-norm
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_state = nn.Linear(hidden_size, state_dim)

        # Skip connection projection and weight
        self.skip_projection = nn.Linear(hidden_size, hidden_size)
        self.skip_weight = nn.Parameter(torch.tensor(0.25))  # Initialize skip weight

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
        # Embed states and actions
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)
        combined_emb = state_emb + action_emb

        # Add positional embeddings
        position_ids = torch.arange(states.size(1), device=states.device).unsqueeze(0)
        position_emb = self.pos_embedding(position_ids)
        combined_emb = combined_emb + position_emb

        # Pre-norm and encode
        combined_emb = self.encoder_norm(combined_emb)
        encoded = self.encoder(combined_emb, mask=self.causal_mask)
        
        return encoded, combined_emb  # Return both for skip connection

    def decode(self, quantized, encoder_emb, targets, teacher_forcing_ratio=0.5):
        batch_size, seq_len = targets.shape[0], targets.shape[1]
        
        # Transform quantized vectors back to hidden size
        decoder_input = self.from_latent(quantized)
        
        # Add weighted skip connection
        skip_connection = self.skip_projection(encoder_emb)
        decoder_input = decoder_input + self.skip_weight * skip_connection

        # Initialize with zeros as start token
        current_input = torch.zeros_like(targets[:, 0]).unsqueeze(1)
        outputs = []
        
        for t in range(seq_len):
            # Pre-norm decoder input
            decoder_normed = self.decoder_norm(self.state_embedding(current_input))
            
            # Create causal mask for current timestep
            tgt_mask = torch.triu(
                torch.ones(t+1, t+1, device=targets.device), 
                diagonal=1
            ).bool()
            
            # Decode with pre-norm
            decoder_output = self.decoder(
                decoder_normed,
                decoder_input,
                tgt_mask=tgt_mask
            )
            
            # Final layer norm before prediction
            decoder_output = self.final_norm(decoder_output)
            pred = self.output_state(decoder_output[:, -1:])
            outputs.append(pred)
            
            if t < seq_len - 1:
                if torch.rand(1).item() < teacher_forcing_ratio:
                    next_input = targets[:, t:t+1]
                else:
                    next_input = pred.detach()
                
                current_input = torch.cat([current_input, next_input], dim=1)
        
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def forward(self, states, actions, targets, teacher_forcing_ratio=0.5):
        # Encode and get embeddings for skip connection
        encoded, encoder_emb = self.encode(states, actions)
        
        # Get latent representation
        latent = self.to_latent(encoded)
        
        # Vector quantization - with entropy regularization
        (quantized, indices, commitment_loss, codebook_loss, 
        perplexity, cosine_sim, avg_euclidean, min_euclidean) = self.vector_quantizer(latent)
        
        # Decode with weighted skip connection
        predicted_states = self.decode(quantized, encoder_emb, targets, teacher_forcing_ratio)
        
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(predicted_states, targets)
        
        # Entropy regularization for codebook usage
        avg_probs = torch.mean(F.one_hot(indices, num_classes=self.vector_quantizer.num_tokens).float(), dim=0)
        entropy_loss = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        
        # Total loss
        total_loss = (
            reconstruction_loss +       # State reconstruction
            commitment_loss +           # VQ commitment
            codebook_loss +             # VQ codebook + diversity
            0.01 * entropy_loss         # Encourage diverse codebook usage
        )
        
        return (predicted_states, total_loss, reconstruction_loss, 
                commitment_loss, codebook_loss, perplexity, 
                cosine_sim, avg_euclidean, min_euclidean)



def train_vqvae_teacher_forcing(model, dataloader, optimizer, args):
    if args.log:
        run_name = f"vqvae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="vqvae-training",
            name=run_name,
            config={
                **model.model_config,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "initial_tf": args.initial_tf,
                "tf_decay": args.tf_decay
            },
            entity="mail-rishav9"
        )
        wandb.watch(model)

    teacher_forcing_ratio = args.initial_tf

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_reconstruction = 0
        total_commitment = 0
        total_codebook = 0
        total_perplexity = 0
        total_cosine_sim = 0
        total_euclidean = 0
        total_min_euclidean = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            states = batch['states'].to("cuda")
            actions = batch['actions'].to("cuda")
            targets = batch['targets'].to("cuda")

            # Updated unpacking
            (_, total_loss, reconstruction_loss, commitment_loss, 
             codebook_loss, perplexity, cosine_sim, 
             avg_euclidean, min_euclidean) = model(
                states, actions, targets, teacher_forcing_ratio)

            loss = total_loss  # Now using total_loss directly
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_reconstruction += reconstruction_loss.item()
            total_commitment += commitment_loss.item()
            total_codebook += codebook_loss.item()
            total_perplexity += perplexity.item()
            total_cosine_sim += cosine_sim.item()
            total_euclidean += avg_euclidean.item()
            total_min_euclidean += min_euclidean.item()


            print(f"reconstruction_loss: {reconstruction_loss.item()}, commitment_loss: {commitment_loss.item()}, codebook_loss: {codebook_loss.item()}, perplexity: {perplexity.item()}, cosine_sim: {cosine_sim.item()}, avg_euclidean: {avg_euclidean.item()}, min_euclidean: {min_euclidean.item()}")

            if batch_idx % 1500 == 0:
                torch.save(model.state_dict(), f"weights/vqvae_mujoco_teacher_forcing_weighted_skip_etploss_{epoch}_{batch_idx}.pt")

            if args.log and batch_idx % 10 == 0:
                wandb.log({
                    "batch/total_loss": loss.item(),
                    "batch/reconstruction_loss": reconstruction_loss.item(),
                    "batch/commitment_loss": commitment_loss.item(),
                    "batch/codebook_loss": codebook_loss.item(),
                    "batch/perplexity": perplexity.item(),
                    "batch/teacher_forcing_ratio": teacher_forcing_ratio,
                    "batch/cosine_similarity": cosine_sim.item(),
                    "batch/avg_euclidean": avg_euclidean,
                    "batch/min_euclidean": min_euclidean,
                    "batch/codebook_usage": (model.vector_quantizer.usage_count > 0).sum().item() / model.vector_quantizer.num_tokens,
                })
                
                if batch_idx % 20 == 0:
                    codebook = model.vector_quantizer.codebook.weight.detach()
                    distances = torch.cdist(codebook, codebook)
                    mask = ~torch.eye(len(codebook), dtype=bool, device=codebook.device)
                    distances = distances[mask].cpu().numpy()
                    
                    hist_data = wandb.Histogram(distances)
                    wandb.log({
                        "batch/euclidean_distances": hist_data,
                        "batch/euclidean_distances_mean": distances.mean(),
                        "batch/euclidean_distances_std": distances.std(),
                    })

        print(
            f"Epoch {epoch + 1}, "
            f"Total Loss: {total_loss/len(dataloader):.4f}, "
            f"Reconstruction: {total_reconstruction/len(dataloader):.4f}, "
            f"Commitment: {total_commitment/len(dataloader):.4f}, "
            f"Codebook: {total_codebook/len(dataloader):.4f}, "
            f"Perplexity: {total_perplexity/len(dataloader):.4f}, "
            f"Cosine Sim: {total_cosine_sim/len(dataloader):.4f}, "
            f"Avg Euclidean: {total_euclidean/len(dataloader):.4f}, "
            f"Min Euclidean: {total_min_euclidean/len(dataloader):.4f}, "
            f"Teacher Forcing: {teacher_forcing_ratio:.4f}"
        )

        if args.log:
            wandb.log({
                "epoch/total_loss": total_loss/len(dataloader),
                "epoch/reconstruction": total_reconstruction/len(dataloader),
                "epoch/commitment": total_commitment/len(dataloader),
                "epoch/codebook": total_codebook/len(dataloader),
                "epoch/perplexity": total_perplexity/len(dataloader),
                "epoch/cosine_similarity": total_cosine_sim/len(dataloader),
                "epoch/avg_euclidean": total_euclidean/len(dataloader),
                "epoch/min_euclidean": total_min_euclidean/len(dataloader),
                "epoch/teacher_forcing_ratio": teacher_forcing_ratio
            })

        teacher_forcing_ratio *= args.tf_decay

    if args.log:
        wandb.finish()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def plot_pca_clusters(codebook, n_components=2, n_clusters=10, title="PCA and KMeans Clustering", save_path=None):
    if n_components not in [2, 3]:
        raise ValueError("n_components must be 2 or 3 for visualization.")
    
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(codebook)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced_vectors)
    
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        for cluster in range(n_clusters):
            cluster_points = reduced_vectors[labels == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster+1}", alpha=0.7)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for cluster in range(n_clusters):
            cluster_points = reduced_vectors[labels == cluster]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f"Cluster {cluster+1}", alpha=0.7)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
    
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()



from utils import render_halfcheetah as render_frame
import os
import cv2

import os
os.environ["MUJOCO_GL"] = "osmesa"
import mujoco

def render_frame(env, state):
    env.reset()
    mujoco_model = env.unwrapped.model
    mujoco_data = env.unwrapped.data

    nq = mujoco_model.nq  # Number of position variables
    nv = mujoco_model.nv  # Number of velocity variables

    # Add zero to the initial position to match qpos dimensions
    qpos = np.insert(state[:nq - 1], 0, 0)  # Add zero as the root position
    qvel = state[nq - 1:nq + nv - 1]

    # Update the MuJoCo simulation state
    mujoco_data.qpos[:] = qpos
    mujoco_data.qvel[:] = qvel
    mujoco.mj_forward(mujoco_model, mujoco_data)  # Propagate the state

    # Render the frame
    img = env.render()
    return img


def render_target_and_predicted_frames(env_name, target_states, predicted_states, save_dir="rendered_frames"):
    assert len(target_states) == len(predicted_states), "Target and predicted states must have the same length"
    import gymnasium 
    os.makedirs(save_dir, exist_ok=True)
    target_dir = os.path.join(save_dir, "target_frames")
    predicted_dir = os.path.join(save_dir, "predicted_frames")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(predicted_dir, exist_ok=True)

    env = gymnasium.make('HalfCheetah-v4', render_mode='rgb_array')
    os.environ["MUJOCO_GL"] = "osmesa"  # Force CPU rendering

    for i, (target, predicted) in enumerate(zip(target_states[0], predicted_states[0])):
        target_frame = render_frame(env, target)
        predicted_frame = render_frame(env, predicted)
        cv2.imwrite(os.path.join(target_dir, f"frame_{i:04d}.png"), cv2.cvtColor(target_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(predicted_dir, f"frame_{i:04d}.png"), cv2.cvtColor(predicted_frame, cv2.COLOR_RGB2BGR))

    env.close()




if __name__ == "__main__":
    args = parse_args()
    
    # Load data
    env_name = "halfcheetah-medium-v2"
    states, actions = load_d4rl_data(env_name)
    dataset = D4RLStateActionDataset(states, actions, context_len=20)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model parameters
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    latent_dim = 64
    num_tokens = 128
    hidden_size = 128
    n_layers = 6
    n_heads = 8
    beta = 0.1

    # Create model
    model = VQVAE_TeacherForcing(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        n_layers=n_layers,
        n_heads=n_heads,
        context_len=20,
        beta=beta
    ).to("cuda")
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # train_vqvae_teacher_forcing(model, dataloader, optimizer, args)


    model.eval()
    model.load_state_dict(torch.load('/home/rishav/scratch/xrl/weights/vqvae_mujoco_teacher_forcing_weighted_skip_etploss_0_4500.pt', weights_only=False))

    with torch.no_grad():
        model.eval()
        for batch_idx, batch in enumerate(dataloader):
            states = batch['states'].to("cuda")
            actions = batch['actions'].to("cuda")
            targets = batch['targets'].to("cuda")
            out = model(states, actions, targets, teacher_forcing_ratio=0.0)
            predicted_states = out[0]
            render_target_and_predicted_frames(env_name, targets.cpu().numpy(), predicted_states.cpu().numpy(), save_dir="rendered_frames")
            break

    with torch.no_grad():
        plot_pca_clusters(model.vector_quantizer.codebook.weight.cpu().numpy(), n_components=2, title="Codebook Vectors", save_path="codebook_vectors_check.png")

   
