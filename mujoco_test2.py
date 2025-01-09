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



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='Enable wandb logging')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--initial_tf', type=float, default=0.5)
    parser.add_argument('--tf_decay', type=float, default=0.85)
    parser.add_argument('--tag', type=str, default="vqvae")
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--num_tokens', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--context_len', type=int, default=50)
    parser.add_argument('--dataset_type', type=str, default='overlap', help='Overlap sequences')

    return parser.parse_args()


class D4RLStateActionDatasetNo(Dataset):
    def __init__(self, states, actions, context_len=20):
        self.context_len = context_len
        self.data = []

        # Step by context_len instead of 1 to avoid overlap
        for i in range(0, len(states) - context_len - 1, context_len):
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






class VectorQuantizerSTE(nn.Module):
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

        # Affine reparameterization parameters
        self.c_mean = nn.Parameter(torch.zeros(latent_dim))
        self.c_std = nn.Parameter(torch.ones(latent_dim))

    def anneal_temperature(self):
        self.temperature = torch.max(
            torch.tensor(self.temp_min, device=self.temperature.device),
            self.temperature * np.exp(-self.anneal_rate)
        )

    def forward(self, latent):
        batch_size, seq_len, latent_dim = latent.shape
        flat_input = latent.reshape(-1, latent_dim)

        # Reparameterize codebook
        reparameterized_codebook = self.c_mean + self.c_std * self.codebook.weight

        # Calculate cosine similarity
        latent_normalized = F.normalize(flat_input, dim=-1)
        codebook_normalized = F.normalize(reparameterized_codebook, dim=-1)
        cosine_sim = torch.matmul(latent_normalized, codebook_normalized.t())

        # Calculate distances for quantization
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True) 
            - 2 * torch.matmul(flat_input, reparameterized_codebook.t())
            + torch.sum(reparameterized_codebook ** 2, dim=1)
        )

        # Apply temperature scaling to distances
        scaled_distances = distances / max(self.temperature.item(), 1e-5)

        # Softmax with temperature for stochastic relaxation
        soft_assign = F.softmax(-scaled_distances, dim=-1)

        # Sample indices probabilistically
        indices = torch.multinomial(soft_assign, 1).squeeze(-1)
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
        codebook_distances = torch.cdist(reparameterized_codebook, reparameterized_codebook)
        mask = ~torch.eye(codebook_distances.shape[0], dtype=bool, device=codebook_distances.device)
        masked_distances = codebook_distances[mask]
        avg_euclidean = masked_distances.mean()
        min_euclidean = masked_distances.min()

        # Quantize
        quantized = torch.matmul(assign, reparameterized_codebook)
        quantized = quantized.view(batch_size, seq_len, latent_dim)

        # Loss
        commitment_loss = self.beta * F.mse_loss(latent.detach(), quantized)
        codebook_loss = F.mse_loss(latent, quantized.detach())

        # Straight-through estimator
        quantized = latent + (quantized - latent).detach()

        # Calculate perplexity
        avg_probs = torch.mean(hard_assign, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Gradient gap regularization
        non_quantized_grad = flat_input.clone().detach()
        quantized_grad = quantized.clone().detach()
        gradient_gap = torch.norm(non_quantized_grad - quantized_grad)

        return (quantized, indices, commitment_loss, codebook_loss, perplexity, selected_cosine_sim, avg_euclidean, min_euclidean, gradient_gap)




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


class VQVAE_TeacherForcing(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, num_tokens, hidden_size, n_layers, n_heads, context_len, beta,
                 temp_init=1.0, temp_min=0.1, anneal_rate=0.00003, ema_decay=0.99):
        super().__init__()

        # Embedding and normalization layers
        self.state_embedding = nn.Sequential(
            # nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_size)
        )
        self.action_embedding = nn.Sequential(
            # nn.LayerNorm(action_dim),
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
            # nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, latent_dim)
        )

        # Vector Quantizer
        self.vector_quantizer = VectorQuantizer(
            num_tokens, latent_dim, beta, 
        )

        # Decoder layers
        self.from_latent = nn.Sequential(
            # nn.LayerNorm(latent_dim),
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



def create_run_name(args):
    # Create run name based on arguments, use all items in args (argparse object)
    run_name = "_".join([f"{k}={v}" for k, v in vars(args).items()])
    return run_name


def train_vqvae_teacher_forcing(model, dataloader, optimizer, args):
    if args.log:
        run_name = f"{create_run_name(args)}_vqvae_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
        wandb.init(
            project="vqvae-final",
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

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    teacher_forcing_ratio = args.initial_tf
    best_loss = float('inf')

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

            (_, loss, reconstruction_loss, commitment_loss, 
             codebook_loss, perplexity, cosine_sim, 
             avg_euclidean, min_euclidean) = model(
                states, actions, targets, teacher_forcing_ratio)

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

            if batch_idx % 3500 == 0:
                name = create_run_name(args)
                torch.save(model.state_dict(), f"{name}_{epoch}_{batch_idx}.pt")

            if batch_idx > 0 and batch_idx % 20000 == 0:
                teacher_forcing_ratio *= args.tf_decay
            if args.log:
                # Existing wandb logging code...
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
                    "batch/learning_rate": optimizer.param_groups[0]['lr']
                })
                
                if batch_idx % 200 == 0:
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

        # Calculate average epoch loss
        avg_epoch_loss = total_loss / len(dataloader)
        
        # Step the scheduler based on average epoch loss
        scheduler.step(avg_epoch_loss)
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), "weights/vqvae_best_model.pt")

        print(
            f"Epoch {epoch + 1}, "
            f"Total Loss: {avg_epoch_loss:.4f}, "
            f"Reconstruction: {total_reconstruction/len(dataloader):.4f}, "
            f"Commitment: {total_commitment/len(dataloader):.4f}, "
            f"Codebook: {total_codebook/len(dataloader):.4f}, "
            f"Perplexity: {total_perplexity/len(dataloader):.4f}, "
            f"Cosine Sim: {total_cosine_sim/len(dataloader):.4f}, "
            f"Avg Euclidean: {total_euclidean/len(dataloader):.4f}, "
            f"Min Euclidean: {total_min_euclidean/len(dataloader):.4f}, "
            f"Teacher Forcing: {teacher_forcing_ratio:.4f}, "
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if args.log:
            wandb.log({
                "epoch/total_loss": avg_epoch_loss,
                "epoch/reconstruction": total_reconstruction/len(dataloader),
                "epoch/commitment": total_commitment/len(dataloader),
                "epoch/codebook": total_codebook/len(dataloader),
                "epoch/perplexity": total_perplexity/len(dataloader),
                "epoch/cosine_similarity": total_cosine_sim/len(dataloader),
                "epoch/avg_euclidean": total_euclidean/len(dataloader),
                "epoch/min_euclidean": total_min_euclidean/len(dataloader),
                "epoch/teacher_forcing_ratio": teacher_forcing_ratio,
                "epoch/learning_rate": optimizer.param_groups[0]['lr']
            })

        teacher_forcing_ratio *= args.tf_decay

    if args.log:
        wandb.finish()



import warnings
np.warnings = warnings

if __name__ == "__main__":
    args = parse_args()
    context_len = args.context_len
    # Load data
    env_name = "halfcheetah-medium-v2"
    states, actions = load_d4rl_data(env_name)
    if args.dataset_type=='overlap':
        dataset = D4RLStateActionDataset(states, actions, context_len=context_len)
    else:
        print("####### creating dataset with no overlap ########")
        dataset = D4RLStateActionDatasetNo(states, actions, context_len=context_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model parameters
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    latent_dim = args.latent_dim
    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    n_layers = args.n_layers
    n_heads = args.n_heads
    beta = args.beta    

    # Create model
    model = VQVAE_TeacherForcing(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        n_layers=n_layers,
        n_heads=n_heads,
        context_len=context_len,
        beta=beta
    ).to("cuda")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # weights = torch.load('log=True_lr=0.0001_batch_size=32_epochs=10_initial_tf=0.4_tf_decay=0.85_tag=vqvae_latent_dim=128_num_tokens=128_hidden_size=128_n_layers=4_n_heads=8_beta=1.0_context_len=30_dataset_type=overlap_0_3500.pt', weights_only=True)


    # # for k,v in weights.items():
    # #     if k in model.state_dict() and v.shape == model.state_dict()[k].shape:
    # #         model.state_dict()[k].copy_(v)
    # model.load_state_dict(weights)
    train_vqvae_teacher_forcing(model, dataloader, optimizer, args)

    from utils import build_transition_matrix, mcl_graph_cut, visualize_clusters

    from utils import render_target_and_predicted_frames, plot_pca_clusters, build_cluster_transition_graph, analyze_quantized_sequences2

    with torch.no_grad():
    
        model.eval()
        all_indices= []
        count = 0
        all_states = []
        for batch_idx, batch in enumerate(dataloader):
            states = batch['states'].to("cuda")
            actions = batch['actions'].to("cuda")
            targets = batch['targets'].to("cuda")
            out = model(states, actions, targets, teacher_forcing_ratio=args.initial_tf)

            for state_seq in states:
                curr_seq_states = []
                for state in state_seq:
                    curr_seq_states.append(state.cpu().numpy())
                all_states.append(curr_seq_states)


            latent = model.encode(states, actions)
            quantized, indices, commitment_loss, codebook_loss, perplexity, cosine_sim, avg_euclidean, min_euclidean = model.vector_quantizer(latent)
            indices = indices.reshape(quantized.shape[0], context_len)
            for ind in indices.cpu().numpy():
                all_indices.append(list(ind))
                count += 1
            
            # if count == 5000:
            #     break
        transition_matrix = build_transition_matrix(all_indices)
        clusters = mcl_graph_cut(model.vector_quantizer.codebook.weight.cpu().numpy(), transition_matrix)
        fig = visualize_clusters(model.vector_quantizer.codebook.weight.cpu().numpy(), clusters, transition_matrix)
        build_cluster_transition_graph(model.vector_quantizer.codebook.weight.cpu().numpy(), all_indices, 10, 20, plot_path="cluster_transition_graph.png")
        analyze_quantized_sequences2(all_indices, model.vector_quantizer.codebook.weight.cpu().numpy(), 10, distance_weight=3, transition_weight=0.5,  save_dir= "yoyo")

        # from utils import get_partition_labels, save_cluster_frames, get_states_by_cluster
        # sequence_labels, partition_labels = get_partition_labels(
        #     all_indices,
        #     model.vector_quantizer.codebook.weight.cpu().numpy(),
        #     n_cuts=10
        # )
        # cluster_states = get_states_by_cluster(all_states, sequence_labels)
        # env = gym.make(env_name) 

        # save_cluster_frames(env, cluster_states, save_dir="cluster_frames", n_frames=100)
    with torch.no_grad():
        plot_pca_clusters(model.vector_quantizer.codebook.weight.cpu().numpy(), n_components=2, title="Codebook Vectors", save_path="codebook_vectors_check.png")

   
