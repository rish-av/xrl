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
from models.mujoco_models import Seq2SeqMujocoTransformer
from spectral_graph import SpectralGraphPartitioner
import os

from utils.mujoco_utils import D4RLStateActionDataset, load_d4rl_data, create_run_name, D4RLStateActionDatasetNo, parse_args

def train_model(model, dataloader, optimizer, args):
    if args.log:
        run_name = f"{create_run_name(args)}_vqvae_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
        wandb.init(
            project="vqvae-final-icml",
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
            run_name = f"{create_run_name(args)}_vqvae_best_model"
            torch.save(model.state_dict(), f"weights/{run_name}.pt")

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


if __name__ == "__main__":
    args = parse_args()
    context_len = args.context_len
    # Load data
    env_name = "halfcheetah-medium-v2"
    states, actions = load_d4rl_data(env_name)
    if args.dataset_type=='overlap':
        dataset = D4RLStateActionDataset(states, actions, context_len=context_len)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        print("####### creating dataset with no overlap ########")
        dataset = D4RLStateActionDatasetNo(states, actions, context_len=context_len)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
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
    model = Seq2SeqMujocoTransformer(
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
    if args.weights_path:
        weights = torch.load(args.weights_path, weights_only=True)
        model.load_state_dict(weights)

    #check if weights folder exists
    if not os.path.exists("weights"):
        os.makedirs("weights")


    #train the model
    train_model(model, dataloader, optimizer, args)
    with torch.no_grad():
        codebook_vectors = model.vector_quantizer.codebook.weight.cpu().numpy()
        usage_probs = model.vector_quantizer.usage_count / model.vector_quantizer.usage_count.sum()
        active_codes = (usage_probs > 0.001).cpu().numpy()
        active_codebook = codebook_vectors[active_codes]

    with torch.no_grad():
        model.eval()
        all_indices= []
        count = 0
        all_states = []
        all_predicted  = []
        all_targets = []
        all_actions = [] 
        recon_losses = []
        for batch_idx, batch in enumerate(dataloader):
            states = batch['states'].to("cuda")
            actions = batch['actions'].to("cuda")
            targets = batch['targets'].to("cuda")

            # predicted_states, total_loss, reconstruction_loss, commitment_loss, 
            #     codebook_loss, perplexity, cosine_sim, avg_euclidean, min_euclidean
            out = model(states, actions, targets, teacher_forcing_ratio=args.initial_tf)
            latent = model.encode(states, actions)
            quantized, indices, commitment_loss, codebook_loss, perplexity, cosine_sim, avg_euclidean, min_euclidean = model.vector_quantizer(latent)
            indices = indices.reshape(quantized.shape[0], context_len)
            print(f"Indices shape: {indices.shape}")
            for ind in indices.cpu().numpy():
                all_indices.append(list(ind))
                count += 1
            
            print(f"count = {count}")

        

        partitioner = SpectralGraphPartitioner(n_clusters=None, scaling='minmax', similarity_mode='combined', alpha=0.8)
        transition_matrix, unique_ids = partitioner.build_transition_matrix(all_indices)
        node_vectors = codebook_vectors[unique_ids]
        labels = partitioner.fit_transform(node_vectors, transition_matrix)
        labels = partitioner._merge_small_close_clusters(node_vectors, labels, similarity_threshold=0.70, size_threshold=10, min_size=5)
        partitioner.visualize_similarity(node_vectors, transition_matrix)
        partitioner.render_partitioned_graph(node_vectors, transition_matrix, labels, combine_nodes=True, edge_threshold=0.45)




   
