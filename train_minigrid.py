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
import numpy as np
import warnings
np.warnings = warnings

from models.minigrid_models import MiniGridSeq2SeqTransformerVQ
from utils.minigrid_utils import get_sequence_codes
from spectral_graph import SpectralGraphPartitioner
from utils.minigrid_utils import MiniGridSaved, parse_args, save_model_checkpoint
import pickle




args = parse_args()




def train_model(model, dataset, epochs=2000, batch_size=4, lr=1e-4, scheduler_step_size=50, 
                scheduler_gamma=0.9, save_interval=100, collate_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    sampling_probability = 0.0


    print("len of dataloader", len(dataloader))
    def masked_mse_loss(pred, target, mask):
        # mask = mask.unsqueeze(-1).expand_as(pred)
        squared_error = (pred - target) ** 2
        masked_error = squared_error * mask
        loss = masked_error.sum() / (mask.sum() + 1e-8)
        return loss

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            states, actions, target_states, mask = batch
            
            # Move to device after adjustment
            states = states.to(device)
            actions = actions.to(device)
            target_states = target_states.to(device)
            mask = mask.to(device)
            
            # Forward pass
            try:
                predicted_frames, vq_loss, indices = model(states, actions, target_states, 
                                                     sampling_probability=sampling_probability)
            except Exception as e:
                print("Error in forward pass ", e)
                #zero the gradients
                optimizer.zero_grad()
                continue
            
            reconstruction_loss = masked_mse_loss(predicted_frames, target_states, mask)
            loss = reconstruction_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if args.log:
                wandb.log({
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "loss": loss.item(),
                    "reconstruction_loss": reconstruction_loss.item(),
                    "vq_loss": vq_loss.item(),
                    "unique_indices": torch.unique(indices).size(0) if indices is not None else 0,
                    "valid_timesteps": mask.sum().item() / batch_size,
                })

            if indices is not None:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], "
                      f"Recon Loss: {reconstruction_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}, "
                      f"Unique Indices: {len(torch.unique(indices))}, "
                      f"Valid Timesteps: {mask.sum().item() / batch_size:.1f}")
                for k, v in model.vq.get_codebook_metrics().items():
                    print(f"{k}: {v:.4f}")
            else:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], "
                      f"Recon Loss: {reconstruction_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}")

        if epoch % 25 == 0:
            sampling_probability = sampling_probability * 0.9

        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            save_model_checkpoint(model, epoch + 1, name='new_model_32', save_dir="checkpoints_new")
            print(f"Model checkpoint saved at epoch {epoch + 1}")



if __name__=='__main__':
    dataset = MiniGridSaved('/home/ubuntu/xrl/combined_data.pkl', max_seq_len=25)
    state_dim = 7 * 7 * 3
    num_actions = 7
    embed_dim = 128
    num_heads = 2
    num_embeddings = 32
    num_layers = 2

    model = MiniGridSeq2SeqTransformerVQ(state_dim, embed_dim, num_heads, num_layers, num_actions, num_embeddings, max_seq_len=50)
    model.to('cuda')
    model.load_state_dict(torch.load('/home/ubuntu/xrl/checkpoints_new/new_model_32_800.pth', weights_only=True))

    # train_model(model, dataset, lr=1e-4, batch_size=128)

    # cluster codebook vectors with usage > 0.01
    codebook_vectors = model.vq.codebook.weight.data.cpu().numpy()
    usage_probs = model.vq.usage_count / model.vq.usage_count.sum()
    active_codes = (usage_probs > 0.00).cpu().numpy()
    active_codebook = codebook_vectors[active_codes]

    indexes = [i for i, value in enumerate(active_codes) if value]
    codes = get_sequence_codes(model, dataset)
    

    partitioner = SpectralGraphPartitioner(n_clusters=None, scaling='minmax', similarity_mode='combined', alpha=0.4)
    transition_matrix, unique_ids = partitioner.build_transition_matrix(codes)
    node_vectors = codebook_vectors[unique_ids]
    labels = partitioner.fit_transform(node_vectors, transition_matrix)
    labels = partitioner._merge_small_close_clusters(node_vectors, labels, similarity_threshold=0.75, size_threshold=8, min_size=2)
    partitioner.visualize_similarity(node_vectors, transition_matrix)
    partitioner.render_partitioned_graph(node_vectors, transition_matrix, labels, combine_nodes=True, edge_threshold=0.2)
    m_map = {idx: label for idx, label in zip(unique_ids, labels)}
    print(labels, len(unique_ids), m_map)

    from minigrid.wrappers import ImgObsWrapper
    with  open(args.dataset_path, 'rb') as f:
        raw_dataset = pickle.load(f)
    #bifurcate into trajectories based on terminals as in MiniGridDataset
    trajectories = []
    episode_start = 0
    for i in range(len(raw_dataset['terminals'])):
        if raw_dataset['terminals'][i] or i == len(raw_dataset['terminals']) - 1:
            if (i + 1) - episode_start > 1:  # Need at least 2 states
                episode = {}
                for k,v in raw_dataset.items():
                    if k == 'actions':
                        episode[k] = v[episode_start+1:i+1] + [np.array(0)]  # Last action is always 0
                    else:
                        episode[k] = v[episode_start:i+1]
                episode_start = i + 1
                trajectories.append(episode)






    





