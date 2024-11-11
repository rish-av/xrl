import gym
import d4rl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from utils import visualize_codebook_clusters, cluster_codebook_vectors
from models_new import BeXRL, BeXRLSoftVQ
from utils import render_videos_for_behavior_codes_opencv, get_args, plot_distances, extract_overlapping_segments, extract_non_overlapping_segments
from utils import D4RLDataset



# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. HalfCheetah Dataset
class HalfCheetahDataset(Dataset):
    def __init__(self, env_name='halfcheetah-medium-v2', segment_length=25, num_items=500000):
        # Initialize environment and load dataset
        self.env = gym.make(env_name)
        self.dataset = self.env.get_dataset()
        self.segment_length = segment_length
        
        # Convert observations and actions to tensors
        self.observations = torch.tensor(self.dataset['observations'], dtype=torch.float32)
        self.actions = torch.tensor(self.dataset['actions'], dtype=torch.float32)
        
        # Concatenate observations and actions to create state-action pairs
        self.data = torch.cat([self.observations, self.actions], dim=-1)

    def update_segment_length(self, segment_length):
        self.segment_length = segment_length

    def __len__(self):
        # Calculate the number of segments
        return len(self.data) // self.segment_length

    def __getitem__(self, idx):
        # Retrieve a segment based on the current segment length
        start_idx = idx * self.segment_length
        segment = self.data[start_idx:start_idx + self.segment_length]
        return segment 








# Training step
def train_step(model, data, optimizer, reconstruction_loss_fn):
    model.train()
    optimizer.zero_grad()
    data = data.to(device)
    reconstructed_sequence, vq_loss, encoding_indices = model(data)
    b, s, c  = data.shape

    #seq2seq loss
    # target_segments = data[:, 1:, :]
    # target_segments = target_segments[:, 1: ,:]

    reconstruction_loss = reconstruction_loss_fn(reconstructed_sequence.reshape(b,s,c), data)
    total_loss = reconstruction_loss + vq_loss 

    # Backpropagation
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), reconstruction_loss.item(), vq_loss.item(), encoding_indices

# Training loop


if __name__ == '__main__':

    args = get_args()

    feature_dim = args.feature_dim
    model_dim = args.model_dim
    num_heads = args.num_heads
    num_layers = args.num_layers
    num_embeddings = args.num_embeddings
    segment_length = args.segment_length
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    render = args.render
    weights_path = args.weights_path

    initial_temperature = 1.0
    min_temperature = 0.1
    anneal_rate = 0.99

    


    # Initialize dataset, dataloader, model, optimizer, and loss function
    dataset = HalfCheetahDataset(segment_length=segment_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    init_data = D4RLDataset(segment_length=25)
    init_dataloader = DataLoader(init_data, batch_size=batch_size, shuffle=True)



    # model = BeXRLSoftVQ(feature_dim, 
    #                     model_dim, 
    #                     num_heads, 
    #                     num_layers, 
    #                     num_embeddings, 
    #                     segment_length, 
    #                     initial_temperature, 
    #                     min_temperature, 
    #                     anneal_rate).to(device)


    model = BeXRL(
        feature_dim=feature_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_embeddings=num_embeddings,
        segment_length=segment_length
    ).to(device)

    if weights_path!='':
        model.load_state_dict(torch.load(weights_path))

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    reconstruction_loss_fn = nn.MSELoss()

    if args.train:
        iterator = iter(init_dataloader)
        init_batch = None
        for i in range(10):
            if init_batch is not None:
                init_batch = torch.cat([init_batch, next(iterator)],  axis=0)
            else:
                init_batch = next(iterator)
        print(f'KMeans used data of shape: {init_batch.shape} for vq-vae')


        model.segment_length = 25
        model.initialize_vq_with_kmeans(init_batch.to(device))
        model.segment_length = segment_length

        for epoch in range(num_epochs):

            # if (epoch + 1) % 5 == 0:
            #     segment_length += 5
            #     print(f"Updating segment length to {segment_length}")
                
            #     # Update dataset and dataloader with the new segment length
            #     dataset.update_segment_length(segment_length)
            #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            #     model.segment_length = segment_length
                # model.vq_layer.reset_unused_embeddings(5)

            epoch_loss, epoch_recon_loss, epoch_vq_loss = 0, 0, 0
            unique_idx = 0
            for batch in dataloader:
                total_loss, recon_loss, vq_loss, e_idx = train_step(
                    model, batch, optimizer, reconstruction_loss_fn
                )
                epoch_loss += total_loss
                epoch_recon_loss += recon_loss
                epoch_vq_loss += vq_loss
                # epoch_e_loss+= e_loss
                unique_idx = max(unique_idx, len(torch.unique(e_idx)))

            print(f"Epoch {epoch+1}, Total Loss: {total_loss:.4f}, Recon Loss: {recon_loss:.4f}, VQ Loss: {vq_loss:.4f}, Unique Idx: {unique_idx}")
            # model.vq_layer.reset_unused_embeddings(usage_threshold=5)
            # model.vq_layer.anneal_temperature()
        

        torch.save(model.state_dict(),f'weights_transformer_nembed_{num_embeddings}_seqlen_{segment_length}_softvq.pth')
    codebook_embeddings = model.vq_layer.embedding.weight.data
    num_clusters = 10
    if isinstance(codebook_embeddings, torch.Tensor):
        codebook_embeddings_np = codebook_embeddings.detach().cpu().numpy()
    else:
        codebook_embeddings_np = codebook_embeddings
    clusters, kmeans, center = cluster_codebook_vectors(codebook_embeddings, num_clusters=num_clusters)
    visualize_codebook_clusters(codebook_embeddings_np, clusters, center, method='pca')
    plot_distances(extract_non_overlapping_segments(dataset.dataset, 25, 500),  model.encoder, model.vq_layer)
    
    if render:
        render_videos_for_behavior_codes_opencv(model, 10, segment_length)