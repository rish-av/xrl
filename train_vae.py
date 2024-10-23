import torch
import torch.nn as nn
import torch.optim as optim
import d4rl  # D4RL dataset library
import gym
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class LSTMTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(LSTMTrajectoryEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = h_n[-1]  # Take the hidden state of the last LSTM layer
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

class LSTMTrajectoryDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, sequence_length, num_layers=1):
        super(LSTMTrajectoryDecoder, self).__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.sequence_length = sequence_length

    def forward(self, z):
        hidden = self.latent_to_hidden(z).unsqueeze(1).repeat(1, self.sequence_length, 1)
        lstm_out, _ = self.lstm(hidden)
        out = self.fc_out(lstm_out)
        return out

class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, sequence_length, num_layers=1):
        super(LSTMVAE, self).__init__()
        self.encoder = LSTMTrajectoryEncoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = LSTMTrajectoryDecoder(latent_dim, hidden_dim, input_dim, sequence_length, num_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar, z

def vae_loss(reconstructed, original, mu, logvar):
    reconstruction_loss = nn.MSELoss()(reconstructed, original)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence

def load_halfcheetah_data(sequence_length=25):
    env = gym.make('halfcheetah-medium-v2')  # D4RL environment
    dataset = env.get_dataset()
    observations = dataset['observations']
    num_sequences = len(observations) // sequence_length
    trajectory_data = np.array([
        observations[i * sequence_length: (i + 1) * sequence_length]
        for i in range(num_sequences)
    ])
    
    return trajectory_data

def train_vae(vae, data_loader, num_epochs=50, lr=0.001):
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    vae.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            trajectories = batch[0]  # Unpack the first element (the tensor) from the tuple
            trajectories = trajectories.float()  # Ensure it is a float tensor
            mean = trajectories.mean(dim=[0, 1], keepdim=True)
            std = trajectories.std(dim=[0, 1], keepdim=True)
            normalized_trajectories = (trajectories - mean) / (std + 1e-8)  # Adding a small value to avoid division by zero
            optimizer.zero_grad()
            reconstructed, mu, logvar, _ = vae(normalized_trajectories)
            loss = vae_loss(reconstructed, normalized_trajectories, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 10.0)
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss/len(data_loader)}')

    print("VAE Training Complete")

def collect_latent_variables(vae, data_loader):
    vae.eval()
    latent_vectors = []
    
    with torch.no_grad():
        for trajectories in data_loader:
            trajectories = trajectories[0].float()
            _, _, _, z = vae(trajectories)
            latent_vectors.append(z.cpu().numpy())
    
    return np.vstack(latent_vectors)

def perform_clustering(latent_vectors, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(latent_vectors)
    return labels, kmeans

def plot_clusters(latent_vectors, labels, method="tsne"):
    """
    Visualizes the clusters of latent vectors in 2D space using t-SNE or PCA.
    :param latent_vectors: Latent vectors to be clustered
    :param labels: Cluster labels
    :param method: Dimensionality reduction method ('tsne' or 'pca')
    """
    if method == "tsne":
        # Use t-SNE to reduce latent vectors to 2D
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        reduced_latent = tsne.fit_transform(latent_vectors[:500])
    elif method == "pca":
        # Use PCA to reduce latent vectors to 2D
        pca = PCA(n_components=2)
        reduced_latent = pca.fit_transform(latent_vectors)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_latent[:, 0], reduced_latent[:, 1], c=labels, cmap='tab10', s=50)
    plt.colorbar(scatter)
    plt.title(f'Clusters Visualized using {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()
    plt.savefig('clusters_vae.png')

def main():
    sequence_length = 25
    input_dim = 17  
    hidden_dim = 128 
    latent_dim = 10  
    num_layers = 1  
    batch_size = 256
    num_epochs = 15
    num_clusters = 5 
    trajectory_data = load_halfcheetah_data(sequence_length=sequence_length)
    trajectory_dataset = TensorDataset(torch.tensor(trajectory_data))
    data_loader = DataLoader(trajectory_dataset, batch_size=batch_size, shuffle=True)
    vae = LSTMVAE(input_dim, hidden_dim, latent_dim, sequence_length, num_layers)
    train_vae(vae, data_loader, num_epochs=num_epochs)
    latent_vectors = collect_latent_variables(vae, data_loader)
    labels, kmeans = perform_clustering(latent_vectors, num_clusters=num_clusters)
    plot_clusters(latent_vectors, labels, method="tsne")

if __name__ == "__main__":
    main()