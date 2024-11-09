import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader, Dataset
import d4rl  # D4RL library
import gym   # OpenAI Gym



class Config:
    # Model parameters
    state_dim = 17  # HalfCheetah state dimension
    action_dim = 6  # HalfCheetah action dimension
    model_dim = 128
    num_layers = 4
    num_heads = 4
    dim_feedforward = 256
    dropout = 0.1
    max_seq_len = 100  # Maximum sequence length

    # VQ-VAE parameters
    num_embeddings = 64  # Size of the codebook
    commitment_cost = 0.25  # Beta in VQ-VAE loss

    # Training parameters
    batch_size = 256
    learning_rate = 1e-4
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize the environment and dataset
env = gym.make('halfcheetah-medium-v2')
dataset = d4rl.qlearning_dataset(env)

# Custom Dataset class
class HalfCheetahDataset(Dataset):
    def __init__(self, dataset, seq_len):
        self.states = dataset['observations'][:256*101]
        self.actions = dataset['actions'][:256*101]
        self.seq_len = seq_len
        self.indices = self.create_indices()

    def create_indices(self):
        # Create indices for sequences of length seq_len
        num_samples = len(self.states)
        indices = []
        for i in range(0, num_samples - self.seq_len):
            indices.append(i)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len
        states = self.states[start_idx:end_idx]
        actions = self.actions[start_idx:end_idx]
        return torch.FloatTensor(states), torch.FloatTensor(actions)

# Create dataset and dataloader
seq_len = Config.max_seq_len
dataset = HalfCheetahDataset(dataset, seq_len)
data_loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, model_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Shape: [max_seq_len, 1, model_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x


class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=1.0, decay=0.99, epsilon=1e-5):
        super(VQEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Initialize embeddings
        embedding = torch.randn(self.num_embeddings, self.embedding_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embedding_avg', embedding.clone())

    def forward(self, inputs):
        # Flatten input
        inputs = inputs.reshape(-1, self.embedding_dim)

        # Compute distances
        distances = (torch.sum(inputs ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding ** 2, dim=1)
                     - 2 * torch.matmul(inputs, self.embedding.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(inputs.dtype)

        # Quantize
        quantized = torch.matmul(encodings, self.embedding)

        # Loss
        commitment_loss = F.mse_loss(inputs, quantized.detach())
        loss = self.commitment_cost * commitment_loss

        # EMA updates
        if self.training:
            encodings_sum = encodings.sum(0)
            embedding_sum = torch.matmul(encodings.t(), inputs)

            self.cluster_size = self.cluster_size * self.decay + encodings_sum * (1 - self.decay)
            self.embedding_avg = self.embedding_avg * self.decay + embedding_sum * (1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon)) * n

            self.embedding = self.embedding_avg / cluster_size.unsqueeze(1)

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Reshape quantized
        quantized = quantized.view_as(inputs)
        encoding_indices = encoding_indices.view(*inputs.shape[:-1])

        return quantized, loss, encoding_indices



class CausalTransformerEncoder(nn.Module):
    def __init__(self, model_dim, num_layers, num_heads, dim_feedforward, dropout, max_seq_len):
        super(CausalTransformerEncoder, self).__init__()
        self.model_dim = model_dim
        self.positional_encoding = PositionalEncoding(model_dim, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src, src_mask=None):
        src = self.positional_encoding(src)
        if src_mask is None:
            seq_len = src.size(0)
            src_mask = generate_square_subsequent_mask(seq_len).to(src.device)
        output = self.transformer(src, mask=src_mask)
        return output  # Shape: [seq_len, batch_size, model_dim]

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask


class CausalTransformerDecoder(nn.Module):
    def __init__(self, model_dim, num_layers, num_heads, dim_feedforward, dropout, action_dim, max_seq_len):
        super(CausalTransformerDecoder, self).__init__()
        self.model_dim = model_dim
        self.positional_encoding = PositionalEncoding(model_dim, max_seq_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, action_dim)

    def forward(self, tgt, memory, tgt_mask=None):
        tgt = self.positional_encoding(tgt)
        if tgt_mask is None:
            seq_len = tgt.size(0)
            tgt_mask = generate_square_subsequent_mask(seq_len).to(tgt.device)
        output = self.transformer(tgt, memory, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output  # Shape: [seq_len, batch_size, action_dim]


class TransformerVQVAE(nn.Module):
    def __init__(self, config):
        super(TransformerVQVAE, self).__init__()
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.model_dim = config.model_dim

        # Embedding layers
        self.state_embedding = nn.Linear(self.state_dim, self.model_dim)
        self.action_embedding = nn.Linear(self.action_dim, self.model_dim)

        # Encoder and Decoder
        self.encoder = CausalTransformerEncoder(
            model_dim=config.model_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len
        )
        self.vq_layer = VQEmbedding(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.model_dim,
            commitment_cost=config.commitment_cost,
            decay=0.99,
            epsilon=1e-5
        )
        self.decoder = CausalTransformerDecoder(
            model_dim=config.model_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            action_dim=config.action_dim,
            max_seq_len=config.max_seq_len
        )

    def forward(self, states, actions):
        # states, actions: [batch_size, seq_len, dim]
        batch_size, seq_len, _ = states.shape

        # Embed states and actions
        states = states.to(self.state_embedding.weight.device)
        actions = actions.to(self.action_embedding.weight.device)
        state_embeds = self.state_embedding(states)  # [batch_size, seq_len, model_dim]
        action_embeds = self.action_embedding(actions)  # [batch_size, seq_len, model_dim]

        # Combine embeddings
        encoder_inputs = state_embeds + action_embeds  # [batch_size, seq_len, model_dim]
        encoder_inputs = encoder_inputs.permute(1, 0, 2)  # [seq_len, batch_size, model_dim]

        # Encoder
        encoder_outputs = self.encoder(encoder_inputs)  # [seq_len, batch_size, model_dim]

        # VQ-VAE per timestep
        quantized_outputs, vq_loss, indices = self.vq_layer(encoder_outputs.permute(1, 0, 2))  # [batch_size, seq_len, model_dim]

        s,b,m = encoder_inputs.shape
        quantized_outputs = quantized_outputs.reshape(s,b,m)  # [seq_len, batch_size, model_dim]

        # Prepare decoder inputs
        decoder_inputs = self.action_embedding(actions)  # [batch_size, seq_len, model_dim]
        decoder_inputs = decoder_inputs.permute(1, 0, 2)  # [seq_len, batch_size, model_dim]

        # Decoder
        decoder_outputs = self.decoder(decoder_inputs, quantized_outputs)  # [seq_len, batch_size, action_dim]

        return decoder_outputs, vq_loss, indices


def compute_loss(decoder_outputs, target_actions, vq_loss):
    # decoder_outputs: [seq_len, batch_size, action_dim]
    # target_actions: [batch_size, seq_len, action_dim]
    target_actions = target_actions.permute(1, 0, 2)  # [seq_len, batch_size, action_dim]

    # Reconstruction loss (MSE loss)
    reconstruction_loss = F.mse_loss(decoder_outputs, target_actions)

    # Total loss
    total_loss = reconstruction_loss + vq_loss

    return total_loss, reconstruction_loss.item(), vq_loss.item()


# Initialize model and optimizer
config = Config()
model = TransformerVQVAE(config).to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Training loop
for epoch in range(config.num_epochs):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    for batch_idx, (states, actions) in enumerate(data_loader):
        states = states.to(config.device)  # [batch_size, seq_len, state_dim]
        actions = actions.to(config.device)  # [batch_size, seq_len, action_dim]

        optimizer.zero_grad()

        # Forward pass
        decoder_outputs, vq_loss, _ = model(states, actions)

        # Compute loss
        loss, recon_loss, vq_loss_value = compute_loss(decoder_outputs, actions, vq_loss)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Logging
        total_loss += loss.item()
        total_recon_loss += recon_loss
        total_vq_loss += vq_loss_value

        if batch_idx % 20 == 0:
            print(f"Epoch [{epoch+1}/{config.num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], "
                  f"Loss: {loss.item():.4f}, Recon Loss: {recon_loss:.4f}, VQ Loss: {vq_loss_value:.4f}")

    avg_loss = total_loss / len(data_loader)
    avg_recon_loss = total_recon_loss / len(data_loader)
    avg_vq_loss = total_vq_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{config.num_epochs}] Completed. "
          f"Avg Loss: {avg_loss:.4f}, Avg Recon Loss: {avg_recon_loss:.4f}, Avg VQ Loss: {avg_vq_loss:.4f}")



import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_encoding_indices(encoding_indices, sequence_name="Sequence"):
    """
    Plots the encoding indices of a sequence over time.

    Args:
        encoding_indices (numpy.ndarray or torch.Tensor): A 1D array or tensor containing encoding indices.
        sequence_name (str): A name or title for the sequence being plotted.
    """
    # Convert to numpy array if input is a tensor
    if isinstance(encoding_indices, torch.Tensor):
        encoding_indices = encoding_indices.cpu().numpy()

    # Ensure encoding_indices is a 1D array
    encoding_indices = encoding_indices.flatten()
    timesteps = np.arange(len(encoding_indices))

    # Create the plot
    plt.figure(figsize=(12, 4))
    plt.plot(timesteps, encoding_indices, marker='o', linestyle='-')
    plt.title(f'Encoding Indices Over Time for {sequence_name}')
    plt.xlabel('Timestep')
    plt.ylabel('Encoding Index')
    plt.grid(True)
    
    # Annotate each point with the encoding index
    for i, (timestep, code_index) in enumerate(zip(timesteps, encoding_indices)):
        plt.annotate(f'{int(code_index)}', xy=(timestep, code_index), xytext=(0, 10),
                     textcoords='offset points', ha='center', fontsize=9)
    plt.savefig('action_transformer.png')
    plt.show()

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_codebook_clusters(codebook_embeddings, clusters, method='pca'):
    """
    Visualizes the codebook clusters in 2D using PCA or t-SNE.

    Args:
        codebook_embeddings (np.ndarray): Codebook embeddings of shape [num_codes, embedding_dim].
        clusters (np.ndarray): Cluster labels for each codebook vector.
        method (str): Dimensionality reduction method ('pca' or 'tsne').

    Returns:
        None
    """
    if isinstance(codebook_embeddings, torch.Tensor):
        codebook_embeddings = codebook_embeddings.detach().cpu().numpy()

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(codebook_embeddings)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(codebook_embeddings)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")

    # Create the plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='tab10', alpha=0.7)

    # Annotate each point with the code index
    for i, (x, y) in enumerate(embeddings_2d):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

    plt.colorbar(scatter, ticks=range(np.max(clusters)+1))
    plt.title(f'Codebook Clusters Visualized using {method.upper()} with Code Indices')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()
    plt.savefig('codebookcluster.png')


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import torch

def cluster_codebook_vectors(codebook_embeddings, num_clusters=10, random_state=42):
    """
    Clusters the codebook vectors using K-Means clustering.

    Args:
        codebook_embeddings (torch.Tensor or np.ndarray): Codebook embeddings of shape [num_codes, embedding_dim].
        num_clusters (int): The number of clusters to form.
        random_state (int): Random seed for reproducibility.

    Returns:
        clusters (np.ndarray): An array of cluster labels for each codebook vector.
        kmeans (KMeans): The fitted KMeans object for further analysis.
    """
    # Convert to numpy array if input is a tensor
    if isinstance(codebook_embeddings, torch.Tensor):
        codebook_embeddings = codebook_embeddings.detach().cpu().numpy()

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(codebook_embeddings)

    return clusters, kmeans


codebook_embeddings = model.vq_layer.embedding 
num_clusters = 10
clusters, kmeans = cluster_codebook_vectors(codebook_embeddings, num_clusters=num_clusters)

# Convert codebook_embeddings to numpy array if necessary
if isinstance(codebook_embeddings, torch.Tensor):
    codebook_embeddings_np = codebook_embeddings.detach().cpu().numpy()
else:
    codebook_embeddings_np = codebook_embeddings

# Visualize using PCA
visualize_codebook_clusters(codebook_embeddings_np, clusters, method='pca')

# Visualize using t-SNE
# visualize_codebook_clusters(codebook_embeddings_np, clusters, method='tsne')




# Assume 'model' is your trained TransformerVQVAE model
# and you have a batch of 'states' and 'actions'

model.eval()
with torch.no_grad():
    # Forward pass to get decoder outputs, VQ loss, and encoding indices
    decoder_outputs, vq_loss, encoding_indices = model(states, actions)  # encoding_indices shape: [batch_size, seq_len]
    b = states.shape[0]
    # Select a sequence from the batch (e.g., the first sequence)
    encoding_indices = encoding_indices.reshape(b, config.max_seq_len)
    # sequence_index = 0
    # sequence_encoding_indices = encoding_indices[sequence_index]  # Shape: [seq_len]

    # Plot the encoding indices for the selected sequence
    plot_encoding_indices(encoding_indices[0], sequence_name=f"Sequence {0}")
