import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import d4rl
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 1e-3
epochs = 10
batch_size = 64
seq_len = 50 
latent_dim = 32 
hidden_dim = 128 
temperature = 0.1  
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


env = gym.make('halfcheetah-medium-replay-v2')
dataset = env.get_dataset()

class TrajectoryEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim):
        super(TrajectoryEncoder, self).__init__()
        self.lstm = nn.LSTM(state_dim + action_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, traj):
        lstm_out, _ = self.lstm(traj)  
        latent = self.fc(lstm_out[:, -1, :])  
        return latent

def contrastive_loss(positive_pairs, negative_pairs, temperature=0.1):
    positive_sim = F.cosine_similarity(positive_pairs[0], positive_pairs[1])
    negative_sim = torch.cat([F.cosine_similarity(positive_pairs[0], neg) for neg in negative_pairs], dim=0)
    
    positive_exp = torch.exp(positive_sim / temperature)
    negative_exp = torch.exp(negative_sim / temperature).sum()
    
    loss = -torch.log(positive_exp / (positive_exp + negative_exp))
    return loss.mean()


def process_trajectories(dataset, seq_len):
    states = dataset['observations']
    actions = dataset['actions']
    num_items = len(states)
    
    trajectories = []
    for i in range(0, num_items - seq_len, seq_len):
        state_seq = states[i:i+seq_len]
        action_seq = actions[i:i+seq_len]
        traj = torch.tensor(np.concatenate([state_seq, action_seq], axis=-1), dtype=torch.float32)
        trajectories.append(traj.to(device))
    return trajectories

trajectories = process_trajectories(dataset, seq_len)


def data_loader(trajectories, batch_size):
    for i in range(0, len(trajectories), batch_size):
        yield torch.stack(trajectories[i:i+batch_size])


def train_contrastive_model(encoder, optimizer, epochs, batch_size, trajectories):
    encoder.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader(trajectories, batch_size):
            optimizer.zero_grad()
            positive_pairs = (encoder(batch[:, :seq_len//2]), encoder(batch[:, seq_len//2:]))
            negative_pairs = [encoder(torch.roll(batch, shifts=i, dims=0)) for i in range(1, batch_size)]
            
            
            loss = contrastive_loss(positive_pairs, negative_pairs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(trajectories):.4f}')


state_dim = 17  
action_dim = 6  
encoder = TrajectoryEncoder(state_dim, action_dim, latent_dim, hidden_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)


train_contrastive_model(encoder, optimizer, epochs, batch_size, trajectories)
encoder.eval()
with torch.no_grad():
    trajectory_embeddings = [encoder(traj.unsqueeze(0)).cpu().numpy() for traj in trajectories]
    trajectory_embeddings = np.vstack(trajectory_embeddings)


num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(trajectory_embeddings)
plt.figure(figsize=(10, 6))
for cluster_id in range(num_clusters):
    cluster_points = trajectory_embeddings[clusters == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')
plt.title('Behavior Segmentation via Contrastive Learning')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend()
plt.show()
plt.savefig('contrastive.png')