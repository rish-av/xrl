import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import d4rl
import gym
from models import EncoderLSTM, DecoderLSTM, Seq2Seq

from utils import sample_trajectories, get_representation



env = gym.make('halfcheetah-medium-replay-v2')
dataset = env.get_dataset()
num_items = 16000
states = dataset['observations'][:num_items]
actions = dataset['actions'][:num_items]
next_states = dataset['next_observations'][:num_items]
rewards = dataset['rewards'][:num_items]
dones = dataset['terminals'][:num_items]



input_dim = states.shape[1] + actions.shape[1]
hidden_dim = 128
output_dim = states.shape[1]
num_layers = 4
seq_len = 32
batch_size = 2 
encoder = EncoderLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
decoder = DecoderLSTM(hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Seq2Seq(encoder, decoder).to(device)
model.load_state_dict(torch.load('/home/rishav/scratch/xrl/seq2seq_32.pth'))


with torch.no_grad():
    trajs = sample_trajectories(states, actions, rewards, next_states, dones)
    
    st_idx = 0
    reps = []
    rep_key = []
    while st_idx < 10000:
        for s_len in range(4 , 32, 4): 
            stat = torch.tensor(trajs[0]['states'][st_idx:st_idx+s_len], dtype=torch.float32)
            act = torch.tensor(trajs[0]['actions'][st_idx:st_idx+s_len], dtype=torch.float32)
            seq = torch.cat([stat, act], dim=-1).unsqueeze(0).to(device)
            rep = get_representation(model, seq).reshape(-1) 
            rep_key.append(s_len) 
            reps.append(rep)
        
        st_idx += 32 
    reps = torch.stack(reps).cpu().numpy()
    rep_key = np.array(rep_key)

    num_clusters = 4
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(reps)
    centroids = kmeans.cluster_centers_

    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(reps)
    reduced_centroids = pca.transform(centroids)

    norm = plt.Normalize(min(rep_key), max(rep_key))
    cmap = cm.get_cmap('plasma')
    point_colors = cmap(norm(rep_key))
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    for cluster_idx in range(num_clusters):
        cluster_points = reduced_vectors[clusters == cluster_idx]
        axs[0].scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=cmap(cluster_idx / num_clusters), s=50, label=f"Cluster {cluster_idx}")

    axs[0].scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], 
                c='red', marker='X', s=200, label="Centroids")
    axs[0].set_title("Clusters Based on KMeans")
    axs[0].set_xlabel("Principal Component 1")
    axs[0].set_ylabel("Principal Component 2")
    axs[0].legend()
    scatter = axs[1].scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], 
                            c=point_colors, s=50)

    axs[1].scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], 
                c='red', marker='X', s=200, label="Centroids")

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[1], label='Sequence Length')
    axs[1].set_title("Color Based on Sequence Length")
    axs[1].set_xlabel("Principal Component 1")
    axs[1].set_ylabel("Principal Component 2")

    plt.tight_layout()
    plt.show()
    plt.savefig('cluster_check.png')