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

from utils import get_representation



env = gym.make('halfcheetah-medium-v2')
dataset = env.get_dataset()
num_items = 10000*2
states = dataset['observations'][:num_items]
actions = dataset['actions'][:num_items]
next_states = dataset['next_observations'][:num_items]
rewards = dataset['rewards'][:num_items]
dones = dataset['terminals'][:num_items]



input_dim = states.shape[1] + actions.shape[1]
hidden_dim = 128
output_dim = states.shape[1]
num_layers = 4
seq_len = 64
batch_size = 2 
encoder = EncoderLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
decoder = DecoderLSTM(hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Seq2Seq(encoder, decoder).to(device)
model.load_state_dict(torch.load('/home/rishav/scratch/xrl/seq2seq_32.pth'))


def create_sequences(states, actions, seq_len):
    seq_states = []
    seq_actions = []
    seq_next_states = []
    for i in range(0,len(states) - seq_len, seq_len):
        seq_states.append(states[i:i+seq_len])
        seq_actions.append(actions[i:i+seq_len])
        seq_next_states.append(next_states[i+1:i+seq_len+1])

    total_seqs = len(seq_states)
    seq_states = torch.Tensor(seq_states)
    seq_actions = torch.Tensor(seq_actions)
    seq_next_states = torch.Tensor(seq_next_states)
    return seq_states, seq_actions, seq_next_states

with torch.no_grad():
    trajs = create_sequences(states, actions, seq_len)
    reps = []
    rep_key = []
    for stat, act, _  in zip(trajs[0], trajs[1], trajs[2]): 
        stat = torch.tensor(stat, dtype=torch.float32)
        act = torch.tensor(act, dtype=torch.float32)
        seq = torch.cat([stat, act], dim=-1).unsqueeze(0).to(device)
        # print(seq.shape)
        rep = get_representation(model, seq).reshape(-1) 
        rep_key.append(seq_len) 
        reps.append(rep)
    reps = torch.stack(reps).cpu().numpy()
    rep_key = np.array(rep_key)

    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(reps)
    centroids = kmeans.cluster_centers_

    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(reps)
    reduced_centroids = pca.transform(centroids)

    norm = plt.Normalize(min(rep_key), max(rep_key))
    cmap = cm.get_cmap('plasma')
    point_colors = cmap(norm(rep_key))
    # fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    for cluster_idx in range(num_clusters):
        cluster_points = reduced_vectors[clusters == cluster_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=cmap(cluster_idx / num_clusters), s=50, label=f"Cluster {cluster_idx}")

    plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], 
                c='red', marker='X', s=200, label="Centroids")
    plt.title(f"Clusters Sequence Length = {seq_len}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f'cluster_seq_len_{seq_len}.png')