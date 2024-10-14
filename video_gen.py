import torch
import numpy as np
import gym
import d4rl
import cv2  # For saving videos
import matplotlib.pyplot as plt  # For plotting
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from models import EncoderLSTM, DecoderLSTM, Seq2Seq
from utils import sample_trajectories, get_representation
env = gym.make('halfcheetah-medium-replay-v2')
dataset = env.get_dataset()
qpos = dataset['infos/qpos']
qvel = dataset['infos/qvel']
actions = dataset['actions']
states = dataset['observations']
next_states = dataset['next_observations']
rewards = dataset['rewards']
dones = dataset['terminals']

input_dim = env.observation_space.shape[0] + env.action_space.shape[0]
hidden_dim = 128
output_dim = env.observation_space.shape[0]
num_layers = 4
encoder = EncoderLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
decoder = DecoderLSTM(hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Seq2Seq(encoder, decoder).to(device)
model.load_state_dict(torch.load('/home/rishav/scratch/xrl/seq2seq_32.pth'))


def set_mujoco_state(env, qpos, qvel):
    assert qpos.shape == (env.model.nq,), f"qpos shape mismatch: {qpos.shape} vs {env.model.nq}"
    assert qvel.shape == (env.model.nv,), f"qvel shape mismatch: {qvel.shape} vs {env.model.nv}"
    env.set_state(qpos, qvel)

def save_video(frames, filename, fps=30):
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        video.write(frame)
    video.release()

def plot_clusters(reps, clusters, rep_key, num_clusters, filename):
    # Check if reps contain any NaN values
    if np.isnan(reps).any():
        print("Warning: NaN values found in reps. Removing NaN values before PCA.")
        reps = reps[~np.isnan(reps).any(axis=1)]  
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(reps)
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    cmap = plt.get_cmap('viridis', num_clusters)
    for cluster_idx in range(num_clusters):
        cluster_points = reduced_vectors[clusters == cluster_idx]
        axs[0].scatter(cluster_points[:, 0], cluster_points[:, 1],
                       color=cmap(cluster_idx / num_clusters), s=50, label=f"Cluster {cluster_idx}")
    axs[0].set_title("Clusters Based on KMeans")
    axs[0].set_xlabel("Principal Component 1")
    axs[0].set_ylabel("Principal Component 2")
    axs[0].legend()
    norm = plt.Normalize(min(rep_key), max(rep_key))
    scatter = axs[1].scatter(reduced_vectors[:, 0], reduced_vectors[:, 1],
                             c=rep_key, cmap='plasma', s=50)
    fig.colorbar(scatter, ax=axs[1], label='Sequence Length')
    axs[1].set_title("Color Based on Sequence Length")
    axs[1].set_xlabel("Principal Component 1")
    axs[1].set_ylabel("Principal Component 2")

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

stored_sequences = {i: [] for i in range(4)}

with torch.no_grad():
    trajs = sample_trajectories(states, actions, rewards, next_states, dones)
    
    st_idx = 0
    reps = []
    rep_key = []
    while st_idx < 10000:
        for s_len in range(4, 32, 4): 
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
    sequence_count_per_cluster = {i: 0 for i in range(num_clusters)}
    for idx, cluster_idx in enumerate(clusters):
        s_len = rep_key[idx]
        if sequence_count_per_cluster[cluster_idx] < 5:
            stat = torch.tensor(trajs[0]['states'][st_idx:st_idx+s_len], dtype=torch.float32)
            act = torch.tensor(trajs[0]['actions'][st_idx:st_idx+s_len], dtype=torch.float32)
            sequence = {"states": stat, "actions": act, "length": s_len}
            stored_sequences[cluster_idx].append(sequence)
            sequence_count_per_cluster[cluster_idx] += 1
    for cluster_idx, sequences in stored_sequences.items():
        for i, seq in enumerate(sequences):
            print(f"Rendering Cluster {cluster_idx}, Sequence {i+1}, Length {seq['length']}")
            frames = []
            env.reset()

            for t in range(seq['length']):
                qpos_t = qpos[t]
                qvel_t = qvel[t]
                action = seq['actions'][t].cpu().numpy()

                set_mujoco_state(env, qpos_t, qvel_t)
                frame = env.render(mode='rgb_array')
                frames.append(frame)
                env.step(action)
            video_filename = f"cluster_{cluster_idx}_sequence_{i+1}.mp4"
            save_video(frames, video_filename)
            print(f"Video saved as {video_filename}")
    plot_filename = "cluster_plot.png"
    plot_clusters(reps, clusters, rep_key, num_clusters, plot_filename)
    print(f"Cluster plot saved as {plot_filename}")

env.close()