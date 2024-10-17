import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
import d4rl
from models import VQ_VAE_Segment
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQ_VAE_Segment(state_dim=17, action_dim=6, latent_dim=32, num_embeddings=512).to(device)
model.load_state_dict(torch.load('vq_vae.pth')) 
model.eval()

env = gym.make('halfcheetah-medium-v2')
dataset = env.get_dataset()

def create_trajectories(dataset, seq_len):
    states = dataset['observations']
    actions = dataset['actions']
    qpos = dataset['infos/qpos']
    qvel = dataset['infos/qvel']
    num_items = len(states)
    
    traj_data = []
    render_data = []
    for i in range(0, num_items - seq_len, seq_len):
        state_seq = states[i:i+seq_len]
        action_seq = actions[i:i+seq_len]

        qpos_seq = qpos[i:i+seq_len]
        qvel_seq = qvel[i:i+seq_len]

        traj = torch.tensor(np.concatenate([state_seq, action_seq], axis=-1), dtype=torch.float32)
        traj_data.append(traj)
        render_data.append([qpos_seq, qvel_seq])
    return traj_data, render_data

seq_len = 25
traj_data, render_data = create_trajectories(dataset, seq_len)

def segment_trajectory(traj, model):
    with torch.no_grad():
        traj = traj.unsqueeze(0).to(device)  
        _, _, encoding_indices = model(traj)

        encoding_indices = encoding_indices.cpu().numpy()
        codebook_vectors = model.vq_layer.embeddings.weight[encoding_indices]

    return encoding_indices, codebook_vectors.cpu().numpy()

codebook_means = []
for traj_num, traj in enumerate(traj_data[:5000]):  
    _, codebook_vecs = segment_trajectory(traj, model)
    codebook_means.append(np.mean(codebook_vecs, axis=0))

codebook_means_array = np.vstack(codebook_means)
pca = PCA(n_components=2)
codebook_means_reduced = pca.fit_transform(codebook_means_array)
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(codebook_means_array)
centroids = kmeans.cluster_centers_
distances_from_centroid = np.linalg.norm(codebook_means_array - centroids[cluster_labels], axis=1)
median_trajectories = {cluster_num: [] for cluster_num in range(num_clusters)}

for cluster_num in range(num_clusters):
    cluster_indices = np.where(cluster_labels == cluster_num)[0]
    cluster_distances = distances_from_centroid[cluster_indices]
    sorted_indices = cluster_indices[np.argsort(cluster_distances)]
    num_median = len(sorted_indices)//2
    ##this is for rendering
    median_trajectories[cluster_num] = [render_data[i] for i in sorted_indices[num_median:num_median+3]]

pca = PCA(n_components=2)
codebook_means_reduced = pca.fit_transform(codebook_means_array)
centroids_reduced = pca.transform(centroids)
plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'orange', 'yellow', 'pink'] 
for cluster_num in range(num_clusters):
    cluster_points = codebook_means_reduced[cluster_labels == cluster_num]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                color=colors[cluster_num], s=100, label=f'Cluster {cluster_num}')
plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], 
            c='red', marker='X', s=200, label='Centroids')

plt.title("PCA on mean of codebook vectors")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()
plt.savefig('check_codebook.png')


def save_video(frames, filename, fps=30):
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        video.write(frame)
    video.release()

def set_mujoco_state(env, qpos, qvel):
    assert qpos.shape == (env.model.nq,), f"qpos shape mismatch: {qpos.shape} vs {env.model.nq}"
    assert qvel.shape == (env.model.nv,), f"qvel shape mismatch: {qvel.shape} vs {env.model.nv}"
    env.set_state(qpos, qvel)


for cluster_num, trajectories in median_trajectories.items():
    for i, traj in enumerate(trajectories):
        frames = []
        for t in range(seq_len):
            qpos_t = traj[0][t]
            qvel_t = traj[1][t]
            set_mujoco_state(env, qpos_t, qvel_t)
            frame = env.render(mode='rgb_array')
            frames.append(frame)
        video_filename = f"video_vq/cluster_{cluster_num}_sequence_{i+1}.mp4"
        save_video(frames, video_filename)
        store = f"images/{cluster_num}/"
        
        os.makedirs(store, exist_ok=True)
        for j, frame in enumerate(frames):
            cv2.imwrite(f"images/{cluster_num}/{j}.png", frame)