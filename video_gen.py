import torch
import numpy as np
import gym
import d4rl
import cv2  # For saving videos
from sklearn.cluster import KMeans
from models import EncoderLSTM, DecoderLSTM, Seq2Seq
from utils import get_representation
import os

env = gym.make('halfcheetah-medium-v2')
dataset = env.get_dataset()
num_items = 10000
qpos = dataset['infos/qpos'][:num_items]
qvel = dataset['infos/qvel'][:num_items]
actions = dataset['actions'][:num_items]
states = dataset['observations'][:num_items]
next_states = dataset['next_observations'][:num_items]
rewards = dataset['rewards'][:num_items]
dones = dataset['terminals'][:num_items]

input_dim = states.shape[1] + actions.shape[1]
hidden_dim = 128
output_dim = states.shape[1]
num_layers = 4
seq_len = 32
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

def create_sequences(states, actions, seq_len):
    seq_states = []
    seq_actions = []
    seq_next_states = []
    render_data = []
    for i in range(0, len(states) - seq_len, seq_len):
        seq_states.append(states[i:i+seq_len])
        seq_actions.append(actions[i:i+seq_len])
        seq_next_states.append(next_states[i+1:i+seq_len+1])
        render_data.append([qpos[i:i+seq_len], qvel[i:i+seq_len]])

    seq_states = torch.Tensor(seq_states)
    seq_actions = torch.Tensor(seq_actions)
    seq_next_states = torch.Tensor(seq_next_states)
    return seq_states, seq_actions, seq_next_states, render_data

with torch.no_grad():
    trajs = create_sequences(states, actions, seq_len)
    reps = []
    trajectories = []
    render_info = [] 

    for stat, act, _, render_d in zip(trajs[0], trajs[1], trajs[2], trajs[-1]): 
        stat = torch.tensor(stat, dtype=torch.float32)
        act = torch.tensor(act, dtype=torch.float32)
        seq = torch.cat([stat, act], dim=-1).unsqueeze(0).to(device)
        rep = get_representation(model, seq).reshape(-1) 
        reps.append(rep)
        
        trajectories.append((stat, act)) 
        render_info.append(render_d)
    
    reps = torch.stack(reps).cpu().numpy()

    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(reps)
    
    render_dict = {i: [] for i in range(num_clusters)}  
    cluster_reps = {i: [] for i in range(num_clusters)} 
    for idx, cluster_label in enumerate(clusters):
        render_dict[cluster_label].append(render_info[idx])
        cluster_reps[cluster_label].append(reps[idx])
    
    median_trajectories = {}

    for cluster_num in range(num_clusters):
        cluster_rep_array = np.vstack(cluster_reps[cluster_num])
        median_rep = np.median(cluster_rep_array, axis=0)
        distances = np.linalg.norm(cluster_rep_array - median_rep, axis=1)
        closest_indices = np.argsort(distances)[:3]
        median_trajectories[cluster_num] = [render_dict[cluster_num][i] for i in closest_indices]
    
    
    # for cluster_num, trajs in median_trajectories.items():
    #     print(f"Cluster {cluster_num} has {len(trajs)} median trajectories.")

    for cluster_num, trajectories in median_trajectories.items():
        for i, traj in enumerate(trajectories):
            frames = []
            for t in range(seq_len):
                qpos_t = traj[0][t]
                qvel_t = traj[1][t]
                set_mujoco_state(env, qpos_t, qvel_t)
                frame = env.render(mode='rgb_array')
                frames.append(frame)

            os.makedirs(f"video_lstm_{seq_len}", exist_ok=True)
            video_filename = f"video_lstm_{seq_len}/cluster_{cluster_num}_sequence_{i+1}.mp4"
            save_video(frames, video_filename)
            store = f"images/{cluster_num}/"
            
            os.makedirs(store, exist_ok=True)
            for j, frame in enumerate(frames):
                cv2.imwrite(f"images_lstm_{seq_len}/{cluster_num}/{j}.png", frame)