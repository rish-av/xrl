import gymnasium as gym
import torch
import numpy as np
from sklearn.cluster import KMeans
from stable_baselines3 import PPO
from minigrid.wrappers import FullyObsWrapper
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA


env = FullyObsWrapper(gym.make('MiniGrid-Empty-16x16-v0'))
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=5000)  # Train the agent

def collect_trajectories(model, env, num_episodes=10):
    trajectories = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        trajectory = []
        while not done:
            action, _ = model.predict(obs)
            next_obs, reward, done, info = env.step(action)
            trajectory.append((obs, action))
            obs = next_obs
        trajectories.append(trajectory)
    return trajectories

trajectories = collect_trajectories(model, env)

# Encode trajectories using the agent's internal policy network
def encode_trajectory(trajectory, model):
    encoded_states = []
    for obs, _ in trajectory:
        obs_tensor = torch.tensor(obs['image']).float().unsqueeze(0)
        with torch.no_grad():
            latent_rep = model.policy.features_extractor(obs_tensor)
        encoded_states.append(latent_rep.numpy().flatten())
    return encoded_states

#Cluster the latent states into behavioral concepts
def discover_and_visualize_latent_concepts(trajectories, model, n_clusters=4, save_path='cluster_data/'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_latent_reps = []
    for traj in trajectories:
        encoded_traj = encode_trajectory(traj, model)
        all_latent_reps.extend(encoded_traj)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(all_latent_reps)
    concept_labels = kmeans.labels_
    pca = PCA(n_components=2)  
    latent_2d = pca.fit_transform(all_latent_reps)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=concept_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'Latent Concept Clusters (n_clusters={n_clusters})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    cluster_fig_path = os.path.join(save_path, 'cluster_visualization.png')
    plt.savefig(cluster_fig_path)
    plt.close()

    print(f"Cluster visualization saved as {cluster_fig_path}")

# Map each segment to a concept label
kmeans, concept_labels = discover_and_visualize_latent_concepts(trajectories)

#Saliency Analysis
def compute_saliency_score(segment, model):
    """
    Simple saliency analysis: sum the policy log-probabilities for the segment actions.
    """
    influence_score = 0
    for obs, action in segment:
        obs_tensor = torch.tensor(obs['image']).float().unsqueeze(0)
        with torch.no_grad():
            action_probs = model.policy(obs_tensor).log_prob(torch.tensor([action]))
        influence_score += action_probs.item()
    return influence_score

# Explanations for Each Trajectory Segment
def generate_explanation(segment, mapped_concept, influence_score):
    """
    Generate a human-readable explanation based on trajectory segments.
    """
    start_state = segment[0][0]
    end_state = segment[-1][0]
    actions = [a for _, a in segment]
    explanation = (
        f"Segment from ({start_state}) to ({end_state}) with actions {actions} is "
        f"mapped to '{mapped_concept}' concept. Influence score: {influence_score:.2f}."
    )
    return explanation

# visual
sample_trajectory = trajectories[0]
encoded_trajectory = encode_trajectory(sample_trajectory, model)

# assign concepts
explanations = []
for i in range(0, len(sample_trajectory), 2):  # Take segments of length 2
    segment = sample_trajectory[i:i+2]
    if len(segment) < 2:
        continue
    segment_encoding = encoded_trajectory[i:i+2]
    concept_label = kmeans.predict(np.array(segment_encoding).reshape(1, -1))[0]
    influence_score = compute_saliency_score(segment, model)
    explanation = generate_explanation(segment, f"Concept {concept_label}", influence_score)
    explanations.append(explanation)

for idx, exp in enumerate(explanations):
    print(f"Explanation {idx + 1}: {exp}")