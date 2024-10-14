import gym
import d4rl  # Import D4RL for the dataset
import torch
import cv2  # OpenCV for saving videos

# Initialize the environment
env = gym.make('halfcheetah-medium-replay-v2')

# Load the dataset
dataset = env.get_dataset()
states = dataset['observations']
actions = dataset['actions']
next_states = dataset['next_observations']
rewards = dataset['rewards']
dones = dataset['terminals']

# Function to set environment's state manually from D4RL dataset
def set_mujoco_state(env, state):
    qpos = state[:env.model.nq]  # First nq elements are positions (qpos)
    qvel = state[env.model.nq:env.model.nq + env.model.nv]  # Next nv elements are velocities (qvel)
    env.set_state(qpos, qvel)

# Function to save video from frames using OpenCV
def save_video(frames, filename, fps=30):
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for frame in frames:
        video.write(frame)
    
    video.release()

# Render and save a trajectory as video
def render_trajectory(env, states, actions, sequence_length=1000, video_filename="trajectory.mp4"):
    frames = []

    # Reset the environment to the initial state
    env.reset()

    for t in range(sequence_length):
        # Set the environment state using the D4RL state
        state = states[t]
        action = actions[t]

        set_mujoco_state(env, state)  # Set MuJoCo state to match the dataset

        # Render the environment and capture the frame
        frame = env.render(mode='rgb_array')
        frames.append(frame)

        # Step the environment using the action
        env.step(action)

    # Save the frames as a video
    save_video(frames, video_filename)
    print(f"Video saved as {video_filename}")

# Render the first 1000 steps of a trajectory and save as a video
render_trajectory(env, states, actions, sequence_length=1000, video_filename="halfcheetah_trajectory.mp4")

# Close the environment
env.close()