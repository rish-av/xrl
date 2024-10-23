import gym
import d4rl
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
from models_minigrid import VQ_TrajectoryVAE
from torch.nn.utils.rnn import pad_sequence
import os 
import pickle


input_channels = 3
output_channels = 3
latent_dim = 64
num_embeddings = 512
hidden_dim = 128
state_dim = 10
action_dim = 1
seq_len = 10
batch_size = 1
learning_rate = 1e-3
epochs = 100
num_trajs = 500

env = gym.make("minigrid-fourrooms-v0")
dataset = env.get_dataset()


def create_trajectories(dataset, pickle_file='trajectories_minigrid4rooms.pkl'):
    if os.path.exists(pickle_file):
        print("Loading trajectories from pickle file...")
        with open(pickle_file, 'rb') as f:
            traj_data_imgs, traj_data_acts = pickle.load(f)
        return traj_data_imgs[:num_trajs], traj_data_acts[:num_trajs]

    states = dataset['observations']
    actions = dataset['actions']
    dones = dataset['terminals']

    agent_pos = dataset['infos/pos']
    agent_dir = dataset['infos/orientation']

    num_items = len(states)

    traj_data_imgs = []
    traj_data_acts = []

    curr_traj_imgs = []
    curr_traj_acts = []
    count = 0
    
    for i in range(0, num_items): 
        env.agent_pos = agent_pos[i]
        env.agent_dir = agent_dir[i]
        state_img = env.render(mode='rgb_array')
        new_size = (96, 96)
        state_img = cv2.resize(state_img, new_size, interpolation=cv2.INTER_LINEAR)
        state_img = np.float32(state_img)
        curr_traj_imgs.append(np.transpose(state_img, (-1, 0, 1)))  
        curr_traj_acts.append(actions[i])

        if dones[i]:
            print(f"Trajectory collected {i+1}/{num_items}")
            count += 1

            traj_data_imgs.append(curr_traj_imgs)
            traj_data_acts.append(curr_traj_acts)
            
            curr_traj_imgs = []
            curr_traj_acts = []
    
    print(f"Saving trajectories to {pickle_file}...")
    with open(pickle_file, 'wb') as f:
        pickle.dump((traj_data_imgs, traj_data_acts), f)

    return traj_data_imgs, traj_data_acts

traj_data_imgs, traj_data_acts = create_trajectories(dataset)

model = VQ_TrajectoryVAE(input_channels, action_dim=action_dim, latent_dim=latent_dim, num_embeddings=num_embeddings, output_channels=output_channels)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()
cross_entropy_loss = nn.CrossEntropyLoss()


def data_loader(imgs,acts, batch_size):
    for i in range(0, len(imgs), batch_size):

        img_batch = imgs[i:i+batch_size]
        act_batch = acts[i:i+batch_size]
        # img_batch = [torch.tensor(seq) for seq in img_batch]
        # act_batch = [torch.tensor(seq) for seq in act_batch]
        # img_batch_padded = pad_sequence(img_batch, batch_first=True, padding_value=0)
        # act_batch_padded = pad_sequence(act_batch, batch_first=True, padding_value=0)

        # print(img_batch.shape, img_batch_padded.shape)

        yield torch.tensor(img_batch[0]), torch.tensor(act_batch[0])

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for data in data_loader(traj_data_imgs, traj_data_acts, batch_size):
        img_batch = data[0]
        act_batch = data[-1]
        img_batch = img_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
        act_batch = act_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
        img_batch= (img_batch - torch.mean(img_batch))/torch.std(img_batch)
        recon_images, action_preds, vq_loss, _ = model(img_batch.unsqueeze(0), act_batch)
        image_loss = mse_loss(recon_images, img_batch.unsqueeze(0))
        total_loss = image_loss  + vq_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss/num_trajs:.4f}")
print("Training complete.")