from models import Seq2Seq, EncoderLSTM, DecoderLSTM
import gym
import d4rl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import wandb
from datetime import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--track', action='store_true')
parser.add_argument('--seq_len', default=16, type=int)
args = parser.parse_args()



env = gym.make('halfcheetah-medium-v2')
dataset = env.get_dataset()
num_items = len( dataset['observations'])
states = dataset['observations'][:num_items]
actions = dataset['actions'][:num_items]
next_states = dataset['next_observations'][:num_items]
rewards = dataset['rewards'][:num_items]
dones = dataset['terminals'][:num_items]

input_dim = states.shape[1] + actions.shape[1]
hidden_dim = 128
output_dim = states.shape[1]
num_layers = 4
seq_len = args.seq_len
batch_size = 256 



if args.track:
    run_name = f'LSTM_{seq_len}_to_{seq_len}_' + datetime.now().strftime("%d-%m-%y-%H-%M-%S")
    wandb.init(project='xrl_lstm_seqs', entity='mail-rishav9', name=run_name)
    wandb.config.update({
        'seq_len':args.seq_len,
        'batch_size':batch_size
    })


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
encoder = EncoderLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
decoder = DecoderLSTM(hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
model = Seq2Seq(encoder, decoder).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)


def create_sequences(states, actions, seq_len, batch_size):
    seq_states = []
    seq_actions = []
    seq_next_states = []
    for i in range(0, len(states) - seq_len, seq_len):
        seq_states.append(states[i:i+seq_len])
        seq_actions.append(actions[i:i+seq_len])
        seq_next_states.append(next_states[i+1:i+seq_len+1])
    seq_states = torch.Tensor(seq_states)
    seq_actions = torch.Tensor(seq_actions)
    seq_next_states = torch.Tensor(seq_next_states)
    num_batches = len(seq_states) // batch_size
    batched_input = []
    batched_output = []

    for i in range(0, num_batches * batch_size, batch_size):
        input_seq = torch.cat((seq_states[i:i+batch_size], seq_actions[i:i+batch_size]), dim=2)
        output_seq = seq_next_states[i:i+batch_size]
        batched_input.append(input_seq.to(device))
        batched_output.append(output_seq.to(device))
    
    return batched_input, batched_output


def train(epochs, model, save_interval, batched_input, batched_output):
    for epoch in range(epochs):
        epoch_loss = 0.0
        nan = False

        for seq_input, seq_output in zip(batched_input, batched_output):
            optimizer.zero_grad()
            predictions = model(seq_input, seq_output)
            loss = criterion(predictions, seq_output)
            if not torch.isnan(loss):
                if args.track:
                    wandb.log({'seq2seq_loss':loss})
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            else:
                nan = True
                print('NaN encountered!')
                break

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss/len(batched_input)}')
        if nan:
            print('NaN encountered!')
            continue
        else:
            print('saving model weights!')
            model_filename = f"seq2seq_{seq_len}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"Model weights saved to {model_filename}")

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)
            if torch.isnan(grad_norm) or grad_norm > 1000:
                print(f"Warning: High gradient value detected in {name}: {grad_norm}")

def init_weights(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

def get_representation(model, seq):
    rep, _ = model.encoder(seq)
    return rep


def sample_trajectories(states, actions, rewards, next_states, dones):
    trajectories = []
    current_trajectory = {'states': [], 'actions': [], 'rewards': [], 'next_states': []}

    for i in range(len(states)):
        current_trajectory['states'].append(states[i])
        current_trajectory['actions'].append(actions[i])
        current_trajectory['rewards'].append(rewards[i])
        current_trajectory['next_states'].append(next_states[i])
        if dones[i]:
            trajectories.append(current_trajectory)
            if(len(trajectories) == 20):
                break
            current_trajectory = {'states': [], 'actions': [], 'rewards': [], 'next_states': []}
        
    if len(current_trajectory['states']) > 0:
        trajectories.append(current_trajectory)

    return trajectories

model.apply(init_weights)
batched_input, batched_output = create_sequences(states, actions, seq_len, batch_size)
epochs = 50
save_interval = 100
model_path = ''
if os.path.exists(model_path):
    print(f"Model loaded from path {model_path}.")
    model.load_state_dict(torch.load(model_path))


train(epochs, model, save_interval, batched_input, batched_output)



