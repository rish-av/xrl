from models import Seq2Seq, EncoderLSTM, DecoderLSTM
import gym
import d4rl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
env = gym.make('halfcheetah-medium-v2')
dataset = d4rl.qlearning_dataset(env)
num_items = 2000
states = dataset['observations'][:num_items]
actions = dataset['actions'][:num_items]
next_states = dataset['next_observations'][:num_items]
rewards = dataset['rewards'][:num_items]
dones = dataset['terminals'][:num_items]



input_dim = states.shape[1] + actions.shape[1]
hidden_dim = 128
output_dim = states.shape[1]
num_layers = 2
seq_len = 16
batch_size = 2 
encoder = EncoderLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
decoder = DecoderLSTM(hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
model = Seq2Seq(encoder, decoder)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
mean_state = np.mean(np.array(states))
std_state = np.std(np.array(states)) + 1e-8
states = (states - mean_state) / std_state
next_states = (next_states - mean_state) / std_state




def create_sequences(states, actions, seq_len, batch_size):
    seq_states = []
    seq_actions = []
    seq_next_states = []
    for i in range(len(states) - seq_len):
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
        batched_input.append(input_seq)
        batched_output.append(output_seq)
    
    return batched_input, batched_output


def train(epochs, model, save_interval, batched_input, batched_output):
    for epoch in range(epochs):
        epoch_loss = 0.0

        for seq_input, seq_output in zip(batched_input, batched_output):
            optimizer.zero_grad()

            predictions = model(seq_input, seq_output)
            loss = criterion(predictions, seq_output)
            loss.backward()
            # check_gradients(model)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss/len(batched_input)}')
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            model_filename = f"seq2seq_model_epoch_{epoch + 1}.pth"
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

# Prepare sequences and batch them
batched_input, batched_output = create_sequences(states, actions, seq_len, batch_size)

# Apply weight initialization
model.apply(init_weights)

# Train the model
epochs = 1000
save_interval = 100
train(epochs, model, save_interval, batched_input, batched_output)