from models import Seq2Seq, EncoderLSTM, DecoderLSTM
import gym
import d4rl 
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make('halfcheetah-medium-v2')
dataset = d4rl.qlearning_dataset(env)

states = dataset['observations']
actions = dataset['actions']
next_states = dataset['next_observations']
rewards = dataset['rewards']


mean_state = states.mean(axis=0)
std_state = states.std(axis=0)
states = (states - mean_state) / std_state
next_states = (next_states - mean_state) / std_state


input_dim = states.shape[1] + actions.shape[1]  
hidden_dim = 128 
output_dim = states.shape[1]
num_layers = 2
seq_len = 32 


encoder = EncoderLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
decoder = DecoderLSTM(hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
model = Seq2Seq(encoder, decoder)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def create_sequences(states, actions, seq_len):
    seq_states = []
    seq_actions = []
    seq_next_states = []
    
    for i in range(len(states) - seq_len):
        seq_states.append(states[i:i+seq_len])
        seq_actions.append(actions[i:i+seq_len])
        seq_next_states.append(states[i+1:i+seq_len+1])
    
    return torch.Tensor(seq_states), torch.Tensor(seq_actions), torch.Tensor(seq_next_states)




def train(epochs, model, save_interval, seq_input, seq_next_states):
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(seq_input, seq_next_states)
        loss = criterion(predictions, seq_next_states)
        loss.backward() 
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            model_filename = f"seq2seq_model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"Model weights saved to {model_filename}")

def get_representations(model, sequence, start, end):
    outs = []
    for i in range(start, end):
        hidden, cell = model.encoder(sequence[0:i+1])
        outs.append(hidden)
    return outs    

seq_states, seq_actions, seq_next_states = create_sequences(states, actions, seq_len)
seq_input = torch.cat((seq_states, seq_actions), dim=2)
epochs = 100
save_interval = 10
train(epochs, model, save_interval, seq_input, seq_next_states)
