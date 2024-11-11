from utils import get_args_mujoco
from models.behavior_models import BeXRLSequence
import d4rl
import gym
import torch.nn as nn
import torch
from utils import D4RLDatasetMujoco
import wandb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = get_args_mujoco()
env_name = args.env_name
model_dim = args.model_dim
num_heads = args.num_heads 
num_encoder_layers = args.num_encoder_layers
num_decoder_layers = args.num_decoder_layers
num_embeddings = args.num_embeddings
batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = args.num_epochs
model_save_path = args.model_save_path
model_load_path = args.model_load_path
train = args.train
log = args.log
wandb_project = args.wandb_project
wandb_entity = args.wandb_entity


if args.log:
    #run name as-> env_name + "_" + DD-HH-MM
    wandb_run_name = env_name + "_" + wandb.util.generate_id()

    #add config to wandb
    wandb.init(project=wandb_project, name=wandb_run_name, entity=wandb_entity, config=args)


env = gym.make(env_name)
dataset = env.get_dataset()
num_samples = 100000
states = dataset['observations'][:num_samples]
actions = dataset['actions'][:num_samples]
next_states = dataset['next_observations'][:num_samples]

input_dim = states.shape[1] + actions.shape[1] 
output_dim = states.shape[1]      


model = BeXRLSequence(input_dim, model_dim, output_dim, num_heads, num_encoder_layers, num_decoder_layers, num_embeddings).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataset = D4RLDatasetMujoco(states, actions, next_states, sequence_length=25)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

if args.train:
    model.train()
    if model_load_path!='':
        model.load_state_dict(torch.load(model_load_path))

    for epoch in range(num_epochs):
        epoch_loss = 0
        for src, tgt in dataloader:
            optimizer.zero_grad()
            
            # Prepare inputs and targets
            src = src.to(device)

            tgt_input = tgt[:, :-1, :].to(device)  # Use all but last next state as target input for the decoder
            tgt_output = tgt[:, 1:, :].to(device)  # Shifted target sequence to predict next state
            
            # Forward pass
            output, vq_loss, encidx = model(src, tgt_input)
            
            # Compute loss
            loss = criterion(output, tgt_output) + vq_loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if args.log:
                wandb.log({"loss": loss.item(), "vq_loss": vq_loss.item(), "num_unique_ids": len(torch.unique(encidx))})
                
                #plot distances between embeddings histogram to see the evolution of embeddings with time.
                with torch.no_grad():
                    embeddings = model.encoder.vq_layer.embedding
                    distances = torch.cdist(embeddings, embeddings, p=2).cpu().numpy()
                    wandb.log({"embedding_distances_histogram": wandb.Histogram(distances)})

        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader)} num unique IDs {len (torch.unique(encidx))}")

    torch.save(model.state_dict(), model_save_path)