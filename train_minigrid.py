from utils import get_args_minigrid
from models.behavior_models import BeXRLSequence, BeXRLState
import d4rl
import gym
import torch.nn as nn
import torch
from utils import D4RLMinigrid
import wandb
import numpy as np
from utils import visualize_codebook_clusters, cluster_codebook_vectors, extract_overlapping_segments_minigrid, plot_distances_minigrid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = get_args_minigrid()
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
model_type = args.model_type


if model_type == 'sequence':
    instance = BeXRLSequence
else:
    instance = BeXRLState


if args.log:
    wandb_run_name = env_name + "_" + model_type + "_" + wandb.util.generate_id()
    wandb.init(project=wandb_project, name=wandb_run_name, entity=wandb_entity, config=args)

env = gym.make(env_name)
dataset_org = env.get_dataset()
states = dataset_org['observations']
actions = dataset_org['actions']
dones = dataset_org['terminals']

dataset = D4RLMinigrid(states, actions, dones, seq_len=5, num_actions=len(np.unique(actions)))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)



input_dim = states[0].flatten().shape[0] + len(np.unique(actions))
output_dim = states[0].flatten().shape[0]    

model = instance(input_dim, model_dim, output_dim, num_heads, num_encoder_layers, num_decoder_layers, num_embeddings).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if model_load_path!='':
    model.load_state_dict(torch.load(model_load_path))

model.eval()
segments = extract_overlapping_segments_minigrid(dataset_org, 5, 100)

for tseg in segments:
    if len(tseg) > 30:
        print("Plotting distances")
        plot_distances_minigrid(tseg, model.encoder)
        break

if args.train:
    model.train()
    

    for epoch in range(num_epochs):
        epoch_loss = 0
        for src, tgt in dataloader:
            optimizer.zero_grad()
            src = src.to(device)

            tgt_input = tgt[:, :-1, :].to(device)  # Use all but last next state as target input for the decoder
            tgt_output = tgt[:, 1:, :].to(device)  # Shifted target sequence to predict next state
            output, vq_loss, encidx = model(src, tgt_input)
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

        

        clusters, kmeans, centers = cluster_codebook_vectors(model.encoder.vq_layer.embedding, 10)
        visualize_codebook_clusters(model.encoder.vq_layer.embedding, clusters, centers, env_name)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader)} num unique IDs {len (torch.unique(encidx))}")



    torch.save(model.state_dict(), model_save_path)
    