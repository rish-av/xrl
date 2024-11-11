
import torch
import gym 
import numpy as np
from tnew2 import TrajectoryTransformer
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


env = gym.make('halfcheetah-medium-v2')
dataset = env.get_dataset()
observations = dataset['observations'][:100000]
actions = dataset['actions'][:100000]
embedding_dim = 128
seq_len = 25
num_heads = 8
num_layers = 4
num_embeddings = 64
input_dim = 23

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TrajectoryTransformer(input_dim, embedding_dim, num_heads, num_layers, num_embeddings).to(device)
model.load_state_dict(torch.load('/home/rishav/scratch/xrl/best_model_embedding_128_codebook_size_64.pth'))



sequences = np.hstack((observations, actions))
sequences = torch.tensor(sequences, dtype=torch.float32)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def render_halfcheetah(env, state):
    env.reset()
    env.env.sim.set_state_from_flattened(np.insert(state, 0, [0,0]))
    env.env.sim.forward()
    img = env.render(mode="rgb_array")
    return img



def plot_codebook_distances(quantizer):
    codebook_vectors = quantizer.embeddings.weight.detach().cpu()
    distance_matrix = torch.cdist(codebook_vectors, codebook_vectors, p=2)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(distance_matrix.numpy(), annot=False, cmap='viridis', cbar=True, square=True)
    plt.title("Pairwise Distance Between Codebook Vectors")
    plt.xlabel("Codebook Vector Index")
    plt.ylabel("Codebook Vector Index")
    plt.show()
    plt.savefig('codebook_distance.png')
plot_codebook_distances(model.quantizer)

def save_sequences_as_videos(behaviors, savepath='visuals_attvq', fps=10):
    for idx, sequences in behaviors.items():
        behavior_path = os.path.join(savepath, f'behavior_{idx}')
        os.makedirs(behavior_path, exist_ok=True)
        
        for seq_num, states in enumerate(sequences):
            height, width = states[0].shape[:2]
            video_path = os.path.join(behavior_path, f'seq_{seq_num}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            for i, state in enumerate(states):
                if state.shape[-1] == 1:
                    state = cv2.cvtColor(state, cv2.COLOR_GRAY2RGB)
                elif state.shape[-1] == 3:
                    state = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
                
                out.write(state)
                print(f"Added frame {i} to {video_path}")
            out.release()
            print(f"Saved video: {video_path}")



def save_sequences(behaviors, savepath='visuals_attvq'):
    for idx, sequences in behaviors.items():
        behavior_path = os.path.join(savepath, f'behavior_{idx}')
        os.makedirs(behavior_path, exist_ok=True)
        
        for seq_num, states in enumerate(sequences):
            sequence_path = os.path.join(behavior_path, f'seq_{seq_num}')
            os.makedirs(sequence_path, exist_ok=True)
            
            for i, state in enumerate(states):
                img = Image.fromarray(state)
                state_save_path = os.path.join(sequence_path, f'state_{i}.png')
                img.save(state_save_path)
                print(f"Saved: {state_save_path}")

def evaluate_model(model, n_sequences, segment_len=10, save=False):
    model.eval()
    results = {}
    
    with torch.no_grad():
        for seq_id in range(n_sequences):
            start_idx = seq_id * segment_len
            end_idx = start_idx + segment_len
            
            if end_idx > sequences.shape[0]:
                print(f"Sequence {seq_id} exceeds available data, stopping.")
                break
            
            src = sequences[start_idx:end_idx].unsqueeze(0).to(device)  # Single sequence batch
            tgt = src.clone()  # Evaluation target is the same sequence
            src_mask = generate_square_subsequent_mask(segment_len).to(device)
            tgt_mask = generate_square_subsequent_mask(segment_len).to(device)
            output, vq_loss, idx = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

            src_states_np = src.squeeze(0).cpu().numpy()
            src_states = [render_halfcheetah(env,item[:17]) for item in src_states_np]

            results[seq_id] = {
                'states': src_states,
                'quantized_indices': idx.view(-1).cpu().numpy()
            }
            
            print(f"Evaluated sequence {seq_id}: quantized indices - {results[seq_id]['quantized_indices']}")

        behaviors = {}
        if save: 
            for k,v in results.items():
                states = v['states']
                idx = v['quantized_indices'][0]
                if idx in behaviors:
                    behaviors[idx].append(states)
                else: 
                    behaviors[idx] = [states]
            save_sequences_as_videos(behaviors, savepath='visuals_attvq6')
    
    return results

evaluate_model(model, 50, 25, True)