import argparse
import torch
from torch.utils.data import Dataset
import d4rl
import gym

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='Enable wandb logging')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--initial_tf', type=float, default=0.5)
    parser.add_argument('--tf_decay', type=float, default=0.85)
    parser.add_argument('--tag', type=str, default="vqvae")
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--num_tokens', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--context_len', type=int, default=50)
    parser.add_argument('--dataset_type', type=str, default='overlap', help='Overlap sequences')
    parser.add_argument('--weights_path', type=str, default=None)

    return parser.parse_args()


class D4RLStateActionDatasetNo(Dataset):
    def __init__(self, states, actions, context_len=20):
        self.context_len = context_len
        self.data = []

        # Step by context_len instead of 1 to avoid overlap
        for i in range(0, len(states) - context_len - 1, context_len):
            seq_states = states[i:i+context_len]
            seq_actions = actions[i:i+context_len]
            target_states = states[i+1:i+1+context_len]
            self.data.append({
                "states": seq_states,
                "actions": seq_actions,
                "targets": target_states
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.data[idx]["states"], dtype=torch.float32),
            "actions": torch.tensor(self.data[idx]["actions"], dtype=torch.float32),
            "targets": torch.tensor(self.data[idx]["targets"], dtype=torch.float32),
        }



class D4RLStateActionDataset(Dataset):
    def __init__(self, states, actions, context_len=20):
        self.context_len = context_len
        self.data = []

        for i in range(len(states) - context_len - 1):
            seq_states = states[i:i+context_len]
            seq_actions = actions[i:i+context_len]
            target_states = states[i+1:i+1+context_len]
            self.data.append({
                "states": seq_states,
                "actions": seq_actions,
                "targets": target_states
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.data[idx]["states"], dtype=torch.float32),
            "actions": torch.tensor(self.data[idx]["actions"], dtype=torch.float32),
            "targets": torch.tensor(self.data[idx]["targets"], dtype=torch.float32),
        }

def load_d4rl_data(env_name):
    env = gym.make(env_name)
    dataset = env.get_dataset()
    states = dataset['observations']
    actions = dataset['actions']
    return states, actions







def create_run_name(args):
    # Create run name based on arguments, use all items in args (argparse object)
    run_name = "_".join([f"{k}={v}" for k, v in vars(args).items()])
    return run_name