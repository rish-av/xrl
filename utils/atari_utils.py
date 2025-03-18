import numpy as np
import torch
from torch.utils.data import Dataset
import os
import gzip
import argparse
import cv2
from torch.utils.data import DataLoader

def get_sequence_codes(model, dataset, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    codes = []
    all_states = []
    for batch_idx, batch in enumerate(dataloader):
        states, actions, target_states = batch
        states = states.to(device)
        actions = actions.to(device).long()
        target_states = target_states.to(device)
        all_states.append(states)
        _, _, indices = model(states, actions, target_states, sampling_probability=0.0)
        codes.append(indices)
    return codes, all_states


def save_frames_atari(states, predicted_frames, epoch, batch_idx, max_examples=5, max_frames=30, folder_prefix="frames"):
    os.makedirs(f'{folder_prefix}_original', exist_ok=True)
    os.makedirs(f'{folder_prefix}_predicted', exist_ok=True)

    for example_idx in range(min(max_examples, states.size(0))):  # Save up to max_examples examples per batch
        for frame_idx in range(min(max_frames, states.size(1))):  # Save up to max_frames per sequence
            original_frame = (states[example_idx, frame_idx, 0].cpu().numpy() * 255).astype('uint8')
            predicted_frame = (predicted_frames[example_idx, frame_idx, 0].detach().cpu().numpy() * 255).astype('uint8')

            # Save original frame
            cv2.imwrite(f'{folder_prefix}_original/epoch_{epoch}_batch_{batch_idx}_example_{example_idx}_frame_{frame_idx}.png', original_frame)
            
            # Save predicted frame
            cv2.imwrite(f'{folder_prefix}_predicted/epoch_{epoch}_batch_{batch_idx}_example_{example_idx}_frame_{frame_idx}.png', predicted_frame)



def get_atari_args():
    parser = argparse.ArgumentParser(description="Train Atari Model with Custom Model Name in WandB")
    
    # Model hyperparameters
    parser.add_argument('--max_seq_len', type=int, default=60, help="Maximum sequence length")
    parser.add_argument('--embed_dim', type=int, default=128, help="Embedding dimension")
    parser.add_argument('--nhead', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of transformer layers")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--num_embeddings', type=int, default=128, help="Number of embeddings for VQ")

    # Logging and experiment tracking
    parser.add_argument('--log', action='store_true', help="Enable logging with WandB")
    parser.add_argument('--wandb_project', type=str, default="atari-xrl-final", help="WandB project name")
    parser.add_argument('--wandb_entity', type=str, default="mail-rishav9", help="WandB entity name")
    parser.add_argument('--wandb_run_name', type=str, default="xrl_atari", help="Base name for WandB run")
    parser.add_argument('--wandb_project_name', type=str, default="atari-xrl-final", help="WandB project name")
    parser.add_argument('--patch_size', type=int, default=4, help="Patch size for image transformer")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument('--load_checkpoint', type=str, default=None, help="Path to load model checkpoint")
    parser.add_argument('--epochs', type=int, default=2000, help="Number of training epochs")
    parser.add_argument('--scheduler_step_size', type=int, default=50, help="Step size for learning rate scheduler")
    parser.add_argument('--frame_skip', type=int, default=4, help="Number of frames to skip in dataset")
    parser.add_argument('--save_frames', action='store_true', help="Save frames")
    parser.add_argument('--save_frame_freq', type=int, default=1000, help="Frequency to save frames")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--dataset_type', type=str, default='overlapping', help="Dataset type (non-overlapping or grayscale)")
    return parser.parse_args()


class OfflineEnvAtari:

    def __init__(self,
                 game=None,
                 index=None,
                 start_epoch= 0,
                 last_epoch= 1,
                 stack=False,
                 path='./datasets'):
        self.game = game
        self.index = index
        self.start_epoch = start_epoch
        self.last_epoch = last_epoch
        self.stack = stack
        self.path = path

    def get_dataset(self):
        observation_stack = []
        action_stack = []
        reward_stack = []
        terminal_stack = []
        for epoch in range(self.start_epoch, self.last_epoch):

            observations = _load('observation', self.path)
            actions = _load('action', self.path)
            rewards = _load('reward', self.path)
            terminals = _load('terminal', self.path)

            # sanity check
            assert observations.shape == (1000000, 84, 84)
            assert actions.shape == (1000000, )
            assert rewards.shape == (1000000, )
            assert terminals.shape == (1000000, )

            observation_stack.append(observations)
            action_stack.append(actions)
            reward_stack.append(rewards)
            terminal_stack.append(terminals)

        if len(observation_stack) > 1:
            observations = np.vstack(observation_stack)
            actions = np.vstack(action_stack).reshape(-1)
            rewards = np.vstack(reward_stack).reshape(-1)
            terminals = np.vstack(terminal_stack).reshape(-1)
        else:
            observations = observation_stack[0]
            actions = action_stack[0]
            rewards = reward_stack[0]
            terminals = terminal_stack[0]

        # memory-efficient stacking
        if self.stack:
            observations = _stack(observations, terminals)
        else:
            observations = observations.reshape(-1, 1, 84, 84)

        data_dict = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'terminals': terminals
        }

        return data_dict

def _stack(observations, terminals, n_channels=4):
    rets = []
    t = 1
    for i in range(observations.shape[0]):
        if t < n_channels:
            padding_shape = (n_channels - t, ) + observations.shape[1:]
            padding = np.zeros(padding_shape, dtype=np.uint8)
            observation = observations[i - t + 1:i + 1]
            observation = np.vstack([padding, observation])
        else:
            # avoid copying data
            observation = observations[i - n_channels + 1:i + 1]

        rets.append(observation)

        if terminals[i]:
            t = 1
        else:
            t += 1
    return rets


def _load(name, dir_path):
    path = os.path.join(dir_path, name + '.gz')
    with gzip.open(path, 'rb') as f:
        print('loading {}...'.format(path))
        return np.load(f, allow_pickle=False)

class AtariGrayscaleDataset(Dataset):
    def __init__(self, dataset_path, max_seq_len=30, transform=None, frame_skip=4):
        """
        Atari Grayscale Dataset for sequence-to-sequence modeling with frame skipping.

        Args:
            dataset_path (str): Path to the dataset.
            max_seq_len (int): Maximum sequence length for each sample.
            transform (callable, optional): Optional transform to apply to frames.
            frame_skip (int): Number of frames to skip (default is 1, i.e., no skipping).
        """
        self.dataset = OfflineEnvAtari(stack=False, path=dataset_path).get_dataset()  # Load metadata
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.frame_skip = frame_skip

        self.total_frames = len(self.dataset['observations'])  # Total number of frames
        self.valid_starts = self._get_valid_start_indices()

    def _get_valid_start_indices(self):
        """
        Identify valid starting indices for sequences, avoiding episode boundaries.
        """
        valid_starts = []
        for i in range(0, self.total_frames - self.max_seq_len * self.frame_skip, self.frame_skip):
            # Check for terminal flags if available
            if 'terminals' in self.dataset:
                terminals = self.dataset['terminals'][i:i+self.max_seq_len*self.frame_skip:self.frame_skip]
                if not np.any(terminals):  # No terminal flag in sequence
                    valid_starts.append(i)
            else:
                valid_starts.append(i)  # Assume all starts are valid if no terminal flags
        return valid_starts

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        """
        Get a sequence of states, actions, and the corresponding target states.

        Args:
            idx (int): Index of the sequence start.

        Returns:
            tuple: (states, actions, target_states)
        """
        start_idx = self.valid_starts[idx]
        indices = range(start_idx, start_idx + self.max_seq_len * self.frame_skip, self.frame_skip)

        states = [self.dataset['observations'][i] for i in indices]
        actions = [self.dataset['actions'][i] for i in indices]
        target_states = [self.dataset['observations'][i+1] for i in indices]

        # Normalize frames
        states = np.array(states) / 255.0
        target_states = np.array(target_states) / 255.0

        # Apply transform if provided
        if self.transform:
            states = self.transform(states)
            target_states = self.transform(target_states)

        # Convert to tensors
        states = torch.FloatTensor(states) # Add channel dimension
        actions = torch.LongTensor(actions)
        target_states = torch.FloatTensor(target_states) # Add channel dimension

        return states, actions, target_states
    


class AtariNonOverlappingDataset(Dataset):
    def __init__(self, dataset_path, max_seq_len=30, transform=None, frame_skip=4):
        self.dataset = OfflineEnvAtari(stack=False, path=dataset_path).get_dataset()
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.frame_skip = frame_skip

        self.total_frames = len(self.dataset['observations'])
        self.valid_starts = self._get_non_overlapping_start_indices()

    def _get_non_overlapping_start_indices(self):
        valid_starts = []
        step_size = self.max_seq_len * self.frame_skip
        for i in range(0, self.total_frames - step_size, step_size):
            if 'terminals' in self.dataset:
                terminals = self.dataset['terminals'][i:i + step_size:self.frame_skip]
                if not np.any(terminals):
                    valid_starts.append(i)
            else:
                valid_starts.append(i)
        return valid_starts

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]
        indices = range(start_idx, start_idx + self.max_seq_len * self.frame_skip, self.frame_skip)

        states = [self.dataset['observations'][i] for i in indices]
        actions = [self.dataset['actions'][i] for i in indices]
        target_states = [self.dataset['observations'][i + 1] for i in indices]

        states = np.array(states) / 255.0
        target_states = np.array(target_states) / 255.0

        if self.transform:
            states = self.transform(states)
            target_states = self.transform(target_states)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        target_states = torch.FloatTensor(target_states)

        return states, actions, target_states
