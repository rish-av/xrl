
import os
import cv2
import torch

def generate_run_name(args):
    """
    Generates a unique run name by combining key hyperparameters.
    Args:
        args: Parsed arguments from argparse.

    Returns:
        str: A descriptive run name.
    """
    run_name_parts = [
        args.wandb_run_name,  # Base run name
        f"seq{args.max_seq_len}",  # Sequence length
        f"embed{args.embed_dim}",  # Embedding dimension
        f"heads{args.nhead}",  # Attention heads
        f"layers{args.num_layers}",  # Transformer layers
        f"batch{args.batch_size}",  # Batch size
        f"lr{args.learning_rate:.0e}"  # Learning rate (scientific notation)
        f"vq{args.num_embeddings}"  # Number of embeddings for VQ
        f"skip{args.frame_skip}"
    ]
    
    # Filter out any empty parts and join with underscores
    run_name = "_".join(map(str, run_name_parts))
    return run_name





def save_model_checkpoint(model, epoch, save_dir="checkpoints"):
    """
    Save the model checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved at {checkpoint_path}")