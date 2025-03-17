import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from utils.general_utils import save_model_checkpoint, generate_run_name
from utils.atari_utils import get_atari_args, AtariGrayscaleDataset, save_frames_atari, get_sequence_codes
from models.atari_models import AtariSeq2SeqTransformerVQ
import cv2
import os


args = get_atari_args()
def train_model(model, dataset, epochs=2000, batch_size=4, lr=1e-4, scheduler_step_size=50, scheduler_gamma=0.9, save_interval=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sampling_probability = 0.0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            states, actions, target_states = batch
            states = states.to(device)
            actions = actions.to(device).long()
            target_states = target_states.to(device)



            # s0,s1,s2,s3 -> s1,s2,s3,s4
            # targrt[:, 1:] -> s2,s3,s4 

            # Forward pass
            predicted_frames, vq_loss, indices = model(states, actions, target_states)
            reconstruction_loss = criterion(predicted_frames, target_states[:, 1:])
            loss = reconstruction_loss + vq_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate batch loss
            epoch_loss += loss.item()

            if args.log:
                wandb.log({
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "loss": loss.item(),
                    "reconstruction_loss": reconstruction_loss.item(),
                    "vq_loss": vq_loss.item(),
                    "unique_indices": torch.unique(indices).size(0)
                })

            # Logging loss for each batch
            if indices is not None:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Recon Loss: {reconstruction_loss.item():.4f}, VQ Loss: {vq_loss.item()}, Unique Indices: {torch.unique(indices)} ")

                for k, v in model.vq.get_codebook_metrics().items():
                    print(f"{k}: {v:.4f}")
            else:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}],  Recon Loss: {reconstruction_loss.item():.4f}, VQ Loss: {vq_loss.item}")

            if args.log:
                metrics = model.vq.get_codebook_metrics()
                for k,v in metrics.items():
                    wandb.log({k: v})

            if batch_idx > 0 and args.save_frames and batch_idx % args.save_frame_freq == 0:
                save_frames_atari(states, predicted_frames, epoch, batch_idx, folder_prefix=f"frames_{generate_run_name(args)}")
            
            if batch_idx % 1000 == 0:
                torch.save(model.state_dict(), f"checkpoints/model_vq2_{generate_run_name(args)}_epoch_{epoch}_batch_{batch_idx}.pth")


        # Average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}")


        ####TODO: make this a parameter
        sampling_probability = sampling_probability * 0.90

        # Scheduler step
        scheduler.step()

        # # Save the model checkpoint every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            save_model_checkpoint(model, epoch + 1)
            print(f"Model checkpoint saved at epoch {epoch + 1}")






if __name__ == "__main__":


    if args.log:
        run_name = generate_run_name(args)
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, name=run_name, config=vars(args))

    model = AtariSeq2SeqTransformerVQ(
        img_size=(84, 84),
        # patch_size=4,
        embed_dim=args.embed_dim,
        num_heads=args.nhead,
        num_layers=args.num_layers,
        action_dim=18,
        num_embeddings=args.num_embeddings,
        max_seq_len=args.max_seq_len,
        ).to("cuda")

    if args.load_checkpoint is not None:
        model.load_state_dict(torch.load(args.load_checkpoint, weights_only=True))
        print(f"Model loaded from {args.load_checkpoint}")


    if args.dataset_type == 'overlapping':
        dataset = AtariGrayscaleDataset( dataset_path=args.dataset_path, max_seq_len=args.max_seq_len, frame_skip=args.frame_skip)
    else:
        from utilsf.atari_utils import AtariNonOverlappingDataset
        print("Loading non-overlapping dataset")
        dataset = AtariNonOverlappingDataset(dataset_path=args.dataset_path, max_seq_len=args.max_seq_len, frame_skip=args.frame_skip)

    print(f"Dataset loaded with {len(dataset)} sequences.")


    

    # Define training parameters
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    scheduler_step_size = args.scheduler_step_size
    scheduler_gamma = 0.9
    train_model(model, dataset, epochs=epochs, batch_size=batch_size, lr=learning_rate, scheduler_step_size=scheduler_step_size, scheduler_gamma=scheduler_gamma)


    #once trained build the graph
    with torch.no_grad():
        model.eval()

        codebook_vectors = model.vq.codebook.weight.cpu().numpy()
        usage_probs = model.vq.usage_count / model.vq.usage_count.sum()

        active_codes = (usage_probs > 0.01).cpu().numpy()
        active_codebook = codebook_vectors[active_codes]
        from utils import plot_pca_clusters

        codes, statesf = get_sequence_codes(model, dataset, batch_size=batch_size)
        all_codes = []
        for batch_code in codes:
            batch_code = batch_code.reshape(-1, args.max_seq_len)
            for code in batch_code:
                all_codes.append(code.cpu().numpy())

        from spectral_graph import SpectralGraphPartitioner

        partitioner = SpectralGraphPartitioner(n_clusters=None, scaling='minmax', similarity_mode='combined', alpha=0.6)
        transition_matrix, unique_ids = partitioner.build_transition_matrix(all_codes)
        node_vectors = codebook_vectors[unique_ids]
        labels = partitioner.fit_transform(node_vectors, transition_matrix)
        labels = partitioner._merge_small_close_clusters(node_vectors, labels, similarity_threshold=0.85, size_threshold=5)
        partitioner.visualize_similarity(node_vectors, transition_matrix)
        partitioner.render_partitioned_graph(node_vectors, transition_matrix, labels, combine_nodes=True, edge_threshold=0.45)

        