import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Add the path to the VJepa codebase
sys.path.append('/gpfs/home/unalg01/jepa')

# Import necessary modules from VJepa
import src.models.vision_transformer as vit
from src.datasets.data_manager import init_data
from evals.video_classification_frozen.utils import make_transforms

# Set random seeds for reproducibility
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='t-SNE Visualization of Encoded Features')
    parser.add_argument('--pretrained_path', type=str, default='/gpfs/data/sodicksonlab/gozde/logs/mnist/jepa-latest.pth.tar', #required=True,
                        help='Path to the pretrained encoder checkpoint')
    parser.add_argument('--checkpoint_key', type=str, default='target_encoder',
                        help='Key for the encoder in the checkpoint')
    parser.add_argument('--model_name', type=str, default='vit_large',
                        help='Name of the encoder model architecture')
    parser.add_argument('--patch_size', type=int, default=7,
                        help='Patch size for the encoder')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Crop size for the input images')
    parser.add_argument('--tubelet_size', type=int, default=2,
                        help='Tubelet size for video data')
    parser.add_argument('--frames_per_clip', type=int, default=16,
                        help='Number of frames per clip (for video data)')
    parser.add_argument('--in_chans', type=int, default=3,
                        help='Number of input channels (e.g., 1 for grayscale)')
    parser.add_argument('--dataset_type', type=str, default='MRIDataset',
                        choices=['VideoDataset', 'MRIDataset'],
                        help='Type of dataset to use')
    parser.add_argument('--data_path', type=str, default='/gpfs/home/unalg01/jepa/src/datasets/mnist3d/nii_volumes.csv', #required=True,
                        help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker threads for data loading')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes in the dataset')
    parser.add_argument('--output_path', type=str, default='tsne_plot.png',
                        help='Path to save the t-SNE plot')
    parser.add_argument('--subset_file', type=str, default=None,
                        help='Path to a subset file if using a subset of data')
    parser.add_argument('--max_samples', type=int, default=300,
                        help='Maximum number of samples to use for t-SNE')
    args = parser.parse_args()
    return args

def load_pretrained_encoder(args, device):
    """
    Loads the pretrained encoder model.
    """
    # Initialize the encoder model
    encoder = vit.__dict__[args.model_name](
        img_size=args.crop_size,
        patch_size=args.patch_size,
        num_frames=args.frames_per_clip,
        tubelet_size=args.tubelet_size,
        in_chans=args.in_chans,
    )

    # Load the pretrained weights
    checkpoint = torch.load(args.pretrained_path, map_location='cpu')
    try:
        pretrained_dict = checkpoint[args.checkpoint_key]
    except KeyError:
        pretrained_dict = checkpoint['encoder']

    # Adjust keys if necessary
    pretrained_dict = {k.replace('module.', ''): v for k, v in
                       pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in
                       pretrained_dict.items()}

    # Load the state dict, handling mismatches
    model_dict = encoder.state_dict()
    for k, v in model_dict.items():
        if k in pretrained_dict and pretrained_dict[k].shape == v.shape:
            model_dict[k] = pretrained_dict[k]
        else:
            print(f"Layer {k} not loaded from checkpoint")
    encoder.load_state_dict(model_dict)
    encoder.to(device)
    encoder.eval()  # Set to evaluation mode
    return encoder

def make_dataloader(args):
    """
    Creates a DataLoader for the dataset.
    """
    # Define the data transformations
    transform = make_transforms(
        training=False,
        num_views_per_clip=1,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4/3),
        random_resize_scale=(0.08, 1.0),
        reprob=0,
        auto_augment=False,
        motion_shift=False,
        crop_size=args.crop_size,
        in_chans=args.in_chans,
    )

    # Initialize the data loader
    data_loader, _ = init_data(
        data=args.dataset_type,
        root_path=[args.data_path],
        transform=transform,
        batch_size=args.batch_size,
        world_size=1,
        rank=0,
        clip_len=args.frames_per_clip,
        frame_sample_rate=1,
        duration=None,
        num_clips=1,
        in_chans=args.in_chans,
        crop_size=args.crop_size,
        random_clip_sampling=False,
        allow_clip_overlap=True,
        num_workers=args.num_workers,
        copy_data=False,
        drop_last=False,
        subset_file=args.subset_file)
    return data_loader

def extract_features(encoder, data_loader, device, args):
    """
    Extracts features from the data using the pretrained encoder.
    """
    features = []
    labels = []
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            # Get the inputs and labels
            inputs = batch[0]  # Inputs
            batch_labels = batch[1]  # Labels

            # Move data to the device
            if isinstance(inputs[0], list) or isinstance(inputs[0], tuple):
                # For video data with multiple clips
                # Flatten the list of lists and move to device
                inputs = [clip.to(device) for clips in inputs for clip in clips]
                batch_size = batch_labels.size(0)
            else:
                # For image data or video data with single clip
                inputs = [clip.to(device) for clip in inputs]
                batch_size = batch_labels.size(0)

            batch_labels = batch_labels.numpy()

            # Pass inputs through the encoder
            outputs = encoder(inputs)

            # outputs shape: [batch_size, K, embed_dim]
            # K is the number of tokens (patches)

            # Ensure outputs is a tensor
            if isinstance(outputs, list):
                # If outputs is a list (e.g., for multiple clips), concatenate along batch dimension
                outputs = torch.cat(outputs, dim=0)  # Shape: [batch_size * num_clips, K, embed_dim]

            # Global Average Pooling over the tokens (patches)
            # outputs shape: [batch_size, K, embed_dim]
            # We need to average over the K dimension (dim=1)
            batch_features = outputs.mean(dim=1)  # Shape: [batch_size, embed_dim]

            # If batch_features does not match batch_size, adjust accordingly
            # For example, if you processed multiple clips per sample and concatenated along batch dimension
            if batch_features.shape[0] != batch_size:
                # Assuming that the outputs are ordered per sample and per clip
                # We can reshape and then average over clips
                num_clips_per_sample = batch_features.shape[0] // batch_size
                batch_features = batch_features.view(batch_size, num_clips_per_sample, -1)
                batch_features = batch_features.mean(dim=1)  # Average over clips

            # Convert to numpy arrays
            batch_features = batch_features.cpu().numpy()

            # Collect features and labels
            features.append(batch_features)
            labels.append(batch_labels)

            total_samples += batch_features.shape[0]
            if total_samples >= args.max_samples:
                break  # Limit the number of samples for t-SNE

    # Concatenate all batches
    features = np.concatenate(features, axis=0)[:args.max_samples]
    labels = np.concatenate(labels, axis=0)[:args.max_samples]
    return features, labels

def plot_tsne(features, labels, args):
    """
    Applies t-SNE to the features and plots the embeddings.
    """
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(features)

    # Create a scatter plot
    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    markers = ['o', '+', 'x', 's', 'd', '^', 'v', '<', '>', 'p', '*']
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for idx, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                    marker=markers[idx % len(markers)],
                    color=colors(idx % 10),
                    label=f'Class {label}', alpha=0.7)
    plt.legend()
    plt.title('t-SNE of Encoded Features')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.savefig(args.output_path)
    plt.show()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Set the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pretrained encoder
    encoder = load_pretrained_encoder(args, device)

    # Create the data loader
    data_loader = make_dataloader(args)

    # Extract features and labels
    features, labels = extract_features(encoder, data_loader, device, args)

    # Apply t-SNE and plot the embeddings
    plot_tsne(features, labels, args)

if __name__ == '__main__':
    main()
