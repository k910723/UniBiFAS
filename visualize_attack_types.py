"""
Script to visualize attack-type specific t-SNE using the trained OCM->I model.
Uses visualization dataloader to load live, print, and replay samples separately.
"""

import yaml
import os
import torch
import warnings
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from utils import Logger
from modeling.unibifas import UniBiFAS_Model
from data.datasets.visualization_dataloader import BuildVisualizationLoader

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", message="Failed to load image Python extension")


def load_model_checkpoint(checkpoint_path, cfg, device):
    """Load a model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = UniBiFAS_Model(cfg).to(device)
    
    # Load model weights - try both possible keys
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise KeyError("No model weights found in checkpoint.")
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    if 'best_AUC' in checkpoint:
        print(f"  Best AUC: {checkpoint['best_AUC']:.4f}")
    if 'best_HTER' in checkpoint:
        print(f"  Best HTER: {checkpoint['best_HTER']:.4f}")
    
    return model


def extract_features_by_attack(model, dataloader, device, max_samples_per_class=500):
    """
    Extract features from the model, organized by attack type.
    Randomly samples from all available data to get better distribution.
    
    Returns:
        features: numpy array of features
        labels: numpy array of attack type labels (0: live, 1: print, 2: replay)
    """
    model.eval()
    
    # First pass: collect ALL features
    all_features = {0: [], 1: [], 2: []}  # Organize by attack type
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            img, attack_labels = batch
            
            img = img.to(device)
            attack_labels = attack_labels.cpu().numpy()
            
            # Forward pass to get features
            img_feat_norm, cls_tokens, patch_tokens, text_feat_b, text_feat_a, text_feat_art = model(img)
            
            # Use the normalized image features for visualization
            features = img_feat_norm.cpu().numpy()
            
            # Organize by attack type
            for i in range(len(features)):
                label = attack_labels[i]
                all_features[label].append(features[i])
    
    # Second pass: randomly sample max_samples_per_class from each attack type
    import random
    features_list = []
    labels_list = []
    
    for label in [0, 1, 2]:
        label_features = all_features[label]
        n_available = len(label_features)
        
        if n_available > 0:
            # Randomly sample up to max_samples_per_class
            n_samples = min(max_samples_per_class, n_available)
            sampled_indices = random.sample(range(n_available), n_samples)
            
            for idx in sampled_indices:
                features_list.append(label_features[idx])
                labels_list.append(label)
            
            print(f"  Attack type {label}: sampled {n_samples} from {n_available} available samples")
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    print(f"\nTotal extracted features: {len(features)}")
    print(f"  Live: {np.sum(labels == 0)} samples")
    print(f"  Print: {np.sum(labels == 1)} samples")
    print(f"  Replay: {np.sum(labels == 2)} samples")
    
    return features, labels


def visualize_attack_tsne(features, labels, save_path, title="t-SNE by Attack Type"):
    """
    Create t-SNE visualization colored by attack type.
    """
    n_samples = len(features)
    print(f"\nRunning t-SNE on {n_samples} samples...")
    print(f"  Live: {np.sum(labels == 0)} samples")
    print(f"  Print: {np.sum(labels == 1)} samples")
    print(f"  Replay: {np.sum(labels == 2)} samples")
    
    # Validate features
    if not np.all(np.isfinite(features)):
        bad_count = np.size(features) - np.count_nonzero(np.isfinite(features))
        print(f"Warning: Found {bad_count} non-finite values. Replacing with 0.")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Adjust perplexity based on sample size
    perplexity = 30
    max_perplexity = max(1, int((n_samples - 1) / 3) - 1)
    if perplexity > max_perplexity:
        print(f"Adjusting perplexity from {perplexity} to {max_perplexity}")
        perplexity = max_perplexity
    
    # Run t-SNE
    method = 'barnes_hut' if n_samples > 1000 else 'exact'
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000,
                   random_state=42, method=method, init='pca')
        tsne_results = tsne.fit_transform(features)
    except Exception as e:
        print(f"t-SNE failed: {e}. Falling back to PCA.")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        tsne_results = pca.fit_transform(features)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Colors: Green for live, Red for print, Orange for replay
    colors = ['#00D084', '#FF9F1C', '#E63946']  # Teal green, Red, Orange
    markers = ['o', 'o', 'o']
    labels_text = ['Live', 'Print Attack', 'Replay Attack']
    
    # Create masks for each attack type
    live_mask = (labels == 0)
    print_mask = (labels == 1)
    replay_mask = (labels == 2)
    masks = [live_mask, print_mask, replay_mask]
    
    # Plot each attack type
    for mask, color, marker, label in zip(masks, colors, markers, labels_text):
        if np.sum(mask) > 0:
            ax.scatter(
                tsne_results[mask, 0],
                tsne_results[mask, 1],
                c=color,
                marker=marker,
                s=150,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5,
                label=f'{label} (n={np.sum(mask)})'
            )
    
    # Remove axes, title, legend, and grid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nt-SNE visualization saved to: {save_path}")
    plt.close()


if __name__ == '__main__':
    # Configuration
    CONFIG_PATH = "configs/unibifas.yaml"
    CHECKPOINT_PATH = "output/CIM_O_best.pt"  # model checkpoint
    MAX_SAMPLES = 200 # Samples per attack type
    
    # Load config
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Update config for visualization
    cfg['dataset']['target'] = 'O'  # Target is Idiap (I)
    target = cfg['dataset']['target']
    
    print("\n" + "="*80)
    print(f"t-SNE Visualization by Attack Type on {target}")
    print("="*80)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model_checkpoint(CHECKPOINT_PATH, cfg, device)
    model.eval()
    
    # Build visualization dataloader
    print("\nBuilding visualization dataloader...")
    vis_loader = BuildVisualizationLoader(cfg)
    print(f"Loaded {len(vis_loader.dataset)} total samples")
    
    # Extract features
    print("\nExtracting features...")
    features, labels = extract_features_by_attack(model, vis_loader, device, MAX_SAMPLES)
    
    # Create visualization directory
    os.makedirs('tsne_visualizations', exist_ok=True)
    save_path = f'tsne_visualizations/attack_type.png'
    #save_path = 'tsne_visualizations/attack_type_ocm_to_i.png'
    
    # Generate t-SNE visualization
    visualize_attack_tsne(features, labels, save_path)
    
    print("\n" + "="*80)
    print("Visualization Complete!")
    print("="*80 + "\n")
