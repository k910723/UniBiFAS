"""
t-SNE Visualization for Face Anti-Spoofing

This module provides visualization tools to analyze the feature space learned by the model.

USAGE FLAGS (set in train_and_visualize.py or visualize_only.py):
    - INCLUDE_SOURCE: Set to True to show both source and target domains
                      Set to False to show target domain only (default)
    
    - FILTER_CORRECT_ONLY: Set to True to only visualize correctly predicted samples
                            Set to False to show all samples (default: True)

OUTPUT:
    - Target only, all samples: tsne_C_to_I_target_only.png
    - Target only, correct: tsne_C_to_I_correct_only_target_only.png
    - Source+Target, all: tsne_C_to_I.png
    - Source+Target, correct: tsne_C_to_I_correct_only.png
"""

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from torch.nn import functional as F

def extract_features(model, dataloader, device, max_samples_per_class=500, filter_correct=False):
    """
    Extract features from the model for t-SNE visualization.
    
    Args:
        model: The trained model
        dataloader: DataLoader for the dataset
        device: cuda or cpu
        max_samples_per_class: Maximum number of samples to extract per class to avoid memory issues
        filter_correct: If True, only include correctly predicted samples
    
    Returns:
        features: numpy array of features
        labels: numpy array of labels (0: real, 1: fake)
    """
    model.eval()
    
    features_list = []
    labels_list = []
    
    # Track how many samples we've collected per class
    real_count = 0
    fake_count = 0
    
    # Track incorrectly predicted samples if filtering
    incorrect_count = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            # Handle both training and test data formats
            if len(batch) == 3:
                # Training data: (img, scm, labels)
                img, _, labels = batch
                binary_labels = labels[:, 0]  # Extract binary labels
            else:
                # Test data: (img, labels)
                img, labels = batch
                binary_labels = labels[:, 0] if labels.dim() > 1 else labels
            
            img = img.to(device)
            binary_labels_tensor = binary_labels.to(device) if isinstance(binary_labels, torch.Tensor) else torch.tensor(binary_labels, device=device)
            binary_labels = binary_labels_tensor.cpu().numpy()
            
            # Forward pass to get features
            img_feat_norm, cls_tokens, patch_tokens, text_feat_b, text_feat_a, text_feat_art = model(img)
            
            # Get predictions if filtering is enabled
            if filter_correct:
                logit_scale = model.logit_scale.exp()
                logits_binary = logit_scale * img_feat_norm @ text_feat_b.t()
                predictions = torch.argmax(logits_binary, dim=1).cpu().numpy()
            
            # Use the normalized image features for visualization
            features = img_feat_norm.cpu().numpy()
            
            # Filter samples based on max_samples_per_class
            for i in range(len(features)):
                label = binary_labels[i]
                
                # If filter_correct is True, skip incorrectly predicted samples
                if filter_correct:
                    if predictions[i] != label:
                        incorrect_count += 1
                        continue
                
                if label == 0 and real_count < max_samples_per_class:
                    features_list.append(features[i])
                    labels_list.append(label)
                    real_count += 1
                elif label == 1 and fake_count < max_samples_per_class:
                    features_list.append(features[i])
                    labels_list.append(label)
                    fake_count += 1
            
            # Stop if we have enough samples
            if real_count >= max_samples_per_class and fake_count >= max_samples_per_class:
                break
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    print(f"Extracted {len(features)} features: {real_count} real, {fake_count} fake")
    if filter_correct:
        print(f"  Filtered out {incorrect_count} incorrectly predicted samples")
    
    return features, labels


def visualize_tsne(source_features, source_labels, target_features, target_labels, 
                   save_path, title="t-SNE Visualization", perplexity=30, n_iter=1000, 
                   include_source=False):
    """
    Create t-SNE visualization for source and target features.
    
    Args:
        source_features: Features from source domain
        source_labels: Labels from source domain (0: real, 1: fake)
        target_features: Features from target domain
        target_labels: Labels from target domain (0: real, 1: fake)
        save_path: Path to save the visualization
        title: Title of the plot
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations for t-SNE
        include_source: If True, include source domain samples in visualization
    """
    # Combine source and target features based on include_source flag
    if include_source:
        all_features = np.vstack([source_features, target_features])
        # Create domain labels: 0 for source, 1 for target
        source_domain = np.zeros(len(source_features))
        target_domain = np.ones(len(target_features))
        domain_labels = np.concatenate([source_domain, target_domain])
        # Combine binary labels
        all_labels = np.concatenate([source_labels, target_labels])
    else:
        # Only use target features
        all_features = target_features
        domain_labels = np.ones(len(target_features))  # All are target
        all_labels = target_labels
    
    n_samples = len(all_features)
    print(f"Running t-SNE on {n_samples} samples...")
    if include_source:
        print(f"  Source: {len(source_features)} samples ({np.sum(source_labels == 0)} real, {np.sum(source_labels == 1)} fake)")
    print(f"  Target: {len(target_features)} samples ({np.sum(target_labels == 0)} real, {np.sum(target_labels == 1)} fake)")

    # Validate features (no NaN/Inf). Replace bad values with zeros and warn.
    if not np.all(np.isfinite(all_features)):
        bad_count = np.size(all_features) - np.count_nonzero(np.isfinite(all_features))
        print(f"Warning: Found {bad_count} non-finite feature values. Replacing with 0.")
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)

    # Adjust perplexity to be valid for the number of samples.
    # sklearn's TSNE requires perplexity < n_samples. A safe heuristic is perplexity <= (n_samples - 1) / 3.
    max_perplexity = max(1, int((n_samples - 1) / 3) - 1)
    if perplexity > max_perplexity:
        print(f"Warning: requested perplexity={perplexity} is too large for {n_samples} samples. Reducing to {max_perplexity}.")
        perplexity = max_perplexity

    # Choose method: 'barnes_hut' for larger datasets, 'exact' for smaller ones
    method = 'barnes_hut' if n_samples > 1000 else 'exact'

    # Apply t-SNE with a safe try/except. Fall back to PCA on failure.
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42, method=method, init='pca')
        tsne_results = tsne.fit_transform(all_features)
    except Exception as e:
        print(f"t-SNE failed with error: {e}\nFalling back to PCA for visualization.")
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            tsne_results = pca.fit_transform(all_features)
        except Exception as e2:
            raise RuntimeError(f"Both t-SNE and PCA failed: {e2}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if include_source:
        # Define colors and markers for 4 categories
        # Source real, source fake, target real, target fake
        colors = ['#2ECC71', '#E74C3C', '#27AE60', '#C0392B']  # Light green/red for source, darker green/red for target
        markers = ['o', 's', '^', 'D']
        labels_text = ['Source Real', 'Source Fake', 'Target Real', 'Target Fake']
        
        # Create masks for each category
        source_real_mask = (domain_labels == 0) & (all_labels == 0)
        source_fake_mask = (domain_labels == 0) & (all_labels == 1)
        target_real_mask = (domain_labels == 1) & (all_labels == 0)
        target_fake_mask = (domain_labels == 1) & (all_labels == 1)
        
        masks = [source_real_mask, source_fake_mask, target_real_mask, target_fake_mask]
    else:
        # Only 2 categories: target real, target fake
        colors = ["#0FA50F", "#D93830"]  # Green for real, Red for fake
        #markers = ['^', 'D']
        markers = ['o', 'o']  # Circles for both
        labels_text = ['Target Real', 'Target Fake']
        
        # Create masks for target only
        target_real_mask = (all_labels == 0)
        target_fake_mask = (all_labels == 1)
        
        masks = [target_real_mask, target_fake_mask]
    
    # Plot each category
    for i, (mask, color, marker, label) in enumerate(zip(masks, colors, markers, labels_text)):
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
    print(f"t-SNE visualization saved to: {save_path}")
    plt.close()


def run_tsne_visualization(cfg, model, source_loader, target_loader, device, save_dir, max_samples=500, filter_correct_target=False, include_source=False):
    """
    Main function to run t-SNE visualization.
    
    Args:
        cfg: Configuration dictionary
        model: Trained model
        source_loader: DataLoader for source domain
        target_loader: DataLoader for target domain
        device: cuda or cpu
        save_dir: Directory to save visualizations
        max_samples: Maximum samples per class to visualize
        filter_correct_target: If True, only visualize correctly predicted target samples
        include_source: If True, include source domain samples in visualization
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("Starting t-SNE Visualization")
    if filter_correct_target:
        print("(Only showing correctly predicted target samples)")
    if not include_source:
        print("(Only showing target domain)")
    print("="*80)
    
    # Extract features from source domain (no filtering) - only if needed
    if include_source:
        print("\nExtracting features from source domain...")
        source_features, source_labels = extract_features(model, source_loader, device, max_samples, filter_correct=False)
    else:
        # Create empty arrays for source (won't be used but needed for function signature)
        source_features = np.array([])
        source_labels = np.array([])
    
    # Extract features from target domain (with optional filtering)
    print("\nExtracting features from target domain...")
    target_features, target_labels = extract_features(model, target_loader, device, max_samples, filter_correct=filter_correct_target)
    
    # Create visualization
    source_name = cfg['dataset']['source'].upper()
    target_name = cfg['dataset']['target'].upper()
    
    # Build suffix and title based on options
    suffix_parts = []
    if filter_correct_target:
        suffix_parts.append("correct_only")
    if not include_source:
        suffix_parts.append("target_only")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    
    title = f"t-SNE: {target_name}"
    if include_source:
        title = f"t-SNE: {source_name} → {target_name}"
    if filter_correct_target:
        title += " (Correct Predictions Only)"
    
    save_path = os.path.join(save_dir, f"tsne_{source_name}_to_{target_name}{suffix}.png")
    
    print("\nGenerating t-SNE visualization...")
    visualize_tsne(
        source_features, source_labels,
        target_features, target_labels,
        save_path, title, include_source=include_source
    )
    
    print("\n" + "="*80)
    print("t-SNE Visualization Complete!")
    print("="*80 + "\n")
