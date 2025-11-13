"""
Script to run t-SNE visualization on an already trained model.
Use this if you already have a trained model and just want to visualize.

Usage:
    python visualize_only.py --config configs/unibifas.yaml --ckpt output/c_i_best.pt
"""

import yaml
import os
import torch
import warnings

from utils import Logger
from tools import GetCfg
from tools.visualize_tsne import run_tsne_visualization
from data.build import BuildLoader
from modeling.unibifas import UniBiFAS_Model

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
        raise KeyError("No model weights found in checkpoint. Expected 'model_state_dict' or 'state_dict'.")
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    if 'best_AUC' in checkpoint:
        print(f"  Best AUC: {checkpoint['best_AUC']:.4f}")
    if 'best_HTER' in checkpoint:
        print(f"  Best HTER: {checkpoint['best_HTER']:.4f}")
    elif 'best_model_EER' in checkpoint:
        print(f"  Best HTER: {checkpoint['best_model_EER']:.4f}")
    
    return model


if __name__ == '__main__':
    # Get configuration using the existing argument parser
    cfg = GetCfg()
    
    # Check if checkpoint path is provided
    if not cfg.get('ckpt'):
        raise ValueError("Please provide a checkpoint path using --ckpt argument")
    
    checkpoint_path = cfg['ckpt']
    max_samples = cfg.get('max_samples', 700)  # Default 500 samples per class
    
    # ==================================================================================
    # VISUALIZATION CONTROL FLAGS - MODIFY THESE TO CHANGE VISUALIZATION BEHAVIOR
    # ==================================================================================
    INCLUDE_SOURCE = False  # Set to True to include source domain in visualization
    FILTER_CORRECT_ONLY = False  # Set to True to only show correctly predicted samples
    # ==================================================================================
    
    # Create experiment directory
    expSavePath = os.path.join(cfg['op_dir'], cfg['exp_name'])
    os.makedirs(expSavePath, exist_ok=True)
    
    # Setup logger
    logName = f"{cfg['exp_name']}_visualization_log.txt"
    logSavePath = os.path.join(expSavePath, logName)
    log = Logger()
    log.open(logSavePath)
    
    print("\n" + "="*80)
    print("t-SNE VISUALIZATION OF TRAINED MODEL")
    print("="*80)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    model = load_model_checkpoint(checkpoint_path, cfg, device)
    model.eval()
    
    # Build dataloaders for source and target domains
    print("\nBuilding source domain dataloader...")
    source_loader = BuildLoader(cfg, isTrain=True, log=log)
    
    print("\nBuilding target domain dataloader...")
    target_loader = BuildLoader(cfg, isTrain=False, log=log)
    
    # Create visualization directory
    vis_dir = os.path.join(expSavePath, 'tsne_visualizations')
    
    # Run t-SNE visualization
    print(f"\n--- Creating t-SNE Visualization ---")
    print(f"  Include Source: {INCLUDE_SOURCE}")
    print(f"  Filter Correct Only: {FILTER_CORRECT_ONLY}")
    
    run_tsne_visualization(
        cfg=cfg,
        model=model,
        source_loader=source_loader,
        target_loader=target_loader,
        device=device,
        save_dir=vis_dir,
        max_samples=max_samples,
        filter_correct_target=FILTER_CORRECT_ONLY,
        include_source=INCLUDE_SOURCE
    )
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print(f"Visualizations saved to: {vis_dir}")
    print("="*80 + "\n")
