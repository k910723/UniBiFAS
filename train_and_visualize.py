"""
Script to train the model for 20 epochs, save the best model based on AUC,
and then run t-SNE visualization on both source and target domains.
"""

import yaml
import os
import torch
import random
import warnings
import numpy as np

from utils import Logger
from tools import GetCfg, train
from tools.visualize_tsne import run_tsne_visualization
from data.build import BuildLoader
from modeling.unibifas import UniBiFAS_Model

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

def load_best_model(cfg, device):
    """Load the best model checkpoint saved during training."""
    source = cfg['dataset']['source']
    target = cfg['dataset']['target']
    checkpoint_path = f"{cfg['train']['save_path']}/{source}_{target}_best.pt"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Best model checkpoint not found at {checkpoint_path}")
    
    print(f"\nLoading best model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = UniBiFAS_Model(cfg).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Best AUC: {checkpoint['best_AUC']:.4f}")
    print(f"  Best HTER: {checkpoint['best_HTER']:.4f}")
    
    return model


if __name__ == '__main__':
    # Get configuration
    cfg = GetCfg()
    
    # Ensure we train for 20 epochs
    cfg['train']['epochs'] = 20
    cfg['base']['repeat_num'] = 1  # Single run for visualization
    
    # Enable checkpoint saving (already set via --save_ckpt flag)
    # cfg['save_ckpt'] is set by GetCfg() from command line args
    
    # Update experiment name to indicate this is for visualization
    cfg['exp_name'] = cfg['exp_name'] + '_tsne'
    
    # Create experiment directory
    expSavePath = os.path.join(cfg['op_dir'], cfg['exp_name'])
    os.makedirs(expSavePath, exist_ok=True)
    
    # Save config
    cfgName = f"{cfg['exp_name']}.yaml"
    cfgSavePath = os.path.join(expSavePath, cfgName)
    with open(cfgSavePath, 'w', encoding='utf-8') as file:
        yaml.dump(cfg, file, allow_unicode=True, default_flow_style=False)
    
    # Setup logger
    logName = f"{cfg['exp_name']}_log.txt"
    logSavePath = os.path.join(expSavePath, logName)
    log = Logger()
    log.open(logSavePath)
    
    # Setup result file
    resultName = f"{cfg['exp_name']}_result.csv"
    resultSavePath = os.path.join(expSavePath, resultName)
    with open(resultSavePath, 'a') as f:
        f.write(f"{'Phase': ^15}{'HTER': ^10}{'AUC': ^10}{'TPR@FPR=1%': ^15}\n")
    
    print("\n" + "="*80)
    print("PHASE 1: TRAINING MODEL FOR 20 EPOCHS")
    print("="*80)
    
    # Train the model
    hter, auc, tpr_fpr = train(cfg, log)
    
    # Log training results
    with open(resultSavePath, 'a') as f:
        f.write(f"{'Training': ^15}{hter: ^10.4f}{auc: ^10.4f}{tpr_fpr: ^15.4f}\n")
    
    print("\n" + "="*80)
    print("PHASE 2: t-SNE VISUALIZATION")
    print("="*80)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the best model
    model = load_best_model(cfg, device)
    model.eval()
    
    # Build dataloaders for source and target domains
    print("\nBuilding source domain dataloader...")
    source_loader = BuildLoader(cfg, isTrain=True, log=log)
    
    print("\nBuilding target domain dataloader...")
    target_loader = BuildLoader(cfg, isTrain=False, log=log)
    
    # Create visualization directory
    vis_dir = os.path.join(expSavePath, 'tsne_visualizations')
    
    # ==================================================================================
    # VISUALIZATION CONTROL FLAGS - MODIFY THESE TO CHANGE VISUALIZATION BEHAVIOR
    # ==================================================================================
    INCLUDE_SOURCE = False  # Set to True to include source domain in visualization
    FILTER_CORRECT_ONLY = False  # Set to True to only show correctly predicted samples
    # ==================================================================================
    
    # Run t-SNE visualization with limited samples (500 per class)
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
        max_samples=500,  # Limit to 500 samples per class to avoid memory issues
        filter_correct_target=FILTER_CORRECT_ONLY,
        include_source=INCLUDE_SOURCE
    )
    
    print("\n" + "="*80)
    print("TRAINING AND VISUALIZATION COMPLETE!")
    print(f"Results saved to: {expSavePath}")
    print(f"Visualizations saved to: {vis_dir}")
    print("="*80 + "\n")
