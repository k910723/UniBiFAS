#!/bin/bash

# =============================================================================
# t-SNE Visualization Setup Complete!
# =============================================================================
#
# Two modes are available:
#
# 1. TRAIN + VISUALIZE (Recommended for first run)
#    - Trains model for 20 epochs
#    - Saves best model (based on AUC)
#    - Generates t-SNE visualization
#
#    Run: bash job_tsne.sh
#
# 2. VISUALIZE ONLY (If you have a trained model)
#    - Loads existing checkpoint
#    - Generates t-SNE visualization
#
#    Run: bash job_visualize_only.sh output/c_i_best.pt
#
# =============================================================================

# Uncomment one of the following:

# Option 1: Train and visualize
bash job_tsne.sh

# Option 2: Visualize only (requires checkpoint path)
# bash job_visualize_only.sh output/c_i_best.pt

# =============================================================================
# Configuration
# =============================================================================
# Edit configs/unibifas.yaml to change:
#   - dataset.source: Training domain (e.g., 'c', 'i', 'cio')
#   - dataset.target: Testing domain (e.g., 'i', 'm')
#   - train.epochs: Number of epochs (default: 20)
#
# =============================================================================
# Output Location
# =============================================================================
# Results saved to: op_dir/<experiment_name>/tsne_visualizations/
# Example: op_dir/unibifas_c_i/tsne_visualizations/tsne_C_to_I.png
#
# =============================================================================
