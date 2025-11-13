#!/bin/bash

# Job script for training and t-SNE visualization
# This script trains the model for 20 epochs and then creates t-SNE visualizations
# bash job_tsne.sh

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Configuration
CONFIG_FILE="configs/unibifas.yaml"

echo "=========================================="
echo "Training Model and Creating t-SNE Visualization"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "=========================================="
echo ""

# Run training and visualization
# Note: --save_ckpt flag enables checkpoint saving
python train_and_visualize.py \
    --config $CONFIG_FILE \
    --save_ckpt

echo ""
echo "=========================================="
echo "Job Complete!"
echo "=========================================="
