# t-SNE Visualization for UniBiFAS

This guide explains how to train the UniBiFAS model and create t-SNE visualizations showing the feature distributions of source and target domains.

## Overview

The t-SNE visualization shows:
- **Source Real**: Real faces from the source training domain
- **Source Fake**: Synthesized/fake faces from the source training domain  
- **Target Real**: Real faces from the target testing domain
- **Target Fake**: Fake faces from the target testing domain

This helps visualize:
1. How well the model separates real vs. fake faces
2. Domain shift between source and target domains
3. Feature space clustering and distribution

## Quick Start

### Option 1: Train and Visualize (Recommended)

Train the model for 20 epochs and automatically generate t-SNE visualizations:

```bash
bash job_tsne.sh
```

Or run directly:

```bash
python train_and_visualize.py --config configs/unibifas.yaml --save_ckpt
```

The experiment name will be automatically generated based on the source and target domains.

### Option 2: Visualize Only (If You Already Have a Trained Model)

If you already have a trained model checkpoint:

```bash
bash job_visualize_only.sh output/c_i_best.pt
```

Or run directly:

```bash
python visualize_only.py \
    --config configs/unibifas.yaml \
    --ckpt output/c_i_best.pt
```

## Configuration

Edit `configs/unibifas.yaml` to change:

- **Source domain**: `dataset.source` (e.g., 'c', 'i', 'o', 'm', or combinations like 'cio')
- **Target domain**: `dataset.target` (e.g., 'c', 'i', 'o', 'm')
- **Training epochs**: `train.epochs` (default: 20)
- **Batch size**: `dataset.batch_size` (default: 32)

## Sample Limiting

To prevent memory issues and improve visualization clarity, the script limits samples:
- **Default**: 500 samples per class (Source Real, Source Fake, Target Real, Target Fake)
- **Adjustable**: Use `--max_samples` flag in `visualize_only.py`

This means the visualization will show up to:
- 500 Source Real samples
- 500 Source Fake samples
- 500 Target Real samples
- 500 Target Fake samples

Total: ~2000 points in the t-SNE plot

## Output

Results are saved to `op_dir/<exp_name>/`:

```
op_dir/tsne_experiment/
├── tsne_experiment.yaml              # Configuration used
├── tsne_experiment_log.txt            # Training logs
├── tsne_experiment_result.csv         # Performance metrics
├── c_i_best.pt                        # Best model checkpoint (based on AUC)
└── tsne_visualizations/
    └── tsne_C_to_I.png               # t-SNE visualization
```

## Understanding the Visualization

The t-SNE plot shows:

1. **Colors**: Different colors for each of the 4 categories
   - Blue (○): Source Real
   - Purple (□): Source Fake  
   - Orange (△): Target Real
   - Red (◇): Target Fake

2. **Good Model Behavior**:
   - Real and fake samples should form distinct clusters
   - Source and target domains may overlap (indicating good domain adaptation)
   - Fake samples should be clearly separated from real samples

3. **Issues to Look For**:
   - Mixed clusters (real and fake intermingled) = poor discrimination
   - Large gap between source and target = domain shift problem
   - Scattered points = feature space not well-organized

## Advanced Usage

### Custom Sample Limits

To visualize with more or fewer samples, edit `tools/visualize_tsne.py` and change the `max_samples` parameter in the `run_tsne_visualization()` call. The default is 500 samples per class.

### Multiple Domain Combinations

To visualize different source → target combinations, modify the config:

```yaml
dataset:
    source: 'cio'  # Train on Casia + Idiap + Oulu
    target: 'm'    # Test on MSU
```

Then run:

```bash
python train_and_visualize.py --config configs/unibifas.yaml --save_ckpt
```

Or specify source/target via command line:

```bash
python train_and_visualize.py --config configs/unibifas.yaml --source cio --target m --save_ckpt
```

## Troubleshooting

### Out of Memory

If you encounter OOM errors:
1. Reduce `max_samples` (try 300 or 200)
2. Reduce `batch_size` in config
3. Use a GPU with more memory

### Slow t-SNE

t-SNE can be slow with many samples. The default settings are optimized for speed:
- `perplexity=30` (lower = faster)
- `n_iter=1000` (fewer iterations = faster)

### Poor Visualization Quality

If the visualization looks cluttered:
1. Increase `perplexity` parameter in `visualize_tsne.py` (line ~80)
2. Increase `n_iter` for better convergence
3. Use fewer samples per class

## Example Results

After running `bash job_tsne.sh`, you should see:

```
================================================================================
PHASE 1: TRAINING MODEL FOR 20 EPOCHS
================================================================================
[Training progress...]

================================================================================
PHASE 2: t-SNE VISUALIZATION  
================================================================================
Extracting features from source domain...
Extracted 1000 features: 500 real, 500 fake

Extracting features from target domain...
Extracted 1000 features: 500 real, 500 fake

Generating t-SNE visualization...
Running t-SNE on 2000 samples...
  Source: 1000 samples (500 real, 500 fake)
  Target: 1000 samples (500 real, 500 fake)
t-SNE visualization saved to: op_dir/tsne_experiment/tsne_visualizations/tsne_C_to_I.png

================================================================================
TRAINING AND VISUALIZATION COMPLETE!
================================================================================
```

## Dependencies

The visualization requires:
- PyTorch
- NumPy
- scikit-learn (for TSNE)
- matplotlib (for plotting)

These should already be installed if you can run the main training script.
