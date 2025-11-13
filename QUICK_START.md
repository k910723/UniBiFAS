# Quick Start Guide: t-SNE Visualization

## What was implemented

1. **Training + Visualization Pipeline** (`train_and_visualize.py`)
   - Trains model for 20 epochs
   - Saves best model based on AUC
   - Automatically generates t-SNE visualizations

2. **Visualization-Only Tool** (`visualize_only.py`)
   - Uses pre-trained model checkpoint
   - Creates t-SNE plots without retraining

3. **Robust t-SNE Implementation** (`tools/visualize_tsne.py`)
   - Auto-adjusts perplexity based on sample count
   - Validates features (removes NaN/Inf)
   - Falls back to PCA if t-SNE fails
   - Limits samples per class to avoid memory issues

## How to Use

### Option 1: Train and Visualize (Full Pipeline)

```bash
bash job_tsne.sh
```

This will:
- Train the model for 20 epochs
- Save the best checkpoint (based on AUC)
- Generate t-SNE visualization automatically

### Option 2: Visualize Pre-trained Model

If you already have a trained model:

```bash
bash job_visualize_only.sh output/c_i_best.pt
```

Replace `output/c_i_best.pt` with your checkpoint path.

## Configuration

Edit `configs/unibifas.yaml`:

```yaml
dataset:
    source: 'c'    # Training domain: c, i, o, m, or combinations
    target: 'i'    # Testing domain: c, i, o, m, or combinations
    
train:
    epochs: 20     # Number of training epochs
```

## Output

Results saved to `op_dir/<experiment_name>/`:

```
op_dir/unibifas_c_i/
├── unibifas_c_i.yaml              # Config used
├── unibifas_c_i_log.txt           # Training logs
├── unibifas_c_i_result.csv        # Metrics
├── c_i_best.pt                    # Best checkpoint
└── tsne_visualizations/
    └── tsne_C_to_I.png            # t-SNE plot
```

## Visualization Details

The t-SNE plot shows 4 categories:
- **Source Real** (blue circles)
- **Source Fake** (purple squares)
- **Target Real** (orange triangles)
- **Target Fake** (red diamonds)

Sample limits (default):
- 500 samples per class
- Total: ~2000 points

## Troubleshooting

### "ValueError: cannot reshape array"
✅ **Fixed!** The code now:
- Auto-adjusts perplexity based on sample count
- Validates features and removes NaN/Inf values
- Falls back to PCA if t-SNE fails

### "Out of Memory"
Reduce samples per class by editing `train_and_visualize.py`:
```python
max_samples=300  # Change from 500 to 300
```

### "Checkpoint not found"
Make sure you're using the correct checkpoint path. After training, it's saved as:
```
output/<source>_<target>_best.pt
```

For example: `output/c_i_best.pt`

## Example Run

```bash
# 1. Train and visualize C→I
cd /home/kevin/UniBiFAS
bash job_tsne.sh

# 2. Or visualize existing model
bash job_visualize_only.sh output/c_i_best.pt
```

Expected output:
```
Running t-SNE on ~2000 samples...
  Source: 1000 samples (500 real, 500 fake)
  Target: 1000 samples (500 real, 500 fake)
t-SNE visualization saved to: op_dir/.../tsne_C_to_I.png
```

## Advanced: Testing Different Domains

```bash
# Edit config to test O→M
vim configs/unibifas.yaml
# Change: source: 'o' and target: 'm'

# Run
bash job_tsne.sh
```

## Files Modified/Created

- ✅ `train_and_visualize.py` - Main training + visualization script
- ✅ `visualize_only.py` - Visualization-only script  
- ✅ `tools/visualize_tsne.py` - Robust t-SNE implementation
- ✅ `job_tsne.sh` - Job script for training + visualization
- ✅ `job_visualize_only.sh` - Job script for visualization only
- ✅ `utils/utils.py` - Updated checkpoint saving with AUC
- ✅ `README_VISUALIZATION.md` - Full documentation
- ✅ `QUICK_START.md` - This file
