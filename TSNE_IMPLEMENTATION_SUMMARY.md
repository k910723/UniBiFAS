# t-SNE Visualization Implementation Summary

## Files Created/Modified

### New Files Created:

1. **`tools/visualize_tsne.py`** - Core t-SNE visualization functions
   - `extract_features()`: Extracts features from source/target domains (limited samples)
   - `visualize_tsne()`: Creates the 4-category t-SNE plot
   - `run_tsne_visualization()`: Main orchestration function

2. **`train_and_visualize.py`** - Main script for training + visualization
   - Trains model for 20 epochs
   - Saves best model based on AUC
   - Runs t-SNE visualization on both domains

3. **`visualize_only.py`** - Standalone visualization script
   - For use with pre-trained models
   - Takes checkpoint path as argument

4. **`job_tsne.sh`** - Bash script for easy execution
   - Sets up environment
   - Runs training and visualization

5. **`README_VISUALIZATION.md`** - Complete documentation
   - Usage instructions
   - Parameter explanations
   - Troubleshooting guide

### Modified Files:

1. **`utils/utils.py`** - Updated `save_checkpoint()`
   - Now saves `best_AUC` and `best_HTER` in checkpoint
   - Added `model_state_dict` key for compatibility

## Features Implemented

### 1. Training with Best Model Saving
- Trains for 20 epochs (configurable)
- Saves checkpoint with highest AUC
- Checkpoint includes all necessary info for visualization

### 2. Sample Limiting
- Default: 500 samples per class
- Prevents memory issues
- Configurable via `--max_samples` flag
- Four categories:
  - Source Real
  - Source Fake (synthesized)
  - Target Real
  - Target Fake

### 3. t-SNE Visualization
- 2D projection of feature space
- Four distinct categories with different colors/markers
- Legend shows sample counts
- High-resolution output (300 DPI)
- Saved as PNG

### 4. Flexible Usage
Two modes:
- **Train + Visualize**: Complete pipeline
- **Visualize Only**: Use existing checkpoint

## Usage

### Quick Start
```bash
bash job_tsne.sh
```

### Train and Visualize
```bash
python train_and_visualize.py --config configs/unibifas.yaml --save_ckpt
```

### Visualize Existing Model
```bash
python visualize_only.py --checkpoint output/c_i_best.pt --max_samples 500
```

### Custom Source/Target
```bash
python train_and_visualize.py \
    --config configs/unibifas.yaml \
    --source cio \
    --target m \
    --save_ckpt
```

## Output Structure

```
op_dir/UniBiFAS_c_i_TIMESTAMP_tsne/
├── UniBiFAS_c_i_TIMESTAMP_tsne.yaml       # Config used
├── UniBiFAS_c_i_TIMESTAMP_tsne_log.txt    # Training logs
├── UniBiFAS_c_i_TIMESTAMP_tsne_result.csv # Metrics
├── c_i_best.pt                            # Best model checkpoint
└── tsne_visualizations/
    └── tsne_C_to_I.png                    # t-SNE plot
```

## Key Design Decisions

### 1. Sample Limiting Strategy
- Prevents memory overflow
- Maintains class balance
- Default 500 per class = ~2000 total points
- Provides clear visualization

### 2. Feature Extraction
- Uses normalized image features (`img_feat_norm`)
- Consistent with model's similarity computation
- 512-dimensional embeddings

### 3. Checkpoint Format
- Backward compatible with existing code
- Includes both old and new key names
- Stores epoch, metrics, and model state

### 4. Integration with Existing Code
- Uses existing `GetCfg()` for argument parsing
- Uses existing `BuildLoader()` for data loading
- Uses existing `train()` for model training
- Minimal changes to core codebase

## Testing Checklist

- [ ] Run `bash job_tsne.sh` successfully
- [ ] Verify 20 epochs of training complete
- [ ] Check best model saved with correct metrics
- [ ] Confirm t-SNE visualization created
- [ ] Verify 4 categories visible in plot
- [ ] Check sample counts in legend
- [ ] Test `visualize_only.py` with saved checkpoint
- [ ] Try different source/target combinations

## Troubleshooting

### Out of Memory
- Reduce `max_samples` to 300 or 200
- Reduce batch size in config

### Slow Execution
- t-SNE is computationally expensive
- Expected time: 1-2 minutes for 2000 samples
- Reduce samples for faster results

### Import Errors
Make sure you're in the UniBiFAS directory:
```bash
cd /home/kevin/UniBiFAS
python train_and_visualize.py --config configs/unibifas.yaml --save_ckpt
```

## Next Steps

To run your visualization:

1. Make sure you're in the correct environment:
   ```bash
   conda activate fas
   ```

2. Navigate to the project directory:
   ```bash
   cd /home/kevin/UniBiFAS
   ```

3. Run the job script:
   ```bash
   bash job_tsne.sh
   ```

4. Check the output in `op_dir/` for your visualization!
