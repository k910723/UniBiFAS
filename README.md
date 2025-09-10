# UniBiFAS

## Running the Code
To run the project, use the following command:
```bash
python main.py --config configs/unibifas.yaml --op_dir ./output --source OCI --target M
```

## TODO
- [x] **Threshold Calculation**
- [x] **SCM Normalization**
- [x] **Hierarchical Loss**
  - Implemented calculation using class tokens from early layers.

- [ ] **Segmentation Loss**
  - Now the segmentation loss only calculate similarity between spoof text and patches tokens.

- [ ] **Dataset Specification**
  - Current source dataset and target dataset have to be specified in argument instead of in the config file.
