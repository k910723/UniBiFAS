# UniBiFAS

## Training and Evaluation
To run the project, use the following command:
```bash
python main.py --config configs/unibifas.yaml --op_dir ./output --source OCI --target M
```
### Unseen visual prompt
1. Pre-train and save the model:
```bash
python main.py --config configs/unibifas.yaml --op_dir ./output --source OM --target DHU --save_ckpt
```
2. Train the visual prompt with fixed model:
```bash
python main.py --config configs/unibifas.yaml --op_dir ./output --source OM --target DHU --visual_prompt_mode train
```
3. Fine-tune the model with learned visual prompt:
```bash
python main.py --config configs/finetune.yaml --op_dir ./output --source OM --target DHU --visual_prompt_mode finetune
```

## TODO
- [x] **Threshold Calculation**
- [x] **SCM Normalization**
- [x] **Hierarchical Loss**
  - Implemented calculation using class tokens from early layers.

- [x] **Segmentation Loss**
  - Now the segmentation loss only calculate similarity between spoof text and patches tokens.

- [ ] **Dataset Specification**
  - Current source dataset and target dataset have to be specified in argument instead of in the config file.
