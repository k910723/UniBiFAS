# UniBiFAS Protocol Modes

This document explains the different protocol modes supported by UniBiFAS for training.

## Protocol Modes

### 1. Full Protocol (Default)
The full hierarchical protocol uses all available losses:
- **Binary Classification Loss**: Two-class (real vs. spoof) cross-entropy loss
- **Attack Type Loss**: Multi-class attack type classification loss (real, print, replay)
- **Artifact Type Loss**: Multi-class artifact classification loss
- **Segmentation Loss**: Spatial spoof localization loss

**Config Example:**
```yaml
protocol:
    mode: full

loss_weights:
    binary: 1.0
    attack: 0.5
    artifact: 0.5
    segmentation: 0.5

losses:
    cls_b:
        name: CrossEntropyLoss
        params:
            reduction: mean
    cls_a:
        name: CrossEntropyLoss
        params:
            reduction: mean
    cls_art:
        name: CrossEntropyLoss
        params:
            reduction: mean
    seg:
        name: L1Loss
```

### 2. Binary-Only Protocol
The binary-only protocol simplifies training to use only the two-class cross-entropy loss:
- **Binary Classification Loss**: Two-class (real vs. spoof) cross-entropy loss only
- No attack type, artifact type, or segmentation losses

**Config Example:**
```yaml
protocol:
    mode: binary_only

loss_weights:
    binary: 1.0

losses:
    cls_b:
        name: CrossEntropyLoss
        params:
            reduction: mean
```

## Usage

### Using the Full Protocol
Use the standard config file:
```bash
python main.py --config configs/unibifas_fsw.yaml --source S --target W
```

### Using the Binary-Only Protocol
Use the binary-only config file:
```bash
python main.py --config configs/unibifas_binary_only.yaml --source S --target W
```

## Backward Compatibility

If the `protocol` section is not specified in the config, the system defaults to `full` mode for backward compatibility with existing config files.

## Implementation Details

### Model Forward Pass
- **Full Mode**: Returns all features: `img_feat_norm, cls_tokens, patch_tokens, text_feat_b, text_feat_a, text_feat_art`
- **Binary-Only Mode**: Uses only `img_feat_norm` and `text_feat_b` from the model output

### Loss Calculation
- **Full Mode**: 
  - Total Loss = binary_loss * w_binary + attack_loss * w_attack + artifact_loss * w_artifact + seg_loss * w_seg
- **Binary-Only Mode**: 
  - Total Loss = binary_loss

### Training Logs
The logging format automatically adapts based on the protocol mode:
- **Full Mode**: Shows all loss components (L_bin, L_atk, L_art, L_seg)
- **Binary-Only Mode**: Shows only binary loss (L_bin)

## When to Use Each Protocol

### Use Full Protocol When:
- You have augmented data with full hierarchical labels (attack types and artifacts)
- You want to leverage spatial spoof localization (segmentation)
- You need fine-grained interpretation of the model's decisions

### Use Binary-Only Protocol When:
- You only have binary labels (real vs. spoof)
- You want faster training with fewer loss components
- You're working with datasets that don't have hierarchical annotations
- You want a simpler baseline model
