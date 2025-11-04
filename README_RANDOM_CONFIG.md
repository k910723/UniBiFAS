# Random Configuration System

This document explains how to use the random configuration system for hyperparameter search.

## Overview

The random configuration system allows you to automatically sample hyperparameters from specified ranges during each training run.

## How It Works

### 1. Define Parameter Ranges

```yaml
parameter_name:
    type: <distribution_type>
    min: <minimum_value>
    max: <maximum_value>
```

### 2. Supported Distribution Types

#### `randint` - Random Integer
Samples a random integer uniformly in [min, max] (inclusive).

```yaml
n_ctx:
    type: randint
    min: 2
    max: 8
```

#### `uniform` - Uniform Distribution
Samples a random float uniformly in [min, max].

```yaml
loss_weight:
    type: uniform
    min: 0.5
    max: 2.0
```

#### `loguniform` - Log-Uniform Distribution
Samples a random float from a log-uniform distribution. Best for parameters that span multiple orders of magnitude (like learning rates).

```yaml
lr:
    type: loguniform
    min: 1.0e-7
    max: 1.0e-5
```

#### `choice` - Random Choice
Randomly selects one option from a list.

```yaml
optimizer_type:
    type: choice
    options:
        - Adam
        - SGD
        - AdamW
```

#### `interaction_layers` - Special Handler
Randomly selects a specified number of unique layer indices from a given range.

```yaml
interaction_layers:
    type: interaction_layers
    layer_min: 6              # Minimum layer index (inclusive)
    layer_max: 11             # Maximum layer index (inclusive)
    num_selections: 2         # Number of layers to select
```

For backward compatibility, you can also specify:
```yaml
interaction_layers:
    type: interaction_layers
    num_layers: 12
    num_selections:
        min: 2
        max: 4
```
This will select 2-4 layers randomly from [0, 11].

## Usage

### Using the Random Config

1. **Create or modify a config file** with parameter ranges (see `configs/unibifas_random.yaml` for example)

2. **Run training with the `--randomize` flag:**

```bash
# Run with auto-generated seed
python main.py --config configs/unibifas_random.yaml --randomize

# Run with specific seed for reproducibility
python main.py --config configs/unibifas.yaml --randomize --seed 42
```

### Reproducing Results

When you find a good parameter set, you can reproduce it using the seed logged in the output:

1. Check the log file: `output/<exp_name>/<exp_name>_log.txt`
2. Find the `Random Seed` value
3. Re-run with that seed:

```bash
python main.py --config configs/unibifas_random.yaml --randomize --seed <SEED_VALUE>
```

This will generate the exact same random parameters.

### Running Multiple Experiments

For hyperparameter search, you typically want to run multiple times with different random seeds:

```bash
# Run 10 experiments with different random parameters
for i in {1..10}; do
    python main.py --config configs/unibifas_random.yaml --randomize
done
```

Or modify `base.repeat_num` in the config to run multiple times in a single execution.

## Output

### Saved Configuration

The actual sampled parameters for each run are saved in the output directory:
- `<exp_name>.yaml` - Contains the exact parameters used for this run
- `<exp_name>_log.txt` - Contains logged information including randomized parameters
- `<exp_name>_result.csv` - Contains performance metrics

### Log Output

When using `--randomize`, the log file will include a section showing the sampled parameters:

```
==================================================
Randomized Parameters:
==================================================

Random Seed: 1234567890
  (Use this seed with --seed argument to reproduce these exact parameters)

Model:
  n_ctx: 5
  prompt_depth: 9
  interaction_layers: [2, 5, 9, 11]

Optimizer:
  lr: 3.456789e-06
  weight_decay: 2.345678e-06

Loss Weights:
  binary: 1.2345
  attack: 0.8765
  artifact: 1.5432
  segmentation: 1.1234
==================================================
```