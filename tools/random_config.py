import random
import numpy as np
from typing import Any, Dict, List

def sample_random_value(param_config: Any) -> Any:
    """
    Sample a random value based on the parameter configuration.
    
    Args:
        param_config: Either a fixed value or a dict with 'type', 'min', 'max' keys
        
    Returns:
        Sampled value or the original value if not a random config
    """
    # If it's not a dict, return as is (fixed value)
    if not isinstance(param_config, dict):
        return param_config
    
    # If it doesn't have a 'type' key, return as is (not a random config)
    if 'type' not in param_config:
        return param_config
    
    param_type = param_config['type']
    
    if param_type == 'randint':
        # Random integer in [min, max] inclusive
        return random.randint(param_config['min'], param_config['max'])
    
    elif param_type == 'uniform':
        # Random float in [min, max] uniform distribution
        return random.uniform(param_config['min'], param_config['max'])
    
    elif param_type == 'loguniform':
        # Random float in [min, max] log-uniform distribution
        log_min = np.log10(param_config['min'])
        log_max = np.log10(param_config['max'])
        return 10 ** random.uniform(log_min, log_max)
    
    elif param_type == 'choice':
        # Random choice from a list of options
        return random.choice(param_config['options'])
    
    elif param_type == 'interaction_layers':
        # Special handler for interaction layers
        # Randomly select N layers from a specified range
        layer_min = param_config.get('layer_min', 0)
        layer_max = param_config.get('layer_max', param_config.get('num_layers', 12) - 1)
        num_selections_config = param_config['num_selections']
        
        if isinstance(num_selections_config, dict):
            num_selections = random.randint(num_selections_config['min'], num_selections_config['max'])
        else:
            num_selections = num_selections_config
        
        # Randomly select unique layer indices from the specified range
        available_layers = range(layer_min, layer_max + 1)
        selected_layers = sorted(random.sample(available_layers, num_selections))
        return selected_layers
    
    elif param_type == 'computed_remainder':
        # This will be computed later in randomize_config
        # Return the config dict as a placeholder
        return param_config
    
    else:
        raise ValueError(f"Unknown random parameter type: {param_type}")


def randomize_config(cfg: Dict) -> Dict:
    """
    Recursively traverse the config dictionary and replace random parameter 
    configurations with sampled values.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Configuration dictionary with randomized values
    """
    randomized_cfg = {}
    
    for key, value in cfg.items():
        if isinstance(value, dict):
            # Check if this is a random parameter config
            if 'type' in value and isinstance(value.get('type'), str):
                # This is a random parameter specification
                randomized_cfg[key] = sample_random_value(value)
            else:
                # Recursively process nested dictionaries
                randomized_cfg[key] = randomize_config(value)
        elif isinstance(value, list):
            # Keep lists as is (unless you want to handle them specially)
            randomized_cfg[key] = value
        else:
            # Keep fixed values as is
            randomized_cfg[key] = value
    
    # Special handling for inference_weights to ensure they sum to 1.0
    if 'inference_weights' in randomized_cfg:
        weights = randomized_cfg['inference_weights']
        if isinstance(weights, dict):
            # Check if artifact is computed_remainder
            if isinstance(weights.get('artifact'), dict) and weights['artifact'].get('type') == 'computed_remainder':
                # Sample binary and attack weights first
                binary_weight = weights.get('binary')
                attack_weight = weights.get('attack')
                
                # Compute artifact weight as remainder
                artifact_weight = 1.0 - binary_weight - attack_weight
                
                # Ensure artifact weight is positive
                if artifact_weight < 0:
                    # Normalize if sum exceeds 1.0
                    total = binary_weight + attack_weight
                    binary_weight = binary_weight / total
                    attack_weight = attack_weight / total
                    artifact_weight = 0.0
                
                weights['artifact'] = artifact_weight
    
    return randomized_cfg


def log_randomized_params(cfg: Dict, log) -> None:
    """
    Log the randomized parameters to help with experiment tracking.
    
    Args:
        cfg: Configuration dictionary with randomized values
        log: Logger object
    """
    log.write('\n' + '='*50)
    log.write('\nRandomized Parameters:')
    log.write('\n' + '='*50)
    
    # Random seed for reproducibility
    if 'random_seed' in cfg:
        log.write(f'\n\nRandom Seed: {cfg["random_seed"]}')
        log.write('\n  (Use this seed with --seed argument to reproduce these exact parameters)')
    
    # Model parameters
    if 'model' in cfg:
        log.write('\nModel:')
        if 'n_ctx' in cfg['model']:
            log.write(f"\n  n_ctx: {cfg['model']['n_ctx']}")
        if 'prompt_depth' in cfg['model']:
            log.write(f"\n  prompt_depth: {cfg['model']['prompt_depth']}")
        if 'interaction_layers' in cfg['model']:
            log.write(f"\n  interaction_layers: {cfg['model']['interaction_layers']}")
    
    # Optimizer parameters
    if 'optimizer' in cfg and 'params' in cfg['optimizer']:
        log.write('\n\nOptimizer:')
        for param_name, param_value in cfg['optimizer']['params'].items():
            log.write(f"\n  {param_name}: {param_value:.6e}" if isinstance(param_value, float) else f"\n  {param_name}: {param_value}")
    
    # Loss weights
    if 'loss_weights' in cfg:
        log.write('\n\nLoss Weights:')
        for weight_name, weight_value in cfg['loss_weights'].items():
            log.write(f"\n  {weight_name}: {weight_value:.4f}" if isinstance(weight_value, float) else f"\n  {weight_name}: {weight_value}")
    
    # Visual prompt training parameters (if present)
    if 'train_visual_prompt' in cfg:
        log.write('\n\nVisual Prompt Training:')
        if 'lr' in cfg['train_visual_prompt']:
            log.write(f"\n  lr: {cfg['train_visual_prompt']['lr']:.6e}")
        if 'weight_decay' in cfg['train_visual_prompt']:
            log.write(f"\n  weight_decay: {cfg['train_visual_prompt']['weight_decay']:.6e}")
    
    # Inference weights
    if 'inference_weights' in cfg:
        log.write('\n\nInference Weights:')
        for weight_name, weight_value in cfg['inference_weights'].items():
            log.write(f"\n  {weight_name}: {weight_value:.4f}" if isinstance(weight_value, float) else f"\n  {weight_name}: {weight_value}")
        # Show sum to verify it equals 1.0
        if all(isinstance(cfg['inference_weights'][k], (int, float)) for k in ['binary', 'attack', 'artifact']):
            total = cfg['inference_weights']['binary'] + cfg['inference_weights']['attack'] + cfg['inference_weights']['artifact']
            log.write(f"\n  total: {total:.4f}")
    
    log.write('\n' + '='*50 + '\n\n')
