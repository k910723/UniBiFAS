import sys, yaml, json, random
import numpy as np
import torch
from datetime import datetime

sys.path.append('.')
from args import get_parser
from .random_config import randomize_config

def GetCfg():
    parser = get_parser()
    args = parser.parse_args()

    args_dict = vars(args)
    # print(json.dumps(args_dict, indent=4))

    cfgPath = args_dict['config']

    with open(cfgPath, 'r') as cfgFile:
        cfg = yaml.safe_load(cfgFile)
    
    # Randomize config parameters if specified
    if args_dict.get('randomize', False):
        # Set seed before randomization for reproducibility
        if args_dict.get('seed') is not None:
            seed = args_dict['seed']
        else:
            # Generate a random seed if not provided
            seed = random.randint(0, 2**32 - 1)
        
        # Set the seed for all random number generators
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Store seed in config
        cfg['random_seed'] = seed
        
        # Now randomize the config with the set seed
        cfg = randomize_config(cfg)
        cfg['randomized'] = True
    else:
        cfg['randomized'] = False
    
    cfg['config'] = args_dict['config']
    cfg['device'] = args_dict['device']
    cfg['ckpt'] = args_dict['ckpt']
    cfg['op_dir'] = args_dict['op_dir']

    if args_dict['source']:
        cfg['dataset']['source'] = args_dict['source']

    if args_dict['target']:
        cfg['dataset']['target'] = args_dict['target']

    if args_dict['size']:
        cfg['dataset']['size'] = args_dict['size']
    
    if args_dict['save_ckpt']:
        cfg['save_ckpt'] = args_dict['save_ckpt']

    if args_dict['visual_prompt_mode']:
        cfg['visual_prompt_mode'] = args_dict['visual_prompt_mode']
        

    cfg['now_time'] = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args_dict['visual_prompt_mode'] == 'train':
        cfg['exp_name'] = f"VP_train_{cfg['dataset']['source']}_{cfg['dataset']['target']}_{cfg['now_time']}"
    elif args_dict['visual_prompt_mode'] == 'finetune':
        cfg['exp_name'] = f"Finetune_{cfg['dataset']['source']}_{cfg['dataset']['target']}_{cfg['now_time']}"
    else:
        cfg['exp_name'] = f"{cfg['model']['mode']}_{cfg['dataset']['source']}_{cfg['dataset']['target']}_{cfg['now_time']}"

    # print(json.dumps(cfg, indent=4))
    return cfg