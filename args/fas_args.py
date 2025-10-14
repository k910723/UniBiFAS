import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='pytorch training and testing ')

    # base args
    parser.add_argument('--config', required = True, help = 'Path to YAML configuration file')

    parser.add_argument('--device', default = 'cuda:0', help = 'Device to use (cpu/cuda:0/cuda:1)')
    parser.add_argument('--ckpt', type = str, default = None, help = 'Path to checkpoint file')
    parser.add_argument('--op_dir', type = str, default = './op_dir', help = 'Directory to save checkpoints')

    parser.add_argument('--source', type = str, help = 'source dataset')
    parser.add_argument('--target', type = str, help = 'target dataset')
    parser.add_argument('--size', type = int, help = 'sample size of each datasetType and labelType in each batch')

    parser.add_argument('--save_ckpt', action='store_true', help = 'Whether to save checkpoints during training')
    parser.add_argument('--visual_prompt_mode', type = str, default=None, help = 'Mode for visual prompt tuning: stage1/stage2')

    return parser