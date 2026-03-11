import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from .transforms import FasTransforms
from .datasets import HierarchicalFasDataset, TestFASDataset
from .batchsampler import BatchSchedulerSampler, SchedulerSampler

def BuildLoader(cfg, isTrain=True, isTrainVisualPrompt=False, isFineTune=False, log=None):
    """
    Builds a DataLoader for either training or testing.

    For training (isTrain=True):
    - Uses HierarchicalFasDataset for datasets with complex directory structures (real, fake, SCM).
    - Reads source domains from config, instantiates a dataset for each, and concatenates them.

    For testing (isTrain=False):
    - Uses TestFASDataset for datasets loaded from .npy files (Oulu, Casia, etc.).
    - Reads target domains from config and creates a single dataset instance.
    """
    
    # Initialize transforms for the images.
    # Create simple transforms that just resize and normalize
    simple_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg['transforms']['mean'], std=cfg['transforms']['std'])
    ])

    if isTrain or isTrainVisualPrompt:
        # --- TRAINING LOADER ---
        # Uses HierarchicalFasDataset for source domains
        print('\n--- Loading Source Domains for Training ---')
        abbr2datasetType = {
            'o': 'Oulu', 'c': 'Casia', 'm': 'Msu', 'i': 'Idiap', 'b': 'Celeb', 'p': 'partial',
            'F': 'C', 'S': 'S', 'W': 'W'  # F/S/W map to actual directory names C/S/W
        }
        #scm_transforms = FasTransforms(cfg, isTrain) # SCM transforms only needed for training
        
        domain_keys = cfg['dataset']['source']  # Keep original case for C/S/W

        # Support both single-character codes ('OCI') and space-separated codes ('O C I')
        if ' ' in domain_keys:
            # Space-separated: 'O C I' or 'F S W' → ['O', 'C', 'I'] or ['F', 'S', 'W']
            key_list = domain_keys.split()
        else:
            # Single string: 'OCI' or 'FSW' → ['O', 'C', 'I'] or ['F', 'S', 'W']
            key_list = list(domain_keys)
        
        datasetTypes = [abbr2datasetType[key] for key in key_list]

        all_datasets = []
        print(f"Training dataset root path: {cfg['dataset']['root_path']}")
        
        # Check if we should load actual spoof data
        actual_spoof_base = cfg['dataset'].get('actual_spoof_dir', None)
        
        # Mapping from dataset names to .npy file prefixes or directory names
        dataset_to_npy_prefix = {
            'Oulu': 'Oulu',
            'Casia': 'casia',
            'Msu': 'MSU',
            'Idiap': 'replay',  # Idiap is stored as 'replay'
            'Celeb': None,
            'partial': None,
            'C': 'CeFA',     # C/S/W directories map to CeFA/SURF/WMCA for actual spoof data
            'S': 'SURF',
            'W': 'WMCA'
        }
        
        for datasetType in datasetTypes:
            dataset_path = os.path.join(cfg['dataset']['root_path'], datasetType)

            if not os.path.isdir(dataset_path):
                print(f"Warning: Directory not found for '{datasetType}' at '{dataset_path}'. Skipping.")
                continue

            # Determine actual spoof path for this specific dataset
            actual_spoof_path = None
            if actual_spoof_base and cfg['dataset'].get('load_actual_spoof', False):
                npy_prefix = dataset_to_npy_prefix.get(datasetType)
                if npy_prefix:
                    # Check if this is a F/S/W dataset (uses image directories)
                    if datasetType in ['C', 'S', 'W']:
                        # Check for image directory structure
                        spoof_dir = os.path.join(actual_spoof_base, npy_prefix, 'spoof', 'profile')
                        real_dir = os.path.join(actual_spoof_base, npy_prefix, 'real', 'profile')
                        if os.path.exists(spoof_dir) and os.path.exists(real_dir):
                            actual_spoof_path = actual_spoof_base
                            print(f"  Loading actual spoof images from: {spoof_dir}")
                        else:
                            print(f"  Warning: Actual spoof image directories not found for '{datasetType}' at {actual_spoof_base}/{npy_prefix}")
                    else:
                        # Check if the .npy files exist (for O/Ca/I/M datasets)
                        spoof_file = os.path.join(actual_spoof_base, f"{npy_prefix}_images_spoof.npy")
                        live_file = os.path.join(actual_spoof_base, f"{npy_prefix}_images_live.npy")
                        if os.path.exists(spoof_file) and os.path.exists(live_file):
                            actual_spoof_path = actual_spoof_base
                            print(f"  Loading actual spoof data from: {spoof_file} and {live_file}")
                        else:
                            print(f"  Warning: Actual spoof .npy files not found for '{datasetType}' (prefix: {npy_prefix})")

            dataset = HierarchicalFasDataset(
                root_dir=dataset_path,
                transform=simple_transforms,
                scm_transform=simple_transforms,
                # live_only=isTrainVisualPrompt,
                finetune=isFineTune,
                visual_prompt_path=cfg['train_visual_prompt']['save_path'] if isFineTune else None,
                actual_spoof_dir=actual_spoof_path,  # Path to .npy files
                actual_spoof_dataset_name=dataset_to_npy_prefix.get(datasetType),  # Pass the prefix for .npy files
            )
            all_datasets.append(dataset)
            print(f"Loaded '{datasetType}' with {len(dataset)} total samples.")

        if not all_datasets:
            raise ValueError("No training datasets were loaded. Check config and paths.")

        concatDataset = ConcatDataset(all_datasets)

        

    else:
        # --- TESTING LOADER ---
        # Uses TestFASDataset for target domains (O, Ca, I, M from .npy files)
        # TestFASDataset already handles muitiple datasets internally
        print('\n--- Loading Target Domains for Testing ---')
        domain_keys = cfg['dataset']['target']  # Keep original case for protocols like 'Ca'

        # Create the protocol list from the domain keys string
        if ' ' in domain_keys:
            protocol = domain_keys.split()
        elif '_' in list(domain_keys):
            protocol = [domain_keys]
        else:
            protocol = list(domain_keys)

        print(f"Loading test protocols: {protocol}")
        
        # Instantiate the test dataset
        concatDataset = TestFASDataset(
            root_dir=cfg['dataset']['test_path'],
            protocol=protocol,
            transform=simple_transforms
        )
        
        if len(concatDataset) == 0:
            raise ValueError("Test dataset is empty. Check config and paths.")

    print(f"\nTotal combined samples: {len(concatDataset)}")

    dataloader = DataLoader(
        concatDataset,
        batch_size=cfg['dataset']['batch_size'],
        shuffle=isTrain or isTrainVisualPrompt,
        num_workers=cfg['dataset']['num_workers']
    )
    
    print(f"DataLoader created with batch size: {cfg['dataset']['batch_size']}")
    #print("Each item in a batch will be a tuple: (images, spoof_cue_maps, labels)")

    return dataloader