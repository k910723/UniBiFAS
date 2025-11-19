import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

class VisualizationLoader(Dataset):
    """
    A simple dataloader specifically for visualization.
    It loads images from specific attack-type .npy files (e.g., 
    'Oulu_live_images.npy', 'Oulu_print_images.npy') and maps them
    to the attack-type labels defined in the config.
    """
    def __init__(self, cfg, transform=None):
        self.transform = transform
        self.total_images = []
        self.total_labels = []
        self.data_stats = []  # To track statistics of loaded data

        # Get dataset names (e.g., 'Oulu', 'casia') from protocol
        protocol = cfg['dataset']['target'] # e.g., 'I' or 'O'
        protocol_map = {
            'O': 'Oulu', 'C': 'casia', 'I': 'replay', 'M': 'MSU'
        }
        # Handle both single character and list
        if isinstance(protocol, str):
            dataset_names = [protocol_map.get(p) for p in protocol if p in protocol_map]
        else:
            dataset_names = [protocol_map.get(p) for p in protocol if p in protocol_map]
        
        # Get attack types (e.g., 'Live', 'Print Attack') from config
        # attack_types = cfg['dataset'].get('attack_types', [])
        attack_types = ['live', 'print', 'replay']  # Example attack types
        
        # Map attack names to partial filenames
        # (You may need to adjust this if your filenames differ)
        attack_to_file = {
            'Live': 'live',
            'Print Attack': 'print',
            'Replay Attack': 'replay',
            'Mask Attack': 'mask' 
            # Add other mappings as needed
        }
        
        # Assume domain-generalization path for simplicity
        # (You can expand this using the logic from TestFASDataset if needed)
        data_path = Path(cfg['dataset']['test_path']) / 'domain-generalization'

        for dataset_name in dataset_names:
            for label_idx, attack_name in enumerate(attack_types):
                
                # file_suffix = attack_to_file.get(attack_name)
                file_suffix = attack_name
                if not file_suffix:
                    print(f"Warning: No file mapping for attack '{attack_name}'. Skipping.")
                    continue
                    
                # Construct file path, e.g., Oulu_print_images.npy
                if attack_name.lower() == 'live':
                    file_name = f"{dataset_name}_images_live.npy"
                else:
                    file_name = f"{dataset_name}_{file_suffix}_images.npy"
                
                file_path = data_path / file_name
                
                if file_path.exists():
                    print(f"Loading: {file_path}")
                    images = np.load(file_path)
                    
                    # Collect statistics about the data
                    stats = {
                        'file': file_name,
                        'dtype': images.dtype,
                        'shape': images.shape,
                        'min': images.min(),
                        'max': images.max(),
                        'mean': images.mean(),
                        'std': images.std()
                    }
                    self.data_stats.append(stats)
                    print(f"  Stats: dtype={stats['dtype']}, shape={stats['shape']}, "
                          f"range=[{stats['min']:.4f}, {stats['max']:.4f}], "
                          f"mean={stats['mean']:.4f}, std={stats['std']:.4f}")
                    
                    self.total_images.append(images)
                    
                    # Create labels for these images
                    labels = np.full(len(images), label_idx, dtype=np.int64)
                    self.total_labels.append(labels)
                else:
                    print(f"Warning: File not found, skipping: {file_path}")

        if not self.total_images:
            raise RuntimeError(f"No data loaded from {data_path}. Check config and file paths.")
            
        self.total_images = np.concatenate(self.total_images, axis=0)
        self.total_labels = np.concatenate(self.total_labels, axis=0)
        
        print(f"\n=== Data Loading Summary ===")
        print(f"Total images loaded: {len(self.total_images)}")
        print(f"Combined data - dtype: {self.total_images.dtype}, "
              f"range: [{self.total_images.min():.4f}, {self.total_images.max():.4f}]")
        print(f"===========================\n")

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        img_data = self.total_images[idx]
        attack_label = self.total_labels[idx]

        try:
            # Pre-processing logic from your TestFASDataset
            processed_img = img_data.copy()
            processed_img = np.squeeze(processed_img)
                        
            if processed_img.dtype != np.uint8:
                # The data appears to be in a very small range (e.g., [0, 0.004])
                # This suggests it's already been normalized or is stored in a compressed format
                # We need to check the actual range and scale appropriately
                
                img_max = processed_img.max()
                
                '''if img_max < 0.1:
                    # Data is in a very small range, likely needs rescaling to [0, 255]
                    # Normalize to [0, 1] first, then scale to [0, 255]
                    img_min = processed_img.min()
                    if img_max - img_min > 1e-6:
                        processed_img = (processed_img - img_min) / (img_max - img_min)
                        processed_img = (processed_img * 255).astype(np.uint8)
                    else:
                        processed_img = np.full_like(processed_img, 128, dtype=np.uint8)
                    if idx < 3:
                        print(f"  Action: Normalized from [{img_min:.6f}, {img_max:.6f}] to [0, 255]")'''
                if img_max <= 1.0:
                    # Data is in [0, 1] range
                    if idx < 3:
                        print(f"  Action: Scaling by 255 (data in [0,1] range)")
                    processed_img = (processed_img * 255).astype(np.uint8)
                else:
                    # Data is already in [0, 255] range
                    if idx < 3:
                        print(f"  Action: Clipping to [0,255] (data already in that range)")
                    processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)
            
            if idx < 3:
                print(f"[Sample {idx}] After processing:")
                print(f"  dtype: {processed_img.dtype}, range: [{processed_img.min()}, {processed_img.max()}]")
            
            primary_image = Image.fromarray(processed_img)
            
        except Exception as e:
            print(f"Error converting numpy array to image at index {idx}. Error: {e}")
            return self.__getitem__((idx + 1) % len(self)) # Fallback

        if self.transform:
            primary_image_tensor = self.transform(primary_image)
        else:
            primary_image_tensor = transforms.ToTensor()(primary_image)
                    
        labels_tensor = torch.tensor(attack_label, dtype=torch.long)
        
        return primary_image_tensor, labels_tensor

def BuildVisualizationLoader(cfg):
    """
    Builds the visualization dataloader with the correct transforms.
    """
    # Use the same transforms as your test set
    data_transforms = transforms.Compose([
        transforms.Resize((cfg['transforms']['imgSize'], cfg['transforms']['imgSize'])),
        transforms.ToTensor(),
        transforms.Normalize(cfg['transforms']['mean'], cfg['transforms']['std'])
    ])
    
    dataset = VisualizationLoader(cfg, transform=data_transforms)
    
    loader = DataLoader(
        dataset,
        batch_size=cfg['dataset']['batch_size'],
        shuffle=False, # Shuffle to get a good random sample
        num_workers=cfg['dataset']['num_workers'],
        pin_memory=True
    )
    return loader