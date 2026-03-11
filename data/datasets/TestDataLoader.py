import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path

class TestFASDataset(Dataset):
    """
    Custom PyTorch Dataset for Face Anti-Spoofing, designed for testing on
    the Oulu-NPU, CASIA-FASD, Idiap Replay-Attack, and MSU-MFSD datasets.

    This dataloader is structured similarly to HierarchicalFasDataset but uses
    pre-processed .npy files as specified in the original dataloader.py.

    It loads data from paths like:
    - root_dir/
      - domain-generalization/
        - Oulu_images_live.npy
        - Oulu_images_spoof.npy
        - casia_images_live.npy
        - etc.
    
    Returns a tuple for compatibility with the hierarchical training setup:
    (image_tensor, dummy_tensor, labels_tensor)
    where dummy_tensor is a zero tensor and labels_tensor contains the binary label.
    """

    def __init__(self, root_dir, protocol, transform=None, live_only=False):
        """
        Args:
            root_dir (string): Root directory containing the 'domain-generalization' folder.
            protocol (list of strings): List of datasets to use, e.g., ['O', 'Ca', 'I', 'M'].
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.live_only = live_only
        self.protocol_map = {
            'O': 'Oulu',
            'C': 'casia',  # Lowercase C for Casia (old dataset)
            'I': 'replay',
            'M': 'MSU',
            'D': '3DMAD',
            'H': 'HKBUv1+',
            'U': 'casia_3d',
            'paper_glasses': 'paper_glasses_RGB',
            'mask_silicone': 'mask_silicone_RGB',
            'funny_eyes': 'funny_eyes_RGB',
            # New F/S/W datasets (F=CeFA, avoiding conflict with Casia)
            'F': 'CeFA',
            'CeFA': 'CeFA',  # Also support full name
            'S': 'SURF',
            'SURF': 'SURF',
            'W': 'WMCA',
            'WMCA': 'WMCA',
        }

        all_live_images = []
        all_spoof_images = []
        # Track which samples are from image files vs numpy arrays
        image_file_samples = []  # List of (image_path, label)

        # data_path = self.root_dir / 'domain-generalization'

        for p in protocol:
            if p not in self.protocol_map:
                print(f"Warning: Protocol '{p}' is not recognized. Skipping.")
                continue
            
            base_name = self.protocol_map[p]
            
            # Check if this is a F/S/W dataset (uses image directories)
            if p in ['F', 'CeFA', 'S', 'SURF', 'W', 'WMCA']:
                data_path = self.root_dir / 'domain-generalization-multi' / base_name
                live_dir = data_path / 'real' / 'profile'
                spoof_dir = data_path / 'spoof' / 'profile'
                
                # Load live images from directory
                if live_dir.exists():
                    live_count = 0
                    for ext in ['.jpg', '.png', '.jpeg']:
                        for img_path in live_dir.rglob(f'*{ext}'):
                            image_file_samples.append((img_path, 0))  # 0 = live/real
                            live_count += 1
                    print(f"Loaded {live_count} live images from {live_dir}")
                else:
                    print(f"Warning: Directory not found at {live_dir}")
                
                # Load spoof images from directory
                if spoof_dir.exists():
                    spoof_count = 0
                    for ext in ['.jpg', '.png', '.jpeg']:
                        for img_path in spoof_dir.rglob(f'*{ext}'):
                            image_file_samples.append((img_path, 1))  # 1 = spoof/fake
                            spoof_count += 1
                    print(f"Loaded {spoof_count} spoof images from {spoof_dir}")
                else:
                    print(f"Warning: Directory not found at {spoof_dir}")
                
                # Skip numpy file loading for F/S/W datasets
                continue
                
            elif p == 'D' or p == 'H' or p == 'U':
                data_path = self.root_dir / "3Dmask"
                live_path = data_path / f"{base_name}_images_live.npy"
                spoof_path = data_path / f"{base_name}_images_spoof.npy"
            elif p in ['paper_glasses', 'mask_silicone', 'funny_eyes']:
                data_path = self.root_dir / "padisi" / "partial" / "test"
                live_path = data_path / "live_RGB.npy"
                spoof_path = data_path / f"{base_name}.npy"
            else:
                data_path = self.root_dir / "domain-generalization"
                live_path = data_path / f"{base_name}_images_live.npy"
                spoof_path = data_path / f"{base_name}_images_spoof.npy"

            if live_path.exists():
                live_data = np.load(live_path)
                all_live_images.append(live_data)
            else:
                print(f"Warning: File not found at {live_path}")

            if spoof_path.exists():
                spoof_data = np.load(spoof_path)
                all_spoof_images.append(spoof_data)
            else:
                print(f"Warning: File not found at {spoof_path}")

        if not all_live_images and not all_spoof_images and not image_file_samples:
            raise RuntimeError("No data loaded. Check root_dir and protocol.")

        # Store image file samples separately
        self.image_file_samples = image_file_samples
        self.use_image_files = len(image_file_samples) > 0

        # Concatenate all loaded images from the specified protocols (.npy files)
        live_images = np.concatenate(all_live_images, axis=0) if all_live_images else np.array([])
        spoof_images = np.concatenate(all_spoof_images, axis=0) if all_spoof_images else np.array([])
        
        # Create labels: 0 for real/live, 1 for fake/spoof (to match HierarchicalDataLoader)
        # This is the reverse of the original FAS_Dataset for consistency.
        live_labels = np.zeros(len(live_images), dtype=np.int64)
        spoof_labels = np.ones(len(spoof_images), dtype=np.int64)
        
        if self.live_only:
            self.total_images = live_images
            self.total_labels = live_labels
        elif len(live_images) > 0 and len(spoof_images) > 0:
            # Combine live and spoof data
            self.total_images = np.concatenate((live_images, spoof_images), axis=0)
            self.total_labels = np.concatenate((live_labels, spoof_labels), axis=0)
        elif len(live_images) > 0:
            self.total_images = live_images
            self.total_labels = live_labels
        elif len(spoof_images) > 0:
            self.total_images = spoof_images
            self.total_labels = spoof_labels
        else:
            # All data is from image files
            self.total_images = np.array([])
            self.total_labels = np.array([])


    def __len__(self):
        if self.use_image_files:
            return len(self.image_file_samples) + len(self.total_images)
        return len(self.total_images)

    def __getitem__(self, idx):
        """
        Fetches a sample from the dataset.
        
        Returns:
            tuple: (primary_image_tensor, dummy_scm_tensor, labels_tensor)
                   to maintain compatibility with the hierarchical dataloader.
        """
        # Check if this index refers to an image file or numpy array
        if self.use_image_files and idx < len(self.image_file_samples):
            # Load from image file
            img_path, binary_label = self.image_file_samples[idx]
            try:
                primary_image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}. Error: {e}")
                # Fallback to the next item
                return self.__getitem__((idx + 1) % len(self))
        else:
            # Load from numpy array
            if self.use_image_files:
                # Adjust index for numpy array portion
                idx = idx - len(self.image_file_samples)
            
            img_data = self.total_images[idx]
            binary_label = self.total_labels[idx]

            # Convert numpy array to PIL Image with proper preprocessing
            try:
                # Handle problematic array shapes and data types
                processed_img = img_data.copy()
                
                # Remove singleton dimensions (e.g., (1,1,3) -> (1,3) -> (3,))
                processed_img = np.squeeze(processed_img)
                # print(processed_img)
                            
                # Convert data type to uint8 if it's float
                if processed_img.dtype == np.float32 or processed_img.dtype == np.float64:
                    if processed_img.max() < 0.1:
                        processed_img *= 255.0  # Scale to [0, 1] if in [0, 0.0039]. For visualization only!!! Impact performance
                    # Assume values are in range [0,1] and scale to [0,255]
                    if processed_img.max() <= 1.0:
                        processed_img = (processed_img * 255).astype(np.uint8)
                    else:
                        # Values might already be in [0,255] range but stored as float
                        # processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)
                        processed_img = (processed_img / 255).astype(np.uint8) # Performance improvement for DHU
                elif processed_img.dtype != np.uint8:
                    # Convert other integer types to uint8
                    processed_img = processed_img.astype(np.uint8)
                
                primary_image = Image.fromarray(processed_img)
                
            except Exception as e:
                print(f"Error converting numpy array to image at index {idx}. Error: {e}")
                print(f"Original shape: {img_data.shape}, dtype: {img_data.dtype}")
                print(f"Value range: [{img_data.min():.4f}, {img_data.max():.4f}]")
                # Fallback to the next item in the dataset
                return self.__getitem__((idx + 1) % len(self))

        # Apply transformations to the image
        if self.transform:
            primary_image_tensor = self.transform(primary_image)
        else:
            # Apply a default transformation if none is provided - resize to 224x224
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            primary_image_tensor = default_transform(primary_image)
                    
        # Create a labels tensor with a hierarchical structure.
        # We only have binary info, so use -1 as a placeholder for attack and artifact tiers.
        labels = (binary_label, -1, -1)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return primary_image_tensor, labels_tensor
