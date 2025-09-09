import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

class HierarchicalFasDataset(Dataset):
    """
    Custom PyTorch Dataset for Face Anti-Spoofing with a hierarchical label structure.
    Parses a directory structure like:

    - root_dir/
      - OULU/
        - real/
            - image0.png
        - fake/
            - print/
                - halftone/
                    - image0.png
                - distortion/
                    - image0.png
            - replay/
                - moire/
                    - image0.png
                - reflection/
                    - image0.png
        - SCM/ (mirrors the structure of 'fake')
    
    For a fake image: (fake_image_tensor, scm_image_tensor, labels_tensor)
    For a real image: (real_image_tensor, zero_tensor, labels_tensor)
    """

    def __init__(self, root_dir, transform=None, scm_transform=None):
        """
        Args:
            root_dir (string): Directory for a specific dataset (e.g., '.../data/OULU').
            transform (callable, optional): Transform for the primary (real/fake) image.
            scm_transform (callable, optional): Transform for the SCM image. If None, uses `transform`.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.scm_transform = scm_transform if scm_transform is not None else transform
        
        # --- Define the label mappings for each tier ---
        # High-Level (Binary): Real vs. Fake
        self.binary_map = {'real': 0, 'fake': 1}
        # Mid-Level (Attack Type)
        self.attack_map = {'real': 0, 'print': 1, 'replay': 2, 'none': 3}
        # Low-Level (Artifact Description)
        # 'none' is for fakes without a specific artifact listed
        self.artifact_map = {'real': 0, 'bnhalftone': 1, 'distortion': 2, 'sfchalftone': 3, 'moire': 4, 'reflection': 5, 'none': 6}
        
        # --- Find all primary image paths (real and fake) ---
        self.samples = []
        image_extensions = ['.png', '.jpg', '.jpeg']
        
        # We only scan `real` and `fake` dirs to build our sample list.
        # The SCM path will be derived on-the-fly.
        for folder in ['real', 'fake']:
            scan_path = self.root_dir / folder
            if not scan_path.exists():
                continue
            
            for ext in image_extensions:
                for image_path in scan_path.rglob(f'*{ext}'):
                    # The path relative to the dataset root (e.g., OULU/fake/print/img.png)
                    relative_path = image_path.relative_to(self.root_dir)
                    labels = self._parse_path_for_labels(relative_path.parts)
                    if labels is not None:
                        self.samples.append((image_path, labels))

    def _parse_path_for_labels(self, path_parts):
        """
        Helper function to extract hierarchical labels from path parts.
        """
        binary_label = -1
        attack_label = self.attack_map['none']
        artifact_label = self.artifact_map['none']
        
        if 'real' in path_parts:
            binary_label = self.binary_map['real']
            attack_label = self.attack_map['real']
            artifact_label = self.artifact_map['real']
            return (binary_label, attack_label, artifact_label)
        elif 'fake' in path_parts:
            binary_label = self.binary_map['fake']
        else:
            return None # Skip if not in 'real' or 'fake'

        for part in path_parts:
            if part in self.attack_map:
                attack_label = self.attack_map[part]
            if part in self.artifact_map:
                artifact_label = self.artifact_map[part]
        
        return (binary_label, attack_label, artifact_label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetches a sample and its corresponding SCM (or a zero placeholder for real images).
        
        Returns:
            tuple: (primary_image, scm_image, labels_tensor)
        """
        image_path, labels = self.samples[idx]
        
        # Load the primary image (real or fake)
        try:
            primary_image = Image.open(image_path).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Warning: Could not read image {image_path}. Skipping. Error: {e}")
            return self.__getitem__((idx + 1) % len(self))

        scm_image = None
        is_fake = (labels[0] == self.binary_map['fake'])

        if is_fake:
            # Construct the path to the corresponding SCM image
            # e.g., OULU/fake/print/img.png -> OULU/SCM/print/img.png
            relative_path_parts = image_path.relative_to(self.root_dir).parts
            scm_path = self.root_dir / 'SCM' / Path(*relative_path_parts[1:])
            
            if scm_path.exists():
                try:
                    scm_image = Image.open(scm_path).convert('RGB')
                except (IOError, OSError) as e:
                    print(f"Warning: Found fake img but failed to load SCM {scm_path}. Using zeros. Error: {e}")
                    scm_image = None # Fallback
            else:
                print(f"Warning: SCM file not found for {image_path} at {scm_path}. Using a zero tensor.")
                scm_image = None # Fallback
        
        # Apply transformations
        if self.transform is not None:
            primary_image_tensor = self.transform(primary_image)
        else:
            # Default transformation: resize to 224x224 and convert to tensor
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            primary_image_tensor = default_transform(primary_image)
        
        if scm_image is not None:
            if self.scm_transform is not None:
                scm_image_tensor = self.scm_transform(scm_image)
            else:
                # Default transformation for SCM: resize to 224x224 and convert to tensor
                default_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                scm_image_tensor = default_transform(scm_image)
        else:
            # For real images or if SCM is missing, create a zero tensor
            # of the same size as the primary image tensor.
            scm_image_tensor = torch.zeros_like(primary_image_tensor)
            
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return primary_image_tensor, scm_image_tensor, labels_tensor
