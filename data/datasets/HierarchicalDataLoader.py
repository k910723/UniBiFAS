import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

class HierarchicalFasDataset(Dataset):
    """
    A unified PyTorch Dataset for Face Anti-Spoofing that supports four modes:
    1.  **Standard Mode:** Loads real, fake, and SCM images for pre-training.
    2.  **Stage 1 (live_only=True):** Loads only real images with a zero tensor for SCM,
        used for training the visual prompt.
    3.  **Stage 2 (finetune=True):** Loads each real image twice: once as a clean 'real'
        sample and once as a 'spoof' sample blended with a visual prompt.
    4.  **Two-Class Mode (actual_spoof_dir):** Loads augmented spoofs with full hierarchy
        and actual spoof data (e.g., OCIM/OULU) for high-level binary classification only.

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
    
    For a fake image: (fake_image_tensor, scm_image_tensor, labels_tensor, is_actual_spoof_flag)
    For a real image: (real_image_tensor, zero_tensor, labels_tensor, is_actual_spoof_flag)
    """

    def __init__(self, root_dir, transform=None, scm_transform=None, live_only=False, finetune=False, visual_prompt_path=None, actual_spoof_dir=None, actual_spoof_dataset_name=None):
        """
        Args:
            root_dir (string): Directory for a specific dataset (e.g., '.../data/OULU').
            transform (callable, optional): Transform for the primary (real/fake) image.
            scm_transform (callable, optional): Transform for the SCM image. If None, uses `transform`.
            live_only (bool): If True, only loads real images (for Stage 1).
            finetune_mode (bool): If True, activates Stage 2 fine-tuning behavior.
            visual_prompt_path (string, optional): Path to visual_prompt.pth for Stage 2.
            actual_spoof_dir (string, optional): Path to directory containing .npy files or image folders
                                                 for actual spoof data (e.g., OULU-NPU).
            actual_spoof_dataset_name (string, optional): Prefix for .npy files (e.g., 'Oulu', 'casia', 'MSU', 'replay')
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.scm_transform = scm_transform if scm_transform is not None else transform
        self.live_only = live_only
        self.finetune = finetune
        self.actual_spoof_dir = Path(actual_spoof_dir) if actual_spoof_dir else None
        self.actual_spoof_dataset_name = actual_spoof_dataset_name
        
        # --- Define the label mappings for each tier ---
        # High-Level (Binary): Real vs. Fake
        self.binary_map = {'real': 0, 'fake': 1}
        # Mid-Level (Attack Type)
        self.attack_map = {'real': 0, 'print': 1, 'replay': 2, 'unseen': 3}
        # Low-Level (Artifact Description)
        # 'none' is for fakes without a specific artifact listed
        self.artifact_map = {'real': 0, 'bnhalftone': 1, 'distortion': 2, 'sfchalftone': 3, 'moire': 4, 'reflection': 5, 'unseen': 6}
        
        if self.live_only:
            scan_folders = ['real']
        elif self.finetune:
            scan_folders = ['real']
            # Load visual prompt but ensure it does NOT require gradients.
            # Saving/loading may preserve Parameter semantics; detach and disable grad to
            # avoid returning tensors with requires_grad=True from the dataset (which
            # causes DataLoader.collate to fail when stacking).
            vp = torch.load(visual_prompt_path, map_location='cpu')
            # If a Parameter was saved, get its tensor data
            if isinstance(vp, torch.nn.Parameter):
                vp = vp.data
            # Ensure no autograd history and no requires_grad
            self.visual_prompt = vp.clone().detach().requires_grad_(False)
            if self.visual_prompt.dim() == 4 and self.visual_prompt.shape[0] == 1:
                self.visual_prompt = self.visual_prompt.squeeze(0)  # now [3, H, W]
        else:
            scan_folders = ['real', 'fake']
            
        # --- Find all primary image paths (real and fake) ---
        self.samples = []
        image_extensions = ['.png', '.jpg', '.jpeg']
        
        # Track counts for reporting
        augmented_count = 0
        
        # We only scan `real` and `fake` dirs to build our sample list.
        # The SCM path will be derived on-the-fly.
        for folder in scan_folders:
            scan_path = self.root_dir / folder
            if not scan_path.exists():
                continue
            
            for ext in image_extensions:
                for image_path in scan_path.rglob(f'*{ext}'):
                    if self.finetune:
                        labels = (self.binary_map['real'], self.attack_map['real'], self.artifact_map['real'])
                        self.samples.append((image_path, labels, False))  # Clean real sample, not actual spoof
                        labels = (self.binary_map['fake'], self.attack_map['unseen'], self.artifact_map['unseen'])
                        self.samples.append((image_path, labels, False))  # Dummy labels for finetune, not actual spoof
                        augmented_count += 2
                    else:
                        # The path relative to the dataset root (e.g., OULU/fake/print/img.png)
                        relative_path = image_path.relative_to(self.root_dir)
                        labels = self._parse_path_for_labels(relative_path.parts)
                        if labels is not None:
                            self.samples.append((image_path, labels, False))  # False = not actual spoof (augmented)
                            augmented_count += 1
        
        print(f"  Loaded {augmented_count} augmented samples")
        
        # --- Load actual spoof data if provided ---
        # Supports two formats:
        # 1. Directory with image folders (real/spoof) - for F/S/W datasets (CeFA/SURF/WMCA)
        # 2. NumPy .npy files (e.g., Oulu_images_live.npy, Oulu_images_spoof.npy) - for O/C/I/M datasets
        if self.actual_spoof_dir and self.actual_spoof_dir.exists() and not self.live_only and self.actual_spoof_dataset_name:
            # First check Format 1: Load from image directories (for C/S/W)
            spoof_dir = self.actual_spoof_dir / self.actual_spoof_dataset_name / "spoof"
            real_dir = self.actual_spoof_dir / self.actual_spoof_dataset_name / "real"
            
            if spoof_dir.exists() and real_dir.exists():
                # Load actual spoof data from image directories
                actual_spoof_count = 0
                actual_live_count = 0
                print(f"  Loading actual spoof images from directories: {spoof_dir}")
                
                # Add spoof samples (only RGB/profile images)
                labels_spoof = (self.binary_map['fake'], self.attack_map['unseen'], self.artifact_map['unseen'])
                spoof_profile_dir = spoof_dir / "profile"
                if spoof_profile_dir.exists():
                    for ext in image_extensions:
                        for image_path in spoof_profile_dir.rglob(f'*{ext}'):
                            self.samples.append((image_path, labels_spoof, True))  # True = actual spoof
                            actual_spoof_count += 1
                
                # Add live samples (only RGB/profile images)
                labels_live = (self.binary_map['real'], self.attack_map['real'], self.artifact_map['real'])
                real_profile_dir = real_dir / "profile"
                if real_profile_dir.exists():
                    for ext in image_extensions:
                        for image_path in real_profile_dir.rglob(f'*{ext}'):
                            self.samples.append((image_path, labels_live, False))  # False = not actual spoof (real)
                            actual_live_count += 1
                
                print(f"  Loaded {actual_spoof_count} actual spoof and {actual_live_count} actual live samples from image directories")
                
            else:
                # Format 2: Load from specific .npy files for this dataset (for O/Ca/I/M)
                spoof_file = self.actual_spoof_dir / f"{self.actual_spoof_dataset_name}_images_spoof.npy"
                live_file = self.actual_spoof_dir / f"{self.actual_spoof_dataset_name}_images_live.npy"
                
                if spoof_file.exists() and live_file.exists():
                    # Load actual spoof data
                    try:
                        spoof_data = np.load(str(spoof_file))
                        live_data = np.load(str(live_file))
                        actual_spoof_count = len(spoof_data)
                        actual_live_count = len(live_data)
                        print(f"  Loaded {actual_spoof_count} actual spoof and {actual_live_count} actual live samples from .npy files")
                        
                        # Add spoof samples
                        labels_spoof = (self.binary_map['fake'], self.attack_map['unseen'], self.artifact_map['unseen'])
                        for idx in range(len(spoof_data)):
                            self.samples.append((spoof_file, labels_spoof, True, 'npy', idx))  # True = actual spoof
                        
                        # Add live samples
                        labels_live = (self.binary_map['real'], self.attack_map['real'], self.artifact_map['real'])
                        for idx in range(len(live_data)):
                            self.samples.append((live_file, labels_live, False, 'npy', idx))  # False = not actual spoof
                            
                    except Exception as e:
                        print(f"  Error loading .npy files: {e}")
                else:
                    print(f"  Warning: Actual spoof data not found for '{self.actual_spoof_dataset_name}'")

    def _parse_path_for_labels(self, path_parts):
        """
        Helper function to extract hierarchical labels from path parts.
        """
        binary_label = -1
        attack_label = self.attack_map['unseen']
        artifact_label = self.artifact_map['unseen']

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
            tuple: (primary_image, scm_image, labels_tensor, is_actual_spoof)
                   is_actual_spoof=True means this is actual spoof data without SCM/hierarchical labels
        """
        sample_data = self.samples[idx]
        
        # Determine format: (path, labels, is_actual_spoof) or (path, labels, is_actual_spoof, format) or (path, labels, is_actual_spoof, format, npy_idx)
        if len(sample_data) >= 5:
            # NumPy file format with index
            npy_path, labels, is_actual_spoof, fmt, npy_idx = sample_data
            # Load the specific image from the numpy array
            try:
                data = np.load(str(npy_path), mmap_mode='r')  # Use memory mapping for efficiency
                image_array = data[npy_idx]
                # Convert numpy array to PIL Image using min-max normalization
                # This handles cases where data was incorrectly divided by 255
                if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                    # Apply min-max normalization to [0, 255]
                    min_val = image_array.min()
                    max_val = image_array.max()
                    if max_val > min_val:  # Avoid division by zero
                        image_array = ((image_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        image_array = np.zeros_like(image_array, dtype=np.uint8)
                primary_image = Image.fromarray(image_array)
            except Exception as e:
                print(f"Warning: Could not read numpy array at index {npy_idx} from {npy_path}. Error: {e}")
                return self.__getitem__((idx + 1) % len(self))
        elif len(sample_data) == 4:
            # This shouldn't happen after expansion, but handle it anyway
            image_path, labels, is_actual_spoof, fmt = sample_data
            if fmt == 'image':
                try:
                    primary_image = Image.open(image_path).convert('RGB')
                except (IOError, OSError) as e:
                    print(f"Warning: Could not read image {image_path}. Error: {e}")
                    return self.__getitem__((idx + 1) % len(self))
            else:
                print(f"Warning: Unexpected format {fmt} without index")
                return self.__getitem__((idx + 1) % len(self))
        else:
            # Old format: (path, labels, is_actual_spoof)
            image_path, labels, is_actual_spoof = sample_data
            try:
                primary_image = Image.open(image_path).convert('RGB')
            except (IOError, OSError) as e:
                print(f"Warning: Could not read image {image_path}. Error: {e}")
                return self.__getitem__((idx + 1) % len(self))

        scm_image = None
        is_fake = (labels[0] == self.binary_map['fake'])

        # Only load SCM for augmented spoofs (not actual spoofs) and only for image files
        if is_fake and not self.finetune and not is_actual_spoof and len(sample_data) == 3:
            # Only try to load SCM for image files (old format with 3 elements)
            image_path = sample_data[0]
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

        # Finetune mode: overlay visual prompt on real images
        if self.finetune and is_fake:
            gamma = torch.rand(1).item() * 0.5 + 0.5 # Random gamma in [0, 0.5]
            # attack_image = (1 - gamma) * primary_image_tensor + gamma * self.visual_prompt
            attack_image = primary_image_tensor + gamma * self.visual_prompt
            # attack_image = torch.clamp(attack_image, 0, 1)
            # Absolute difference as SCM
            scm_image_tensor = torch.abs(primary_image_tensor - attack_image)
            primary_image_tensor = attack_image

        # Return flag indicating if this is actual spoof data
        return primary_image_tensor, scm_image_tensor, labels_tensor, is_actual_spoof
