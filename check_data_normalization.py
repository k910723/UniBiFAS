#!/usr/bin/env python3
"""
Script to check data normalization in TestFASDataset (target dataloader)
Analyzes the raw data ranges and what values the model actually receives after preprocessing.
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

# python check_data_normalization.py --test-path /shared/shared/ --target OCIM

# Import the test dataloader
import sys
sys.path.append(str(Path(__file__).parent / 'data' / 'datasets'))
from TestDataLoader import TestFASDataset


def analyze_raw_data(root_dir, protocols):
    """Analyze the raw .npy files before any preprocessing"""
    print("="*80)
    print("ANALYZING RAW DATA FILES")
    print("="*80)
    
    protocol_map = {
        'O': 'Oulu',
        'C': 'casia',
        'I': 'replay',
        'M': 'MSU',
        'D': '3DMAD',
        'H': 'HKBUv1+',
        'U': 'casia_3d',
    }
    
    root_path = Path(root_dir)
    all_stats = []
    
    for p in protocols:
        if p not in protocol_map:
            print(f"Skipping unknown protocol: {p}")
            continue
            
        base_name = protocol_map[p]
        
        if p in ['D', 'H', 'U']:
            data_path = root_path / "3Dmask"
        else:
            data_path = root_path / "domain-generalization"
        
        for label in ['live', 'spoof']:
            file_path = data_path / f"{base_name}_images_{label}.npy"
            
            if not file_path.exists():
                print(f"⚠ File not found: {file_path}")
                continue
            
            print(f"\n📁 Loading: {file_path.name}")
            data = np.load(file_path)
            
            stats = {
                'file': file_path.name,
                'protocol': p,
                'label': label,
                'shape': data.shape,
                'dtype': data.dtype,
                'min': data.min(),
                'max': data.max(),
                'mean': data.mean(),
                'std': data.std(),
                'unique_values': len(np.unique(data)),
                'data': data  # Keep for later
            }
            
            print(f"  Shape: {stats['shape']}")
            print(f"  Dtype: {stats['dtype']}")
            print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std: {stats['std']:.6f}")
            print(f"  Unique values: {stats['unique_values']}")
            
            # Check if data looks problematic
            if stats['max'] < 0.1:
                print(f"  🚨 WARNING: Max value ({stats['max']:.6f}) is very small!")
                print(f"     This data appears to be in a compressed range, not [0,1] or [0,255]")
            elif stats['max'] <= 1.0:
                print(f"  ✓ Data appears to be in [0, 1] range")
            elif stats['max'] <= 255.0:
                print(f"  ✓ Data appears to be in [0, 255] range")
            else:
                print(f"  ⚠ Data has unusual range (max > 255)")
            
            all_stats.append(stats)
    
    return all_stats


def simulate_preprocessing(stats_list):
    """Simulate what happens during preprocessing in TestFASDataset.__getitem__"""
    print("\n" + "="*80)
    print("SIMULATING PREPROCESSING (TestFASDataset.__getitem__ logic)")
    print("="*80)
    
    for stats in stats_list:
        print(f"\n📊 File: {stats['file']} ({stats['label']})")
        print(f"   Original range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        
        # Take a sample image
        sample_img = stats['data'][0].copy()
        sample_img = np.squeeze(sample_img)
        
        print(f"   Sample shape after squeeze: {sample_img.shape}")
        print(f"   Sample dtype: {sample_img.dtype}")
        print(f"   Sample range: [{sample_img.min():.6f}, {sample_img.max():.6f}]")
        
        # Apply the preprocessing logic from TestDataLoader
        if sample_img.dtype == np.float32 or sample_img.dtype == np.float64:
            if sample_img.max() <= 1.0:
                print(f"   Action: Multiplying by 255 (assuming [0,1] range)")
                processed = (sample_img * 255).astype(np.uint8)
            else:
                print(f"   Action: Dividing by 255 (DHU case)")
                processed = (sample_img / 255).astype(np.uint8)
        else:
            print(f"   Action: Direct conversion to uint8")
            processed = sample_img.astype(np.uint8)
        
        print(f"   After preprocessing:")
        print(f"     dtype: {processed.dtype}")
        print(f"     range: [{processed.min()}, {processed.max()}]")
        print(f"     unique values: {len(np.unique(processed))}")
        
        # Check if preprocessing is problematic
        if processed.max() <= 1:
            print(f"   🚨 CRITICAL ISSUE: After preprocessing, max value is {processed.max()}")
            print(f"      The image is essentially BLACK/ZERO!")
            print(f"      Model will NOT see actual image content!")
        elif processed.max() < 50:
            print(f"   ⚠ WARNING: After preprocessing, max value is only {processed.max()}")
            print(f"      Image is very dark, most information lost!")
        else:
            print(f"   ✓ Preprocessing looks reasonable")


def test_full_pipeline(root_dir, protocols, num_samples=5):
    """Test the full pipeline including transforms"""
    print("\n" + "="*80)
    print("TESTING FULL PIPELINE (with CLIP normalization)")
    print("="*80)
    
    # CLIP's standard normalization
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std)
    ])
    
    dataset = TestFASDataset(root_dir, protocols, transform=transform)
    
    print(f"\nDataset size: {len(dataset)} images")
    print(f"Analyzing first {num_samples} samples...\n")
    
    for i in range(min(num_samples, len(dataset))):
        img_tensor, label = dataset[i]
        
        print(f"Sample {i}:")
        print(f"  Label: {label[0].item()} ({'live' if label[0].item() == 0 else 'spoof'})")
        print(f"  Tensor shape: {img_tensor.shape}")
        print(f"  Tensor dtype: {img_tensor.dtype}")
        print(f"  After CLIP normalization:")
        print(f"    Range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
        print(f"    Mean: {img_tensor.mean():.4f}")
        print(f"    Std: {img_tensor.std():.4f}")
        
        # Check each channel
        for c in range(3):
            print(f"    Channel {c}: [{img_tensor[c].min():.4f}, {img_tensor[c].max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description='Check data normalization in target dataloader')
    parser.add_argument('--test-path', type=str, required=True,
                        help='Path to test data directory (e.g., /shared/shared/)')
    parser.add_argument('--target', type=str, default='O',
                        help='Target protocol(s) to test (e.g., O, I, C, M, OCI)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to analyze in full pipeline test')
    
    args = parser.parse_args()
    
    # Parse protocols
    protocols = list(args.target)
    
    print(f"Testing with protocols: {protocols}")
    print(f"Data path: {args.test_path}\n")
    
    try:
        # Step 1: Analyze raw data
        stats = analyze_raw_data(args.test_path, protocols)
        
        if not stats:
            print("\n❌ No data found. Check your --test-path and --target arguments.")
            return
        
        # Step 2: Simulate preprocessing
        simulate_preprocessing(stats)
        
        # Step 3: Test full pipeline
        test_full_pipeline(args.test_path, protocols, args.num_samples)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        problematic = [s for s in stats if s['max'] < 0.1]
        if problematic:
            print(f"\n🚨 CRITICAL ISSUES FOUND:")
            print(f"   {len(problematic)}/{len(stats)} files have data in very small range (< 0.1)")
            print(f"   These will become essentially zero after preprocessing!")
            print(f"\n   Affected files:")
            for s in problematic:
                print(f"     - {s['file']}: range [{s['min']:.6f}, {s['max']:.6f}]")
            print(f"\n   Recommendation: Fix normalization and RETRAIN model")
        else:
            print(f"\n✓ All data files appear to be in reasonable ranges")
            print(f"   Preprocessing should work correctly")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
