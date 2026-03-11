"""
Test script to verify UniBiFAS dataloader works with C/S/W datasets.
This script tests both training and testing dataloaders.

Usage:
    python test_csw_dataloader.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from data.build import BuildLoader
import yaml

def load_config(config_path):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_training_loader():
    """Test training dataloader with FSW datasets."""
    print("\n" + "="*80)
    print("Testing Training Dataloader with FSW datasets (F=CeFA)")
    print("="*80)
    
    # Create a test config
    config = {
        'dataset': {
            'root_path': '/shared/shared/FAS_AUG',
            'test_path': '/shared/shared/',
            'source': 'F',  # Test with CeFA first (F=CeFA)
            'target': 'S',  # SURF as target
            'batch_size': 8,
            'num_workers': 2,
            'actual_spoof_dir': '/shared/shared/domain-generalization-multi',
            'load_actual_spoof': True,
        },
        'transforms': {
            'imgSize': 224,
            'mean': [0.535, 0.419, 0.358],
            'std': [0.276, 0.251, 0.242]
        }
    }
    
    try:
        # Build training dataloader
        train_loader = BuildLoader(config, isTrain=True)
        
        print(f"\nDataLoader created successfully!")
        print(f"Total batches: {len(train_loader)}")
        
        # Test loading a few batches
        print("\nTesting batch loading...")
        for i, batch in enumerate(train_loader):
            if len(batch) == 4:
                img, scm, labels, is_actual_spoof = batch
                print(f"Batch {i+1}:")
                print(f"  Image shape: {img.shape}")
                print(f"  SCM shape: {scm.shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Is actual spoof: {is_actual_spoof.sum().item()}/{len(is_actual_spoof)} samples")
                print(f"  Binary labels (real=0, fake=1): {labels[:, 0][:5].tolist()}...")
            else:
                print(f"Unexpected batch format with {len(batch)} elements")
            
            if i >= 2:  # Test first 3 batches
                break
        
        print("\n✓ Training dataloader test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Training dataloader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_testing_loader():
    """Test testing dataloader with FSW datasets."""
    print("\n" + "="*80)
    print("Testing Testing Dataloader with FSW datasets (F=CeFA)")
    print("="*80)
    
    # Create a test config
    config = {
        'dataset': {
            'root_path': '/shared/shared/FAS_AUG',
            'test_path': '/shared/shared/',
            'source': 'F',
            'target': 'F',  # Test with CeFA (F=CeFA)
            'batch_size': 8,
            'num_workers': 2,
        },
        'transforms': {
            'imgSize': 224,
            'mean': [0.535, 0.419, 0.358],
            'std': [0.276, 0.251, 0.242]
        }
    }
    
    try:
        # Build testing dataloader
        test_loader = BuildLoader(config, isTrain=False)
        
        print(f"\nDataLoader created successfully!")
        print(f"Total batches: {len(test_loader)}")
        
        # Test loading a few batches
        print("\nTesting batch loading...")
        for i, batch in enumerate(test_loader):
            img, labels = batch
            print(f"Batch {i+1}:")
            print(f"  Image shape: {img.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Binary labels (real=0, fake=1): {labels[:, 0][:5].tolist()}...")
            
            if i >= 2:  # Test first 3 batches
                break
        
        print("\n✓ Testing dataloader test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Testing dataloader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_csw_datasets():
    """Test loading all FSW datasets."""
    print("\n" + "="*80)
    print("Testing All FSW Datasets Together (F=CeFA, S=SURF, W=WMCA)")
    print("="*80)
    
    config = {
        'dataset': {
            'root_path': '/shared/shared/FAS_AUG',
            'test_path': '/shared/shared/',
            'source': 'FSW',  # All three datasets (F=CeFA)
            'target': 'F',
            'batch_size': 16,
            'num_workers': 2,
            'actual_spoof_dir': '/shared/shared/domain-generalization-multi',
            'load_actual_spoof': True,
        },
        'transforms': {
            'imgSize': 224,
            'mean': [0.535, 0.419, 0.358],
            'std': [0.276, 0.251, 0.242]
        }
    }
    
    try:
        # Build training dataloader with all CSW datasets
        train_loader = BuildLoader(config, isTrain=True)
        
        print(f"\nDataLoader created successfully!")
        print(f"Total batches: {len(train_loader)}")
        
        # Test loading one batch
        batch = next(iter(train_loader))
        if len(batch) == 4:
            img, scm, labels, is_actual_spoof = batch
            print(f"\nSample batch:")
            print(f"  Image shape: {img.shape}")
            print(f"  SCM shape: {scm.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Actual spoof samples: {is_actual_spoof.sum().item()}/{len(is_actual_spoof)}")
        
        print("\n✓ All FSW datasets test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ All FSW datasets test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("\n" + "="*80)
    print("UniBiFAS FSW Dataloader Test Suite (F=CeFA, S=SURF, W=WMCA)")
    print("="*80)
    
    results = []
    
    # Test 1: Training loader with single dataset
    results.append(("Training Loader (Single Dataset)", test_training_loader()))
    
    # Test 2: Testing loader with single dataset
    results.append(("Testing Loader (Single Dataset)", test_testing_loader()))
    
    # Test 3: All FSW datasets together
    results.append(("All FSW Datasets", test_all_csw_datasets()))
    
    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED! 🎉")
    else:
        print("SOME TESTS FAILED ❌")
    print("="*80)
    
    sys.exit(0 if all_passed else 1)
