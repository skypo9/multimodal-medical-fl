"""
Multi-Modal Medical Federated Learning Simulation
=================================================

This script implements federated learning across different medical imaging modalities
using publicly available datasets. It demonstrates cross-modal federated learning
for medical AI applications.

Datasets Used:
1. Skin Cancer Classification (Dermoscopy Images) - HAM10000 dataset
2. Pneumonia Detection (Chest X-rays) - Chest X-ray Pneumonia dataset

Author: Akuila Pohiva
Date: August 2025
"""

import argparse
import collections
import os
import os.path
import csv
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, random_split

    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
except ImportError as e:
    raise ImportError(
        "Missing required packages. Please install torch and torchvision: "
        "pip install torch torchvision"
    ) from e

try:
    import flwr as fl
except ImportError as e:
    raise ImportError(
        "Missing Flower framework. Please install with: pip install flwr[simulation]"
    ) from e


######################################################################
# Dataset loading and preparation
######################################################################

def quick_dataset_status_check(base_dir: str) -> Dict[str, bool]:
    """Quick check of all datasets status without downloading.
    
    Parameters
    ----------
    base_dir : str
        Base directory where datasets should be stored
        
    Returns
    -------
    Dict[str, bool]
        Dictionary mapping dataset names to their availability status
    """
    datasets_info = {
        "skin_cancer": ["benign", "malignant"],
        "pneumonia_xray": ["normal", "pneumonia"]
    }
    
    status = {}
    print(f"\n[QUICK DATASET STATUS CHECK]")
    print(f"{'='*50}")
    
    for dataset_name, expected_classes in datasets_info.items():
        dataset_dir = os.path.join(base_dir, dataset_name)
        is_available = check_dataset_exists_and_organized(dataset_dir, expected_classes, min_samples_per_class=5)
        status[dataset_name] = is_available
        
        if is_available:
            print(f"✅ {dataset_name}: Ready")
        else:
            print(f"❌ {dataset_name}: Not available or incomplete")
    
    available_count = sum(status.values())
    print(f"\n📊 Summary: {available_count}/{len(status)} datasets available")
    
    if available_count == len(status):
        print("🚀 All datasets ready! Can proceed with federated learning.")
    elif available_count > 0:
        print("⚠️  Some datasets available. Missing datasets will be downloaded or created as synthetic.")
    else:
        print("📥 No datasets found. Will download or create synthetic datasets.")
    
    return status

def check_dataset_exists_and_organized(dataset_dir: str, expected_classes: List[str], min_samples_per_class: int = 10) -> bool:
    """Comprehensive check if a dataset exists and is properly organized.
    
    Parameters
    ----------
    dataset_dir : str
        Path to the dataset directory
    expected_classes : List[str]
        List of expected class folder names
    min_samples_per_class : int
        Minimum number of samples required per class
        
    Returns
    -------
    bool
        True if dataset exists, is organized, and has sufficient samples
    """
    if not os.path.exists(dataset_dir):
        return False
    
    print(f"🔍 Checking dataset: {os.path.basename(dataset_dir)}")
    
    total_samples = 0
    for class_name in expected_classes:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"   ❌ Missing class directory: {class_name}")
            return False
        
        # Count image files
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if len(image_files) < min_samples_per_class:
            print(f"   ❌ Insufficient samples in {class_name}: {len(image_files)} < {min_samples_per_class}")
            return False
        
        total_samples += len(image_files)
        print(f"   ✅ {class_name}: {len(image_files)} samples")
    
    print(f"   📊 Total samples: {total_samples}")
    return True

def download_multimodal_datasets(base_dir: str) -> List[str]:
    """Download multi-modal medical datasets from Kaggle.
    
    Parameters
    ----------
    base_dir : str
        Base directory where datasets will be stored
        
    Returns
    -------
    List[str]
        Paths to the dataset directories
    """
    import os
    import shutil
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    # Define multi-modal medical datasets
    kaggle_datasets = {
        "skin_cancer": {
            "dataset": "kmader/skin-cancer-mnist-ham10000",
            "description": "Skin Cancer Classification (Dermoscopy)",
            "modality": "Dermoscopy",
            "task": "Skin lesion classification",
            "classes": ["benign", "malignant"],
            "organization_type": "csv_based"  # Uses CSV for labels
        },
        "pneumonia_xray": {
            "dataset": "paultimothymooney/chest-xray-pneumonia",
            "description": "Pneumonia Detection (Chest X-ray)",
            "modality": "Chest Radiography", 
            "task": "Pneumonia detection",
            "classes": ["NORMAL", "PNEUMONIA"],
            "class_mapping": {"NORMAL": "normal", "PNEUMONIA": "pneumonia"},
            "organization_type": "folder_based"  # Pre-organized in folders
        }
    }
    
    dataset_paths = []
    
    for dataset_name, info in kaggle_datasets.items():
        dataset_dir = os.path.join(base_dir, dataset_name)
        
        print(f"\n{'='*60}")
        print(f"Processing {info['description']}")
        print(f"Modality: {info['modality']}")
        print(f"Task: {info['task']}")
        print(f"{'='*60}")
        
        # Enhanced existence check with comprehensive validation
        if info["organization_type"] == "folder_based":
            expected_classes = list(info["class_mapping"].values())
            if check_dataset_exists_and_organized(dataset_dir, expected_classes):
                print(f"✅ {dataset_name} dataset already exists and is properly organized. Skipping download.")
                dataset_paths.append(dataset_dir)
                continue
        else:
            # For CSV-based datasets, check if organized folders exist
            expected_classes = info["classes"]
            if check_dataset_exists_and_organized(dataset_dir, expected_classes):
                print(f"✅ {dataset_name} dataset already exists and is properly organized. Skipping download.")
                dataset_paths.append(dataset_dir)
                continue
            
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"⬇️  Downloading {dataset_name} dataset...")
        
        try:
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Download dataset
            api.dataset_download_files(info["dataset"], path=dataset_dir, unzip=True)
            
            # Organize dataset based on type
            if dataset_name == "pneumonia_xray":
                organize_pneumonia_dataset(dataset_dir, info)
            elif dataset_name == "skin_cancer":
                organize_skin_cancer_dataset(dataset_dir, info)
            
            dataset_paths.append(dataset_dir)
            print(f"✅ Downloaded and organized {dataset_name} dataset")
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {e}")
            print(f"🔧 Creating synthetic {dataset_name} dataset for demonstration...")
            
            # Create synthetic dataset
            if create_synthetic_dataset(dataset_dir, info):
                dataset_paths.append(dataset_dir)
                print(f"✅ Created synthetic {dataset_name} dataset")
    
    return dataset_paths

def create_synthetic_dataset(dataset_dir: str, info: Dict) -> bool:
    """Create a synthetic medical dataset for demonstration purposes."""
    import os
    import shutil
    from PIL import Image, ImageDraw
    import random
    
    try:
        # Determine class structure
        if info["organization_type"] == "folder_based":
            classes = list(info["class_mapping"].values())
        else:
            classes = info["classes"]
        
        print(f"📁 Creating synthetic dataset with classes: {classes}")
        
        # Create class directories
        for class_name in classes:
            class_dir = os.path.join(dataset_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Generate synthetic medical images
            num_images = 100  # Create 100 synthetic images per class
            for i in range(num_images):
                # Create synthetic medical-like image
                img = create_synthetic_medical_image(info["modality"], class_name)
                img_path = os.path.join(class_dir, f"synthetic_{class_name}_{i:03d}.png")
                img.save(img_path)
        
        print(f"🎨 Generated {num_images} synthetic images per class")
        return True
        
    except Exception as e:
        print(f"❌ Error creating synthetic dataset: {e}")
        return False

def create_synthetic_medical_image(modality: str, class_name: str) -> Image.Image:
    """Create a synthetic medical image based on modality and class."""
    from PIL import Image, ImageDraw
    import random
    
    # Create base image
    img = Image.new('RGB', (224, 224), color='black')
    draw = ImageDraw.Draw(img)
    
    if modality == "Dermoscopy":
        # Create skin-like texture
        base_color = (180, 140, 120) if "benign" in class_name else (160, 100, 80)
        
        # Add skin texture
        for _ in range(500):
            x = random.randint(0, 223)
            y = random.randint(0, 223)
            color_var = random.randint(-20, 20)
            pixel_color = tuple(max(0, min(255, c + color_var)) for c in base_color)
            draw.point((x, y), fill=pixel_color)
        
        # Add lesion-like features
        if "malignant" in class_name:
            # More irregular shapes for malignant
            for _ in range(3):
                x = random.randint(50, 174)
                y = random.randint(50, 174)
                size = random.randint(20, 40)
                color = (80, 40, 20)
                draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
        else:
            # Regular shapes for benign
            x = random.randint(80, 144)
            y = random.randint(80, 144)
            size = random.randint(15, 25)
            color = (120, 80, 60)
            draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
    
    elif modality == "Chest Radiography":
        # Create chest X-ray like image
        if "normal" in class_name:
            # Clear lungs
            base_color = (40, 40, 40)
            lung_color = (20, 20, 20)
        else:
            # Cloudy/infected lungs
            base_color = (60, 60, 60)
            lung_color = (80, 80, 80)
        
        # Fill background
        draw.rectangle([0, 0, 223, 223], fill=base_color)
        
        # Draw lung shapes
        # Left lung
        draw.ellipse([30, 50, 100, 180], fill=lung_color)
        # Right lung
        draw.ellipse([124, 50, 194, 180], fill=lung_color)
        
        # Add ribs
        for i in range(8):
            y = 60 + i * 15
            draw.arc([20, y-5, 204, y+5], start=0, end=180, fill=(60, 60, 60), width=2)
        
        # Add pathology if pneumonia
        if "pneumonia" in class_name:
            for _ in range(5):
                x = random.randint(40, 180)
                y = random.randint(70, 160)
                size = random.randint(8, 15)
                draw.ellipse([x-size, y-size, x+size, y+size], fill=(120, 120, 120))
    
    return img

def organize_pneumonia_dataset(dataset_dir: str, info: Dict):
    """Organize the pneumonia chest X-ray dataset."""
    import shutil
    
    print("📁 Organizing Pneumonia dataset...")
    
    # Check for typical structure: chest_xray/train, chest_xray/test, chest_xray/val
    chest_xray_dir = os.path.join(dataset_dir, "chest_xray")
    
    if os.path.exists(chest_xray_dir):
        # Create organized structure
        normal_dir = os.path.join(dataset_dir, "normal")
        pneumonia_dir = os.path.join(dataset_dir, "pneumonia")
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(pneumonia_dir, exist_ok=True)
        
        # Combine all train/test/val data
        source_dirs = ["train", "test", "val"]
        normal_count = 0
        pneumonia_count = 0
        max_per_class = 1000  # Limit for manageable dataset size
        
        for source_dir in source_dirs:
            source_path = os.path.join(chest_xray_dir, source_dir)
            if os.path.exists(source_path):
                # Process NORMAL images
                normal_source = os.path.join(source_path, "NORMAL")
                if os.path.exists(normal_source) and normal_count < max_per_class:
                    for img_file in os.listdir(normal_source):
                        if normal_count >= max_per_class:
                            break
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            src_path = os.path.join(normal_source, img_file)
                            dst_path = os.path.join(normal_dir, f"{source_dir}_{img_file}")
                            shutil.copy2(src_path, dst_path)
                            normal_count += 1
                
                # Process PNEUMONIA images
                pneumonia_source = os.path.join(source_path, "PNEUMONIA")
                if os.path.exists(pneumonia_source) and pneumonia_count < max_per_class:
                    for img_file in os.listdir(pneumonia_source):
                        if pneumonia_count >= max_per_class:
                            break
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            src_path = os.path.join(pneumonia_source, img_file)
                            dst_path = os.path.join(pneumonia_dir, f"{source_dir}_{img_file}")
                            shutil.copy2(src_path, dst_path)
                            pneumonia_count += 1
        
        print(f"📊 Organized {normal_count} normal and {pneumonia_count} pneumonia chest X-rays")

def organize_skin_cancer_dataset(dataset_dir: str, info: Dict):
    """Organize the skin cancer dataset using CSV metadata."""
    import shutil
    
    print("📁 Organizing Skin Cancer dataset...")
    
    # Look for HAM10000 images and metadata
    images_dir = None
    metadata_file = None
    
    # Find images directory and metadata file
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.csv') and 'metadata' in file.lower():
                metadata_file = os.path.join(root, file)
            elif file.endswith('.csv') and any(x in file.lower() for x in ['ham', 'lesion']):
                metadata_file = os.path.join(root, file)
        
        # Look for directory with many images
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if len([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]) > 100:
                images_dir = dir_path
    
    # If metadata approach fails, create synthetic organization
    if not metadata_file or not images_dir:
        print("📋 Metadata not found, using directory-based organization...")
        
        # Find any directory with images
        for root, dirs, files in os.walk(dataset_dir):
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if len(image_files) > 10:
                images_dir = root
                break
        
        if images_dir:
            # Create synthetic organization based on filename patterns or random assignment
            organize_skin_images_synthetically(dataset_dir, images_dir)
    else:
        # Use metadata file for proper organization
        organize_skin_images_with_metadata(dataset_dir, images_dir, metadata_file)

def organize_skin_images_synthetically(dataset_dir: str, images_dir: str):
    """Organize skin cancer images synthetically when metadata is not available."""
    import shutil
    import random
    
    benign_dir = os.path.join(dataset_dir, "benign")
    malignant_dir = os.path.join(dataset_dir, "malignant")
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Randomly assign to benign/malignant for demonstration
    # In practice, you'd use actual medical labels
    random.shuffle(image_files)
    
    benign_count = 0
    malignant_count = 0
    max_per_class = 500
    
    for i, img_file in enumerate(image_files):
        if benign_count >= max_per_class and malignant_count >= max_per_class:
            break
            
        src_path = os.path.join(images_dir, img_file)
        
        # Assign based on even/odd index for demonstration
        if i % 2 == 0 and benign_count < max_per_class:
            dst_path = os.path.join(benign_dir, img_file)
            shutil.copy2(src_path, dst_path)
            benign_count += 1
        elif malignant_count < max_per_class:
            dst_path = os.path.join(malignant_dir, img_file)
            shutil.copy2(src_path, dst_path)
            malignant_count += 1
    
    print(f"📊 Organized {benign_count} benign and {malignant_count} malignant skin lesion images")

def organize_skin_images_with_metadata(dataset_dir: str, images_dir: str, metadata_file: str):
    """Organize skin cancer images using metadata CSV file."""
    import shutil
    
    try:
        import pandas as pd
        
        benign_dir = os.path.join(dataset_dir, "benign")
        malignant_dir = os.path.join(dataset_dir, "malignant")
        os.makedirs(benign_dir, exist_ok=True)
        os.makedirs(malignant_dir, exist_ok=True)
        
        # Read metadata
        metadata_df = pd.read_csv(metadata_file)
        
        # Map diagnosis to benign/malignant
        benign_diagnoses = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']  # Adapt based on actual labels
        malignant_diagnoses = ['mel']  # Melanoma is malignant
        
        benign_count = 0
        malignant_count = 0
        max_per_class = 500
        
        for _, row in metadata_df.iterrows():
            if benign_count >= max_per_class and malignant_count >= max_per_class:
                break
                
            image_id = row.get('image_id', row.get('lesion_id', ''))
            diagnosis = row.get('dx', row.get('diagnosis', ''))
            
            # Find corresponding image file
            possible_extensions = ['.jpg', '.png', '.jpeg']
            src_path = None
            
            for ext in possible_extensions:
                candidate_path = os.path.join(images_dir, f"{image_id}{ext}")
                if os.path.exists(candidate_path):
                    src_path = candidate_path
                    break
            
            if src_path:
                if diagnosis in malignant_diagnoses and malignant_count < max_per_class:
                    dst_path = os.path.join(malignant_dir, f"{image_id}.jpg")
                    shutil.copy2(src_path, dst_path)
                    malignant_count += 1
                elif diagnosis not in malignant_diagnoses and benign_count < max_per_class:
                    dst_path = os.path.join(benign_dir, f"{image_id}.jpg")
                    shutil.copy2(src_path, dst_path)
                    benign_count += 1
        
        print(f"📊 Organized {benign_count} benign and {malignant_count} malignant skin lesions using metadata")
        
    except Exception as e:
        print(f"⚠️  Error using metadata: {e}")
        print("🔄 Falling back to synthetic organization...")
        organize_skin_images_synthetically(dataset_dir, images_dir)

def load_multimodal_dataset(data_dir: str, num_clients: int, sample_fraction: float = 1.0) -> List[Tuple[Dataset, Dataset]]:
    """Load a multi-modal medical imaging dataset and split it among clients.

    Parameters
    ----------
    data_dir : str
        Path to the root directory of the dataset
    num_clients : int
        Number of clients to create
    sample_fraction : float
        Fraction of the dataset to use (0.0 to 1.0). Default is 1.0 (use all data)

    Returns
    -------
    List[Tuple[Dataset, Dataset]]
        A list of (train_dataset, test_dataset) pairs for each client
    """
    # Define transforms optimized for different medical imaging modalities
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),  # Less aggressive for medical images
        transforms.RandomRotation(degrees=10),   # Slight rotation for augmentation
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Subtle color adjustment
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Create a binary classification dataset
    if os.path.isdir(data_dir):
        samples = []
        class_counts = {}
        
        # Get all class directories
        all_class_dirs = [d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
        
        print(f"🔍 Found classes: {all_class_dirs}")
        
        # Dataset-specific class selection logic
        dataset_name = os.path.basename(data_dir)
        
        if "skin_cancer" in dataset_name.lower():
            # For skin cancer: use benign and malignant
            preferred_classes = ["benign", "malignant"]
        elif "pneumonia" in dataset_name.lower() or "chest" in dataset_name.lower():
            # For pneumonia: use normal and pneumonia (skip chest_xray folder which contains subdirs)
            preferred_classes = ["normal", "pneumonia"]
        else:
            # Generic case: will be determined by available classes with data
            preferred_classes = []
        
        # Find classes that actually have image data
        valid_classes = []
        for class_name in all_class_dirs:
            class_dir = os.path.join(data_dir, class_name)
            # Check if class has actual image files
            image_files = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.bmp', '.tiff'))]
            if len(image_files) > 0:
                valid_classes.append((class_name, len(image_files)))
        
        print(f"📊 Valid classes with image data: {[(name, count) for name, count in valid_classes]}")
        
        # Select target classes based on preferences and data availability
        if preferred_classes:
            # Use preferred classes if they have data
            target_classes = []
            for pref_class in preferred_classes:
                if any(name == pref_class for name, count in valid_classes):
                    target_classes.append(pref_class)
            
            # If we don't have enough preferred classes, fall back to available ones
            if len(target_classes) < 2:
                print(f"⚠️  Not enough preferred classes ({target_classes}), using available classes")
                target_classes = [name for name, count in sorted(valid_classes, key=lambda x: -x[1])[:2]]
        else:
            # Use the 2 classes with most data
            target_classes = [name for name, count in sorted(valid_classes, key=lambda x: -x[1])[:2]]
        
        print(f"🎯 Target classes for binary classification: {target_classes}")
        
        # Process only the target classes
        for i, class_name in enumerate(target_classes):
            if class_name not in all_class_dirs:
                print(f"⚠️  Warning: Target class '{class_name}' not found in dataset")
                continue
                
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                target = i  # 0 for first target class, 1 for second target class
                
                class_samples = []
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                        img_path = os.path.join(class_dir, img_name)
                        class_samples.append((img_path, target))
                
                # Apply sampling if requested
                if sample_fraction < 1.0:
                    import random
                    sample_size = int(len(class_samples) * sample_fraction)
                    class_samples = random.sample(class_samples, min(sample_size, len(class_samples)))
                    print(f"🔬 Sampled {len(class_samples)} images from {class_name} class")
                else:
                    print(f"🔬 Loaded {len(class_samples)} images from {class_name} class")
                
                samples.extend(class_samples)
                class_counts[class_name] = len(class_samples)
        
        # Skip other classes that weren't selected
        for class_name in all_class_dirs:
            if class_name not in target_classes:
                print(f"⏭️  Skipping class '{class_name}' - not in target classes for binary classification")

        print(f"📊 Total samples loaded: {len(samples)}")
        print(f"📋 Class distribution: {class_counts}")
        
        # Validate that we have a proper binary classification setup
        if len(class_counts) != 2:
            print(f"❌ Error: Expected 2 classes for binary classification, got {len(class_counts)}")
            print(f"   Available classes: {list(class_counts.keys())}")
            return []
        
        if min(class_counts.values()) == 0:
            print(f"❌ Error: One or more classes has no samples")
            print(f"   Class counts: {class_counts}")
            return []

        print(f"📊 Total samples loaded: {len(samples)}")
        print(f"📋 Class distribution: {class_counts}")

        # Create a dataset with our binary labels
        class MultiModalMedicalDataset(Dataset):
            def __init__(self, samples, transform=None):
                self.samples = samples
                self.transform = transform

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img_path, target = self.samples[idx]
                try:
                    img = Image.open(img_path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    return img, target
                except Exception as e:
                    print(f"⚠️  Error loading image {img_path}: {e}")
                    # Return a black image as fallback
                    img = Image.new('RGB', (224, 224), color='black')
                    if self.transform:
                        img = self.transform(img)
                    return img, target

        dataset = MultiModalMedicalDataset(samples, transform=train_transform)
    else:
        # Fallback: create a synthetic dataset if directory does not exist
        print(f"⚠️  Warning: dataset directory '{data_dir}' not found.")
        print("🔧 Using synthetic data for demonstration purposes.")
        
        class DummyMultiModalDataset(Dataset):
            def __init__(self, n_samples: int = 1000):
                self.images = torch.rand(n_samples, 3, 224, 224)
                self.labels = torch.randint(0, 2, (n_samples,))

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]

        dataset = DummyMultiModalDataset(n_samples=2000)

    # Split dataset into train/test sets for each client
    total_size = len(dataset)
    if total_size == 0:
        print(f"❌ Warning: No data found in {data_dir}")
        return []
        
    base_size = total_size // num_clients
    remainder = total_size % num_clients
    split_sizes = [base_size + 1 if i < remainder else base_size 
                  for i in range(num_clients)]
    
    client_datasets = random_split(dataset, split_sizes)
    clients_data = []
    
    for i, client_ds in enumerate(client_datasets):
        # Split each client's data into train/test (80/20)
        n_total = len(client_ds)
        if n_total == 0:
            clients_data.append(([], []))
            continue
            
        n_train = int(0.8 * n_total)
        n_test = n_total - n_train
        
        if n_train == 0:
            # If too small, use all data for training
            train_ds, test_ds = client_ds, []
        else:
            train_ds, test_ds = random_split(client_ds, [n_train, n_test])
            
        clients_data.append((train_ds, test_ds))
        print(f"👥 Client {i+1}: {len(train_ds)} training, {len(test_ds)} test samples")

    return clients_data


######################################################################
# Enhanced Model definition for multi-modal medical imaging
######################################################################

class MultiModalMedicalCNN(nn.Module):
    """Enhanced CNN architecture for multi-modal medical image classification.
    
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super().__init__()
        
        # Feature extraction backbone
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # Block 4 - Deep features
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Classification
        output = self.classifier(attended_features)
        return output


######################################################################
# Enhanced Federated client implementation
######################################################################

class MultiModalMedicalClient(fl.client.NumPyClient):
    """Enhanced Flower client for multi-modal medical imaging."""

    def __init__(self, model: nn.Module, train_ds: Dataset, 
                 test_ds: Dataset, device: torch.device, client_name: str = ""):
        self.model = model
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.device = device
        self.client_name = client_name

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        train_loader = DataLoader(self.train_ds, batch_size=16, shuffle=True)  # Smaller batch for medical images
        
        # Enhanced training with learning rate scheduling
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
        
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            # Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        scheduler.step(avg_loss)

        return self.get_parameters({}), len(self.train_ds), {
            "loss": avg_loss,
            "client": self.client_name
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        test_loader = DataLoader(self.test_ds, batch_size=16)
        
        # Enhanced evaluation with detailed metrics
        correct, total = 0, 0
        loss = 0.0
        predictions = []
        true_labels = []
        
        self.model.eval()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss += F.cross_entropy(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(target.cpu().numpy())

        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        try:
            precision = precision_score(true_labels, predictions, average='binary', zero_division=0)
            recall = recall_score(true_labels, predictions, average='binary', zero_division=0)
            f1 = f1_score(true_labels, predictions, average='binary', zero_division=0)
        except:
            precision = recall = f1 = 0.0

        return loss, len(self.test_ds), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "client": self.client_name
        }


def validate_dataset_for_federated_learning(dataset_paths: List[str]) -> bool:
    """Validate datasets before starting federated learning.
    
    Parameters
    ----------
    dataset_paths : List[str]
        List of paths to datasets
        
    Returns
    -------
    bool
        True if all datasets are valid for federated learning
    """
    print(f"\n🔍 DATASET VALIDATION FOR FEDERATED LEARNING")
    print(f"{'='*60}")
    
    all_valid = True
    
    for i, dataset_path in enumerate(dataset_paths):
        dataset_name = os.path.basename(dataset_path)
        print(f"\n📋 Validating Dataset {i+1}: {dataset_name}")
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            print(f"   ❌ Dataset directory not found")
            all_valid = False
            continue
        
        # Get class directories
        class_dirs = [d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')]
        
        if len(class_dirs) < 2:
            print(f"   ❌ Insufficient classes: {len(class_dirs)} (need at least 2)")
            all_valid = False
            continue
        
        # Check class sample counts (only for directories with images)
        class_counts = {}
        total_samples = 0
        
        for class_name in class_dirs:
            class_dir = os.path.join(dataset_path, class_name)
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.bmp', '.tiff'))]
            if len(image_files) > 0:  # Only count classes with actual images
                class_counts[class_name] = len(image_files)
                total_samples += len(image_files)
        
        print(f"   📊 Classes with images: {list(class_counts.keys())}")
        print(f"   📈 Samples per class: {class_counts}")
        print(f"   📉 Total samples: {total_samples}")
        
        # Check for empty directories (directories without images)
        empty_dirs = [name for name in class_dirs if name not in class_counts]
        if empty_dirs:
            print(f"   📁 Empty directories (no images): {empty_dirs}")
        
        # Validation checks
        if len(class_counts) < 2:
            print(f"   ❌ Insufficient classes with images: {len(class_counts)} (need at least 2)")
            all_valid = False
        elif total_samples == 0:
            print(f"   ❌ No image samples found")
            all_valid = False
        elif total_samples < 20:
            print(f"   ⚠️  Warning: Very few samples ({total_samples}) - results may not be reliable")
        elif min(class_counts.values()) < 5:
            print(f"   ⚠️  Warning: Some classes have very few samples (min: {min(class_counts.values())})")
        elif max(class_counts.values()) / min(class_counts.values()) > 10:
            print(f"   ⚠️  Warning: High class imbalance detected")
        else:
            print(f"   ✅ Dataset validation passed")
    
    print(f"\n{'='*60}")
    if all_valid:
        print(f"✅ ALL DATASETS VALID - Ready for federated learning!")
    else:
        print(f"❌ VALIDATION FAILED - Please fix dataset issues before proceeding")
    
    return all_valid


######################################################################
# Enhanced FedBN Strategy for multi-modal learning
######################################################################

class MultiModalFedBNStrategy(fl.server.strategy.FedAvg):
    """Enhanced FedBN strategy for multi-modal medical imaging.
    
    Implements domain-specific batch normalization to handle
    different medical imaging modalities effectively.
    """

    def aggregate_fit(self, rnd: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], failures: List[BaseException]) -> Tuple[List[np.ndarray], Dict]:
        if not results:
            return None, {}
        
        # Convert parameters to numpy arrays
        weights = [result.parameters for _, result in results]
        parameters_list = [fl.common.parameters_to_ndarrays(w) for w in weights]

        # Compute total examples
        total_examples = sum(result.num_examples for _, result in results)

        # Initialize aggregated parameters
        new_params = [np.zeros_like(p) for p in parameters_list[0]]

        # Identify BN layer indices (more comprehensive for deeper model)
        dummy_model = MultiModalMedicalCNN()
        bn_indices = [i for i, (name, _) in enumerate(dummy_model.state_dict().items()) 
                     if "bn" in name or "norm" in name]

        print(f"🔄 Round {rnd}: Aggregating parameters (excluding {len(bn_indices)} BN layers)")

        # Aggregate parameters except BN layers
        for params, (_, result) in zip(parameters_list, results):
            num_examples = result.num_examples
            for i, p in enumerate(params):
                if i in bn_indices:
                    continue  # Skip BN layers for domain-specific statistics
                new_params[i] += p * (num_examples / total_examples)

        # Keep BN parameters from first client (or could implement more sophisticated selection)
        for idx in bn_indices:
            new_params[idx] = parameters_list[0][idx]

        # Aggregate metrics
        metrics = {"round": rnd}
        for _, result in results:
            for k, v in result.metrics.items():
                if isinstance(v, (int, float)):
                    metrics.setdefault(k, 0.0)
                    metrics[k] += v * (result.num_examples / total_examples)

        aggregated = fl.common.ndarrays_to_parameters(new_params)
        return aggregated, metrics


######################################################################
# Main simulation driver for multi-modal federated learning
######################################################################

def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-Modal Medical Federated Learning")
    parser.add_argument("--data_dir", type=str, default="./multimodal_datasets",
                      help="Path to the base datasets directory")
    parser.add_argument("--clients", type=int, default=2,
                      help="Number of federated clients (max 2 for this demo)")
    parser.add_argument("--rounds", type=int, default=5,
                      help="Number of federated rounds")
    parser.add_argument("--sample_fraction", type=float, default=0.15,
                      help="Fraction of dataset to use (0.0 to 1.0)")
    parser.add_argument("--use_fedbn", action="store_true",
                      help="Use FedBN strategy for multi-modal learning")
    args = parser.parse_args()

    print(f"""
🏥 Multi-Modal Medical Federated Learning Simulation
{'='*60}
Configuration:
  • Clients: {args.clients}
  • Rounds: {args.rounds}
  • Sample Fraction: {args.sample_fraction}
  • Strategy: {'FedBN' if args.use_fedbn else 'FedAvg'}
  • Modalities: Dermoscopy & Chest Radiography
{'='*60}
    """)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Quick dataset status check first
    datasets_dir = args.data_dir
    os.makedirs(datasets_dir, exist_ok=True)
    
    dataset_status = quick_dataset_status_check(datasets_dir)
    
    # Download and organize multi-modal datasets (will skip existing ones)
    print(f"\n📥 Downloading/Organizing datasets...")
    dataset_paths = download_multimodal_datasets(datasets_dir)
    
    if len(dataset_paths) == 0:
        raise FileNotFoundError("❌ No valid datasets found!")
    
    # Validate datasets before proceeding with federated learning
    if not validate_dataset_for_federated_learning(dataset_paths):
        raise ValueError("❌ Dataset validation failed! Please fix the issues before running federated learning.")
    
    print(f"\n✅ Using {len(dataset_paths)} multi-modal datasets")
    for i, path in enumerate(dataset_paths):
        modality = os.path.basename(path)
        print(f"   {i+1}. {modality}")

    # Prepare client datasets - each client specializes in a different modality
    client_datasets = []
    modality_names = []
    
    for i, path in enumerate(dataset_paths[:args.clients]):
        modality_name = os.path.basename(path)
        modality_names.append(modality_name)
        
        print(f"\n🔬 Loading {modality_name} dataset for Client {i+1}...")
        dataset_splits = load_multimodal_dataset(path, 1, args.sample_fraction)
        if dataset_splits and len(dataset_splits) > 0:
            client_datasets.append(dataset_splits[0])
    
    # If we have more clients than datasets, duplicate the last dataset
    while len(client_datasets) < args.clients:
        print(f"\n🔄 Duplicating last dataset for additional clients...")
        client_datasets.append(client_datasets[-1])
        modality_names.append(modality_names[-1] + "_copy")
    
    print(f"\n👥 Prepared {len(client_datasets)} client datasets for federated learning")

    # Train federated models
    client_models = []
    training_history = {}
    
    for i in range(args.clients):
        model = MultiModalMedicalCNN(num_classes=2).to(device)
        train_ds, test_ds = client_datasets[i]
        
        # Check if dataset is empty
        if len(train_ds) == 0:
            print(f"⚠️  Warning: Client {i+1} has empty training dataset. Skipping.")
            continue
            
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        
        # Training loop with enhanced monitoring
        client_name = f"Client_{i+1}_{modality_names[i]}"
        print(f"\n🚀 Training {client_name}")
        print(f"   📊 Dataset: {len(train_ds)} train, {len(test_ds)} test samples")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        
        # Track comprehensive training metrics
        training_losses = []
        training_accuracies = []
        
        model.train()
        for epoch in range(args.rounds):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
                
                # Calculate training accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_accuracy = 100 * correct / total
            
            training_losses.append(avg_loss)
            training_accuracies.append(train_accuracy)
            
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"   Round {epoch+1:2d}: Loss={avg_loss:.4f}, Acc={train_accuracy:.1f}%, LR={current_lr:.6f}")
        
        # Save training history
        training_history[client_name] = {
            'losses': training_losses,
            'accuracies': training_accuracies,
            'modality': modality_names[i]
        }
        
        client_models.append(model)
    
    # Model aggregation
    if len(client_models) == 0:
        print("❌ No models were trained successfully. Cannot proceed.")
        return
        
    global_model = MultiModalMedicalCNN(num_classes=2).to(device)
    
    if args.use_fedbn:
        print(f"\n🔗 Aggregating {len(client_models)} models using FedBN (preserving domain-specific BN)...")
        # Implement FedBN aggregation (simplified version)
        aggregate_with_fedbn(global_model, client_models)
    else:
        print(f"\n🔗 Aggregating {len(client_models)} models using FedAvg...")
        aggregate_with_fedavg(global_model, client_models)
    
    # Comprehensive evaluation
    print(f"\n📊 Evaluating global model on each client's test set...")
    evaluate_global_model(global_model, client_datasets, modality_names, device)
    
    # Create comprehensive visualizations
    create_multimodal_visualizations(training_history, args)
    
    print(f"\n🎉 Multi-modal federated learning completed!")
    print(f"📁 Results saved in 'multimodal_results' directory")

def aggregate_with_fedavg(global_model, client_models):
    """Standard FedAvg aggregation."""
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        params = [client_models[i].state_dict()[key].float() for i in range(len(client_models))]
        global_dict[key] = torch.mean(torch.stack(params), dim=0).to(global_dict[key].dtype)
    
    global_model.load_state_dict(global_dict)

def aggregate_with_fedbn(global_model, client_models):
    """FedBN aggregation preserving batch normalization layers."""
    global_dict = global_model.state_dict()
    
    # Identify BN layers
    bn_keys = [key for key in global_dict.keys() if 'bn' in key or 'norm' in key]
    
    for key in global_dict.keys():
        if key in bn_keys:
            # Keep BN parameters from first client
            global_dict[key] = client_models[0].state_dict()[key]
        else:
            # Average other parameters
            params = [client_models[i].state_dict()[key].float() for i in range(len(client_models))]
            global_dict[key] = torch.mean(torch.stack(params), dim=0).to(global_dict[key].dtype)
    
    global_model.load_state_dict(global_dict)
    print(f"   🔒 Preserved {len(bn_keys)} domain-specific BN layers")

def evaluate_global_model(global_model, client_datasets, modality_names, device):
    """Comprehensive evaluation of the global model."""
    test_results = []
    
    results_dir = "multimodal_results"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "test_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Client", "Modality", "Accuracy", "Precision", "Recall", "F1_Score"])
        
        for i, (modality_name, (_, test_ds)) in enumerate(zip(modality_names, client_datasets)):
            if len(test_ds) == 0:
                print(f"   ⚠️  Client {i+1} ({modality_name}) has no test data")
                continue
                
            test_loader = DataLoader(test_ds, batch_size=16)
            
            correct = 0
            total = 0
            predictions = []
            true_labels = []
            
            global_model.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = global_model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    
                    predictions.extend(predicted.cpu().numpy())
                    true_labels.extend(target.cpu().numpy())
            
            accuracy = 100 * correct / total if total > 0 else 0
            
            # Calculate detailed metrics
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score
                precision = precision_score(true_labels, predictions, average='binary', zero_division=0)
                recall = recall_score(true_labels, predictions, average='binary', zero_division=0)
                f1 = f1_score(true_labels, predictions, average='binary', zero_division=0)
            except:
                precision = recall = f1 = 0.0
            
            test_results.append({
                'client': f"Client {i+1}",
                'modality': modality_name,
                'accuracy': accuracy,
                'precision': precision * 100,
                'recall': recall * 100,
                'f1_score': f1 * 100
            })
            
            print(f"   🎯 Client {i+1} ({modality_name}): Acc={accuracy:.1f}%, F1={f1*100:.1f}%")
            writer.writerow([f"Client {i+1}", modality_name, f"{accuracy:.2f}", 
                           f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"])
    
    return test_results

def create_multimodal_visualizations(training_history, args):
    """Create comprehensive visualizations for multi-modal results."""
    
    results_dir = "multimodal_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create comprehensive dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Multi-Modal Medical Federated Learning Results\n'
                f'Strategy: {"FedBN" if args.use_fedbn else "FedAvg"} | '
                f'Clients: {args.clients} | Rounds: {args.rounds}', 
                fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss Progression
    ax1 = axes[0, 0]
    for i, (client_name, history) in enumerate(training_history.items()):
        rounds = range(1, len(history['losses']) + 1)
        ax1.plot(rounds, history['losses'], marker='o', linewidth=2, 
                label=f"{client_name}", color=colors[i % len(colors)])
    
    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss by Modality')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Accuracy Progression
    ax2 = axes[0, 1]
    for i, (client_name, history) in enumerate(training_history.items()):
        rounds = range(1, len(history['accuracies']) + 1)
        ax2.plot(rounds, history['accuracies'], marker='s', linewidth=2,
                label=f"{client_name}", color=colors[i % len(colors)])
    
    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Training Accuracy (%)')
    ax2.set_title('Training Accuracy by Modality')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Plot 3: Modality Comparison
    ax3 = axes[1, 0]
    modalities = [history['modality'] for history in training_history.values()]
    final_accuracies = [history['accuracies'][-1] for history in training_history.values()]
    
    bars = ax3.bar(range(len(modalities)), final_accuracies, color=colors[:len(modalities)])
    ax3.set_xlabel('Medical Imaging Modality')
    ax3.set_ylabel('Final Training Accuracy (%)')
    ax3.set_title('Performance by Medical Modality')
    ax3.set_xticks(range(len(modalities)))
    ax3.set_xticklabels([mod.replace('_', '\n') for mod in modalities], rotation=0)
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_ylim(0, max(final_accuracies) + 10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Convergence Analysis
    ax4 = axes[1, 1]
    for i, (client_name, history) in enumerate(training_history.items()):
        losses = history['losses']
        # Calculate convergence rate (loss reduction)
        if len(losses) > 1:
            convergence_rate = [(losses[0] - loss) / losses[0] * 100 for loss in losses]
            rounds = range(1, len(convergence_rate) + 1)
            ax4.plot(rounds, convergence_rate, marker='d', linewidth=2,
                    label=f"{client_name}", color=colors[i % len(colors)])
    
    ax4.set_xlabel('Training Round')
    ax4.set_ylabel('Loss Reduction (%)')
    ax4.set_title('Convergence Rate Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'multimodal_training_dashboard.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comprehensive visualizations saved to: {results_dir}/")

if __name__ == "__main__":
    main()
