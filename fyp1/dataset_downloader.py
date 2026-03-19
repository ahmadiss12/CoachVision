"""
Step 1: Download and explore the Kaggle squat dataset.
This script handles dataset download, exploration, and analysis.
"""

import os
import pandas as pd
from pathlib import Path

def download_kaggle_dataset():
    """Download the squat dataset from Kaggle."""
    print("\n" + "="*60)
    print("STEP 1: DOWNLOAD KAGGLE DATASET")
    print("="*60)
    
    print("\n1. Install Kaggle CLI (if not already installed):")
    print("   pip install kaggle")
    
    print("\n2. Set up Kaggle API credentials:")
    print("   - Go to https://www.kaggle.com/settings/account")
    print("   - Click 'Create New API Token'")
    print("   - This downloads kaggle.json")
    print("   - Move it to: %USERPROFILE%\\.kaggle\\kaggle.json")
    
    print("\n3. Download the dataset:")
    dataset_name = "thashmiladewmini/squat-exercise-pose-dataset"
    download_dir = "squat_dataset"
    
    # Check if dataset already exists
    if os.path.exists(download_dir):
        print(f"\n✓ Dataset directory '{download_dir}' already exists")
        return download_dir
    
    print(f"\nDownloading from Kaggle...")
    os.system(f"kaggle datasets download -d {dataset_name} -p {download_dir}")
    
    # Extract if zip
    if os.path.exists(os.path.join(download_dir)):
        print(f"✓ Dataset downloaded to '{download_dir}'")
        
        # Try to extract zip files
        import zipfile
        for file in os.listdir(download_dir):
            if file.endswith('.zip'):
                with zipfile.ZipFile(os.path.join(download_dir, file), 'r') as zip_ref:
                    zip_ref.extractall(download_dir)
                print(f"✓ Extracted {file}")
    
    return download_dir


def explore_dataset(dataset_dir="squat_dataset"):
    """Explore and visualize the dataset."""
    print("\n" + "="*60)
    print("STEP 1B: EXPLORE DATASET")
    print("="*60)
    
    # Find CSV files
    csv_files = list(Path(dataset_dir).rglob("*.csv"))
    
    if not csv_files:
        print(f"\n❌ No CSV files found in {dataset_dir}")
        print("Check the dataset directory structure")
        return None
    
    print(f"\n✓ Found {len(csv_files)} CSV file(s):")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    # Load and analyze the main dataset
    df = pd.read_csv(csv_files[0])
    
    print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print(f"\n📋 Column Names:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. {col}")
    
    print(f"\n🏷️ Class Distribution:")
    if 'label' in df.columns:
        label_map = {
            0: "Correct squat",
            1: "Shallow squat",
            2: "Forward lean",
            3: "Knees caving in",
            4: "Heels off ground",
            5: "Asymmetric squat"
        }
        print(df['label'].value_counts().sort_index())
        for idx, name in label_map.items():
            count = (df['label'] == idx).sum()
            print(f"  {idx}: {name} = {count} samples")
    
    print(f"\n📈 Feature Statistics:")
    print(df.describe())
    
    print(f"\n✓ Dataset ready for training!")
    return str(csv_files[0])


if __name__ == "__main__":
    # Step 1: Download
    dataset_dir = download_kaggle_dataset()
    
    # Step 1B: Explore
    csv_path = explore_dataset(dataset_dir)
    
    if csv_path:
        print("\n" + "="*60)
        print("✓ STEP 1 COMPLETE - Dataset downloaded and explored")
        print(f"CSV file location: {csv_path}")
        print("="*60)
        print("\nNext: Run 'python train_squat_classifier.py'")
