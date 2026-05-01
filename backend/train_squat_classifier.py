"""
Step 2: Train a squat form classifier using the dataset.
This classifier will detect 6 squat forms: correct, shallow, forward_lean, 
knees_caving, heels_off, asymmetric.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


class SquatFormClassifier:
    """Train and save a squat form classifier."""
    
    # Label mapping
    LABEL_MAP = {
        0: "Correct",
        1: "Shallow", 
        2: "Forward Lean",
        3: "Knees Caving",
        4: "Heels Off",
        5: "Asymmetric"
    }
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None
        
    def load_data(self, csv_path):
        """Load dataset from CSV."""
        print(f"\n[FILE] Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"[OK] Loaded {len(df)} samples")
        print(f"[OK] Features: {len(df.columns) - 2} columns")  # Exclude label and filename
        
        return df
    
    def prepare_data(self, df):
        """Prepare features and labels."""
        print("\n[PREP] Preparing data...")
        
        # Print actual columns in dataset
        print(f"\n  Columns in dataset: {list(df.columns)}")
        
        # Select only numeric columns (exclude label and any string columns)
        exclude_cols = ['label', 'video_filename', 'frame_number', 'filename']
        
        # Select columns that are numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"\n  Numeric columns found: {len(numeric_cols)}")
        print(f"  Excluded: {exclude_cols}")
        
        if not self.feature_cols:
            print("ERROR: No numeric columns found!")
            print(f"All columns: {df.columns.tolist()}")
            raise ValueError("No features to train on")
        
        X = df[self.feature_cols].values
        y = df['label'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        print(f"[OK] Features selected: {len(self.feature_cols)}")
        print(f"  Selected features: {self.feature_cols[:5]}...")  # Show first 5
        print(f"  - Joint angles: knee, hip, ankle")
        print(f"  - Body angles: spine, torso lean")
        print(f"  - Metrics: hip depth, symmetry, lateral deviation")
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n[DATA] Data split:")
        print(f"  - Training: {len(X_train)} samples (80%)")
        print(f"  - Testing: {len(X_test)} samples (20%)")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """Train RandomForest classifier."""
        print("\n[TRAIN] Training RandomForest classifier...")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        print("[OK] Training complete!")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set."""
        print("\n[EVAL] Evaluating model...")
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n[OK] Overall Accuracy: {accuracy:.2%}")
        
        print(f"\n[REPORT] Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=[self.LABEL_MAP[i] for i in range(6)],
            zero_division=0
        ))
        
        # Feature importance
        importances = self.model.feature_importances_
        top_features_idx = np.argsort(importances)[-10:]
        
        print(f"\n[TOP] Top 10 Most Important Features:")
        for i, idx in enumerate(reversed(top_features_idx), 1):
            print(f"  {i}. {self.feature_cols[idx]}: {importances[idx]:.4f}")
        
        return accuracy
    
    def save(self, model_path="models/squat_classifier.pkl", scaler_path="models/scaler.pkl"):
        """Save trained model and scaler."""
        os.makedirs("models", exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"\n[SAVE] Model saved to: {model_path}")
        print(f"[SAVE] Scaler saved to: {scaler_path}")
    
    def predict(self, features):
        """Predict squat form from features."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        confidence = self.model.predict_proba(features_scaled)[0].max()
        
        return prediction, self.LABEL_MAP[prediction], confidence


def main():
    print("\n" + "="*60)
    print("STEP 2: TRAIN SQUAT FORM CLASSIFIER")
    print("="*60)
    
    # Find CSV file
    csv_files = list(Path("squat_dataset").rglob("*.csv"))
    if not csv_files:
        print("[ERROR] Dataset not found. Run 'python dataset_downloader.py' first")
        return
    
    csv_path = csv_files[0]
    
    # Initialize classifier
    classifier = SquatFormClassifier()
    
    # Load and prepare data
    df = classifier.load_data(csv_path)
    X_train, X_test, y_train, y_test = classifier.prepare_data(df)
    
    # Train
    classifier.train(X_train, y_train)
    
    # Evaluate
    accuracy = classifier.evaluate(X_test, y_test)
    
    # Save
    classifier.save()
    
    print("\n" + "="*60)
    print(f"[OK] STEP 2 COMPLETE - Model trained with {accuracy:.2%} accuracy")
    print("="*60)
    print("\nNext: Integrate classifier into your main.py")


if __name__ == "__main__":
    main()
