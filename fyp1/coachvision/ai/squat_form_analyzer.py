"""
Step 3: Integrate the trained classifier into the main system.
This module provides enhanced squat form detection.
"""

import pickle
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path


class SquatFormAnalyzer:
    """
    Analyzes squat form using pre-trained classifier.
    Combines MediaPipe pose detection with machine learning.
    """
    
    FORM_FEEDBACK = {
        0: "[OK] Perfect form! Keep going.",
        1: "[WARN] Go deeper - shallow squat",
        2: "[WARN] Lean forward - stack your torso",
        3: "[WARN] Knees caving - push them outward",
        4: "[WARN] Heels lifting - keep feet flat",
        5: "[WARN] Asymmetric - balance both sides"
    }
    
    def __init__(self, model_path="models/squat_classifier.pkl", 
                 scaler_path="models/scaler.pkl"):
        """
        Initialize the form analyzer.
        
        Args:
            model_path: Path to trained RandomForest model
            scaler_path: Path to StandardScaler
        """
        self.model = None
        self.scaler = None
        self.load_model(model_path, scaler_path)
    
    def load_model(self, model_path, scaler_path):
        """Load pre-trained model and scaler."""
        if not Path(model_path).exists():
            print(f"[WARNING] Model not found at {model_path}")
            print("Run: python train_squat_classifier.py")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print(f"[OK] Loaded classifier from {model_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Loading model: {e}")
            return False
    
    def extract_squat_features(self, landmarks: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract squat-specific features from MediaPipe landmarks.
        Must match the 13 features used in training:
        left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle,
        left_ankle_angle, right_ankle_angle, spine_angle, torso_lean,
        left_knee_lateral, right_knee_lateral, symmetry_score, hip_depth
        
        Args:
            landmarks: MediaPipe pose landmarks (33 x 3)
            
        Returns:
            Feature vector for classifier
        """
        try:
            # Landmark indices
            NOSE = 0
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_HIP = 23
            RIGHT_HIP = 24
            LEFT_KNEE = 25
            RIGHT_KNEE = 26
            LEFT_ANKLE = 27
            RIGHT_ANKLE = 28
            LEFT_FOOT_INDEX = 31
            RIGHT_FOOT_INDEX = 32
            
            # Extract (x, y) coordinates
            def get_xy(idx):
                return landmarks[idx][0], landmarks[idx][1]
            
            # Calculate angles
            def angle_between_points(p1, p2, p3):
                """Calculate angle at p2 formed by p1-p2-p3."""
                v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
                v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                return angle
            
            # Key angles (matching training data)
            left_knee_angle = angle_between_points(
                get_xy(LEFT_HIP), get_xy(LEFT_KNEE), get_xy(LEFT_ANKLE)
            )
            right_knee_angle = angle_between_points(
                get_xy(RIGHT_HIP), get_xy(RIGHT_KNEE), get_xy(RIGHT_ANKLE)
            )
            
            left_hip_angle = angle_between_points(
                get_xy(LEFT_SHOULDER), get_xy(LEFT_HIP), get_xy(LEFT_KNEE)
            )
            right_hip_angle = angle_between_points(
                get_xy(RIGHT_SHOULDER), get_xy(RIGHT_HIP), get_xy(RIGHT_KNEE)
            )
            
            left_ankle_angle = angle_between_points(
                get_xy(LEFT_KNEE), get_xy(LEFT_ANKLE), get_xy(LEFT_FOOT_INDEX)
            )
            right_ankle_angle = angle_between_points(
                get_xy(RIGHT_KNEE), get_xy(RIGHT_ANKLE), get_xy(RIGHT_FOOT_INDEX)
            )
            
            # Spine angle (straight line from shoulder to hip)
            spine_angle = angle_between_points(
                get_xy(LEFT_SHOULDER), get_xy(LEFT_HIP), get_xy(RIGHT_HIP)
            )
            
            # Torso lean (angle from vertical)
            torso_lean = abs(landmarks[LEFT_SHOULDER][0] - landmarks[LEFT_HIP][0])
            
            # Lateral deviations (knee alignment with hips)
            left_knee_lateral = abs(landmarks[LEFT_KNEE][0] - landmarks[LEFT_HIP][0])
            right_knee_lateral = abs(landmarks[RIGHT_KNEE][0] - landmarks[RIGHT_HIP][0])
            
            # Symmetry score (left-right difference)
            symmetry_score = abs(left_knee_angle - right_knee_angle)
            
            # Hip depth (vertical distance from shoulder to hip)
            hip_depth = landmarks[LEFT_HIP][1] - landmarks[LEFT_SHOULDER][1]
            
            # Heels off ground (average heel height difference)
            left_heel_distance = abs(landmarks[LEFT_ANKLE][1] - landmarks[LEFT_FOOT_INDEX][1])
            right_heel_distance = abs(landmarks[RIGHT_ANKLE][1] - landmarks[RIGHT_FOOT_INDEX][1])
            avg_heel_distance = (left_heel_distance + right_heel_distance) / 2
            
            # Create feature vector - exactly 13 features matching training data
            features = np.array([
                left_knee_angle, right_knee_angle,
                left_hip_angle, right_hip_angle,
                left_ankle_angle, right_ankle_angle,
                spine_angle, torso_lean,
                left_knee_lateral, right_knee_lateral,
                symmetry_score, hip_depth,
                avg_heel_distance
            ])
            
            return features
        
        except Exception as e:
            print(f"[ERROR] Extracting features: {e}")
            return None
    
    def analyze(self, landmarks: np.ndarray) -> Dict:
        """
        Analyze squat form from MediaPipe landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks (33 x 3)
            
        Returns:
            Dictionary with form classification and feedback
        """
        if self.model is None:
            return {"form": "unknown", "feedback": "Classifier not loaded"}
        
        # Extract features
        features = self.extract_squat_features(landmarks)
        if features is None:
            return {"form": "error", "feedback": "Could not extract features"}
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        form_id = self.model.predict(features_scaled)[0]
        confidence = self.model.predict_proba(features_scaled)[0].max()
        
        form_names = ["Correct", "Shallow", "Forward Lean", 
                     "Knees Caving", "Heels Off", "Asymmetric"]
        
        return {
            "form_id": int(form_id),
            "form_name": form_names[form_id],
            "feedback": self.FORM_FEEDBACK[form_id],
            "confidence": float(confidence),
            "features": features.tolist()
        }


# Example usage
def test_analyzer():
    """Test the analyzer with sample landmarks."""
    analyzer = SquatFormAnalyzer()
    
    if analyzer.model is None:
        print("Analysis: Train the model first using train_squat_classifier.py")
        return
    
    # Simulate MediaPipe landmarks (33 joints x 3 coordinates)
    sample_landmarks = np.random.randn(33, 3) * 0.1 + 0.5
    
    result = analyzer.analyze(sample_landmarks)
    print(f"\n📊 Analysis Result:")
    print(f"  Form: {result['form_name']}")
    print(f"  Feedback: {result['feedback']}")
    print(f"  Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    test_analyzer()
