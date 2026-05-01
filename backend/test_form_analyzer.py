"""
Quick test to verify form analyzer integration works
"""

import numpy as np
from backend.squat_form_analyzer import SquatFormAnalyzer

print("[TEST] Loading form analyzer...")
analyzer = SquatFormAnalyzer()

if analyzer.model is None:
    print("[ERROR] Model failed to load!")
    exit(1)

print("[OK] Model loaded successfully!")
print(f"[OK] Scaler: {analyzer.scaler is not None}")

# Test with dummy landmarks
print("\n[TEST] Testing with dummy landmarks...")
dummy_landmarks = np.random.randn(33, 3) * 0.2 + 0.5
analysis = analyzer.analyze(dummy_landmarks)

print(f"[OK] Form analysis result:")
print(f"  - Form: {analysis['form_name']}")
print(f"  - Confidence: {analysis['confidence']:.2%}")
print(f"  - Feedback: {analysis['feedback']}")

print("\n[SUCCESS] Form analyzer working correctly!")
