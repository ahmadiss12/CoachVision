"""
QUICK START: Fine-Tune MediaPipe for Squat Detection

Run these commands in order, one at a time.
Each step takes 5-15 minutes.
"""

# ============================================================
# STEP 1: INSTALL DEPENDENCIES (5 min)
# ============================================================

# Open PowerShell in your project directory
cd e:\ua\fyp\template\fyp1

# Install required packages
pip install kaggle pandas scikit-learn matplotlib numpy

# Check if installed
pip list | Select-String "kaggle|pandas|scikit"


# ============================================================
# STEP 2: SETUP KAGGLE API (2 min)
# ============================================================

# Go to https://www.kaggle.com/settings/account
# Click "Create New API Token" to download kaggle.json
# 
# On Windows, move the file to:
# C:\Users\YourUsername\.kaggle\kaggle.json
#
# (Replace YourUsername with your actual Windows username)


# ============================================================
# STEP 3: DOWNLOAD DATASET (3 min)
# ============================================================

python dataset_downloader.py

# You should see:
# ✓ Dataset downloaded to 'squat_dataset'
# ✓ Found X CSV files
# 📊 Dataset Shape: XXXX rows X XX columns
# 🏷️ Class Distribution: (shows distribution of forms)


# ============================================================
# STEP 4: EXPLORE THE DATA (2 min)
# ============================================================

# The previous script already explored it, but you can manually check:
python -c "import pandas as pd; df = pd.read_csv('squat_dataset/squat_data.csv'); print(df.head()); print(df.columns.tolist())"


# ============================================================
# STEP 5: TRAIN THE CLASSIFIER (10 min)
# ============================================================

python train_squat_classifier.py

# You should see:
# 🚀 Training RandomForest classifier...
# ✓ Training complete!
# 📋 Evaluating model...
# ✓ Overall Accuracy: 92-95%
# 💾 Model saved to: models/squat_classifier.pkl


# ============================================================
# STEP 6: VERIFY MODEL WAS SAVED (1 min)
# ============================================================

# Check if model files exist
Get-ChildItem models/

# You should see:
# squat_classifier.pkl
# scaler.pkl


# ============================================================
# STEP 7: TEST THE CLASSIFIER (5 min)
# ============================================================

python -c "from backend.squat_form_analyzer import SquatFormAnalyzer; a = SquatFormAnalyzer(); print('Classifier loaded successfully!')"

# Should print: Classifier loaded successfully!


# ============================================================
# STEP 8: INTEGRATE INTO MAIN SYSTEM (15 min)
# ============================================================

# Edit main.py and add these lines:
#
# At the top:
# from backend.squat_form_analyzer import SquatFormAnalyzer
#
# In __init__:
# self.form_analyzer = SquatFormAnalyzer()
#
# In the main loop:
# if landmarks:
#     analysis = self.form_analyzer.analyze(landmarks)
#     if analysis['confidence'] > 0.7:
#         print(f"[FORM] {analysis['form_name']}: {analysis['feedback']}")


# ============================================================
# STEP 9: TEST ON YOUR VIDEOS (5 min)
# ============================================================

time python main.py --exercise squat --difficulty beginner --video 1.mp4

# You should see form analysis in the output


# ============================================================
# TROUBLESHOOTING
# ============================================================

# Issue: "ModuleNotFoundError: No module named 'kaggle'"
# Solution:
pip install --upgrade kaggle

# Issue: "Kaggle API error or credentials not found"
# Solution:
# 1. Make sure kaggle.json is saved correctly
# 2. Run: kaggle datasets download -d thashmiladewmini/squat-exercise-pose-dataset
# 3. If it works, you're good to go

# Issue: "Model accuracy is too low (< 85%)"
# Solution:
# 1. Your squat form might be different from training data
# 2. Try collecting more diverse squat videos
# 3. Test with different camera angles
# 4. Check that MediaPipe is detecting landmarks correctly

# Issue: "ImportError in squat_form_analyzer.py"
# Solution:
pip install -r requirements.txt
# Or manually:
pip install numpy scikit-learn pickle5


# ============================================================
# ADVANCED: IMPROVE ACCURACY
# ============================================================

# If accuracy < 90%, try:

# A) Use a better model (Gradient Boosting)
# Edit train_squat_classifier.py, line ~80
# Change from RandomForestClassifier to:
from sklearn.ensemble import GradientBoostingClassifier
# Then rerun: python train_squat_classifier.py

# B) Collect more training data
# Use your own squat videos, extract landmarks, label them
# Add them to squat_dataset folder

# C) Engineer better features
# Edit backend/squat_form_analyzer.py extract_squat_features()
# Add more specific features for squat detection


# ============================================================
# VERIFICATION CHECKLIST
# ============================================================

# [ ] Kaggle API working (test with: kaggle --version)
# [ ] Dataset downloaded to squat_dataset/
# [ ] CSV file contains expected columns
# [ ] Classifier trained with > 85% accuracy
# [ ] Model saved to models/squat_classifier.pkl
# [ ] Scaler saved to models/scaler.pkl
# [ ] SquatFormAnalyzer loads without errors
# [ ] main.py integrated with analyzer
# [ ] Test video runs and shows form analysis

# If all checked, you're done! 🎉


# ============================================================
# PERFORMANCE MONITORING
# ============================================================

# After integration, check:
# - Does form detection match your observations?
# - Is confidence > 80% for most frames?
# - Does the feedback match the actual form?
# - Is inference fast enough (< 50ms per frame)?

# If not, tune the parameters in train_squat_classifier.py
