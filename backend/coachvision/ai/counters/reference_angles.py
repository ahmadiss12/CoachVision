"""
Reference-based angle targets for exercise counters.

These values are aligned with commonly cited clinical and coaching literature used
by the American College of Sports Medicine (ACSM) and related sources. They are
intended for real-time pose estimation (2D landmarks); measured angles may differ
slightly from laboratory goniometry or 3D motion capture.

---------------------------------------------------------------------------
Primary sources (for thesis / documentation citations)
---------------------------------------------------------------------------

1. American College of Sports Medicine (ACSM). *ACSM's Guidelines for Exercise
   Testing and Prescription* (current edition). Wolters Kluwer.
   - Resistance training: use a full, pain-free range of motion; technique
     targets such as thigh parallel to the floor for squats are widely used in
     general-population programming.

2. Escamilla, R. F., et al. (2001). "A three-dimensional biomechanical analysis
   of sumo and conventional style deadlifts." *Medicine & Science in Sports &
   Exercise* — companion literature on squat/deadlift knee and hip loading;
   dynamic squat knee flexion patterns are commonly reported in the 0–90°+ range.

3. Escamilla, R. F., et al. (2001). "Knee biomechanics of the dynamic squat
   exercise." *Medicine & Science in Sports & Exercise*. — knee flexion and
   loading across squat depths; supports using defined depth categories (e.g.
   parallel ~90° knee flexion between thigh and shank in sagittal analysis).

4. For push-up elbow angle near 90° at the bottom, standard exercise-science
   and ACSM-aligned instructional materials describe the elbows bending to
   ~90° for a "classic" push-up depth (verify against your landmark definition:
   shoulder-elbow-wrist interior angle).

---------------------------------------------------------------------------
Angle convention (this codebase)
---------------------------------------------------------------------------

- **Squat**: Interior angle at the knee (hip–knee–ankle). ~180° ≈ standing;
  smaller values = deeper flexion. "Parallel" squat is often operationalized as
  ~90° (thigh roughly parallel to the floor) in instructional literature.

- **Push-up**: Interior angle at the elbow (shoulder–elbow–wrist).

- **Deadlift**: Composite / hip hinge angles as defined in `deadlift.py`
  (see that module for landmark triplets).

- **Lunge, bicep curl, shoulder press, sit-up, high knees, mountain climber,
  jumping jack**: Defaults are centralized below; many are pose-engineering
  targets with ACSM-aligned coaching cues (e.g. ~90° knee flexion wall sit).

---------------------------------------------------------------------------
Preset flexion targets (knee, squat) — used by CONFIG_PRESETS
---------------------------------------------------------------------------
"""

from typing import Dict

# --- Squat: knee interior angle at bottom of rep (smaller = deeper) ---

# Parallel squat: widely cited in ACSM-aligned coaching (~90° knee flexion).
SQUAT_KNEE_PARALLEL_DEG: float = 90.0

# Partial-depth squat for novices / mobility limits: above parallel but still
# controlled range (common in beginner programming).
SQUAT_KNEE_PARTIAL_DEPTH_DEG: float = 110.0

# Deep squat threshold (below parallel): use with caution; individual anatomy varies.
SQUAT_KNEE_DEEP_DEG: float = 80.0

# Feedback: "too shallow" band upper bound (above this, depth is inadequate vs parallel target).
SQUAT_KNEE_SHALLOW_WARNING_DEG: float = 100.0

# Feedback: "too deep" for conservative programming (very deep flexion).
SQUAT_KNEE_DEEP_WARNING_DEG: float = 80.0

# Standing / top of rep: pose-estimation systems often report <180°; tune for your camera.
SQUAT_KNEE_EXTENSION_STANDING_POSE_DEG: float = 155.0

# Hysteresis buffer (not from ACSM — engineering choice for noisy pose data).
SQUAT_HYSTERESIS_BUFFER_DEG: float = 10.0

# Presets by difficulty (flexion_threshold = max knee angle at bottom to count as "deep enough")
SQUAT_PRESET_FLEXION: Dict[str, float] = {
    "beginner": SQUAT_KNEE_PARTIAL_DEPTH_DEG,
    "intermediate": SQUAT_KNEE_PARALLEL_DEG,
    "advanced": SQUAT_KNEE_DEEP_DEG,
}

SQUAT_PRESET_BUFFER: Dict[str, float] = {
    "beginner": 10.0,
    "intermediate": 10.0,
    "advanced": 5.0,
}

# --- Push-up: elbow interior angle (smaller = more bent) ---

# ACSM-aligned instructional cue: ~90° elbow flexion at bottom for standard push-up.
PUSHUP_ELBOW_PARALLEL_TARGET_DEG: float = 90.0

PUSHUP_PRESET_FLEXION: Dict[str, float] = {
    "beginner": 100.0,
    "intermediate": PUSHUP_ELBOW_PARALLEL_TARGET_DEG,
    "advanced": 80.0,
}

PUSHUP_PRESET_BUFFER: Dict[str, float] = {
    "beginner": 20.0,
    "intermediate": 15.0,
    "advanced": 10.0,
}

# --- Lunge: knee flexion at front leg (similar depth logic to squat) ---

LUNGE_PRESET_FLEXION: Dict[str, float] = {
    "beginner": 90.0,
    "intermediate": 80.0,
    "advanced": 70.0,
}

LUNGE_PRESET_BUFFER: Dict[str, float] = {
    "beginner": 15.0,
    "intermediate": 10.0,
    "advanced": 5.0,
}

# --- Deadlift: hip hinge (see DeadliftConfig; values are pose-tuned) ---

# Literature reports wide hip/knee angles at bottom; these presets follow
# conservative hinge depth for counting reps (not a substitute for barbell coaching).
DEADLIFT_PRESET_FLEXION: Dict[str, float] = {
    "beginner": 130.0,
    "intermediate": 120.0,
    "advanced": 110.0,
}

DEADLIFT_PRESET_BUFFER: Dict[str, float] = {
    "beginner": 10.0,
    "intermediate": 5.0,
    "advanced": 5.0,
}

# --- Lunge: front knee (same convention as squat; smaller = deeper) ---

LUNGE_EXTENSION_STANDING_DEG: float = 160.0
LUNGE_FLEXION_BOTTOM_INTERMEDIATE_DEG: float = LUNGE_PRESET_FLEXION["intermediate"]
LUNGE_DEFAULT_BUFFER_DEG: float = LUNGE_PRESET_BUFFER["intermediate"]
LUNGE_TOO_SHALLOW_DEG: float = 100.0
LUNGE_TOO_DEEP_DEG: float = 70.0
LUNGE_REAR_KNEE_TARGET_DEG: float = 90.0
LUNGE_REAR_KNEE_TOLERANCE_DEG: float = 20.0

# --- Bicep curl: elbow flexion (smaller = more curled at top) ---

BICEP_ELBOW_EXTENDED_DEG: float = 160.0
BICEP_ELBOW_FULL_CURL_DEG: float = 45.0
BICEP_BUFFER_DEG: float = 15.0

# --- Shoulder press: elbow (extended overhead vs start at shoulders) ---

SHOULDER_PRESS_ELBOW_OVERHEAD_DEG: float = 170.0
SHOULDER_PRESS_ELBOW_START_DEG: float = 90.0
SHOULDER_PRESS_BUFFER_DEG: float = 15.0

# --- Sit-up: torso vs horizontal (0° flat, ~90° upright) — ACSM-aligned ROM in sagittal plane ---

SITUP_TORSO_UP_DEG: float = 80.0
SITUP_TORSO_DOWN_DEG: float = 20.0
SITUP_BUFFER_DEG: float = 10.0
SITUP_KNEE_TARGET_DEG: float = 90.0
SITUP_KNEE_TOLERANCE_DEG: float = 20.0

# --- High knees: hip flexion / extension ---

HIGH_KNEES_HIP_FLEXION_DEG: float = 90.0
HIGH_KNEES_HIP_EXTENSION_DEG: float = 170.0
HIGH_KNEES_BUFFER_DEG: float = 10.0

# --- Mountain climber: plank line + knee drive ---

MOUNTAIN_CLIMBER_PLANK_MIN_DEG: float = 160.0
MOUNTAIN_CLIMBER_KNEE_DRIVE_DEG: float = 90.0
MOUNTAIN_CLIMBER_KNEE_EXTENDED_DEG: float = 160.0
MOUNTAIN_CLIMBER_BUFFER_DEG: float = 15.0

# --- Jumping jack: mixed degrees + normalized positions (pose-tuned) ---

JUMPING_JACK_ARM_STRAIGHT_DEG: float = 160.0
JUMPING_JACK_ARM_ABDUCTION_DEG: float = 45.0
JUMPING_JACK_LEG_STRAIGHT_DEG: float = 170.0
# Normalized screen-space thresholds (engineering, not ACSM)
JUMPING_JACK_FEET_APART_NORM: float = 0.15
JUMPING_JACK_FEET_TOGETHER_NORM: float = 0.05
JUMPING_JACK_ARM_OVERHEAD_NORM: float = 0.1
JUMPING_JACK_POSITION_BUFFER_NORM: float = 0.1
JUMPING_JACK_COORDINATION_NORM: float = 0.2

# --- Plank: shoulder–hip–ankle line (neutral spine) ---

PLANK_BODY_LINE_MIN_DEG: float = 170.0
PLANK_BODY_LINE_WARNING_DEG: float = 160.0
PLANK_MIN_HOLD_SEC: float = 5.0
PLANK_GOOD_HOLD_SEC: float = 30.0
PLANK_EXCELLENT_HOLD_SEC: float = 60.0

# --- Wall sit: knee ~90° vs wall, common coaching cue ---

WALL_SIT_KNEE_TARGET_DEG: float = 90.0
WALL_SIT_KNEE_TOLERANCE_DEG: float = 15.0
WALL_SIT_BACK_TARGET_DEG: float = 90.0
WALL_SIT_BACK_TOLERANCE_DEG: float = 10.0
WALL_SIT_MIN_HOLD_SEC: float = 10.0
WALL_SIT_GOOD_HOLD_SEC: float = 30.0
WALL_SIT_EXCELLENT_HOLD_SEC: float = 60.0
