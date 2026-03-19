"""
Exercise counter dispatcher.
Routes pose landmarks to the appropriate exercise counter based on exercise name.
Provides a unified interface for the main pipeline.
"""

from typing import Dict, Tuple, Optional, Any, List
from enum import Enum
import importlib

from .interface import ExerciseCounter
from .squat import SquatCounter, SquatConfig
from .pushup import PushUpCounter, PushUpConfig
from .lunge import LungeCounter, LungeConfig
from .deadlift import DeadliftCounter, DeadliftConfig
from .plank import PlankCounter, PlankConfig
from .bicep_curl import BicepCurlCounter, BicepCurlConfig
from .shoulder_press import ShoulderPressCounter, ShoulderPressConfig
from .situp import SitUpCounter, SitUpConfig
from .jumping_jack import JumpingJackCounter, JumpingJackConfig
from .high_knees import HighKneesCounter, HighKneesConfig
from .mountain_climber import MountainClimberCounter, MountainClimberConfig
from .wall_sit import WallSitCounter, WallSitConfig


class ExerciseType(Enum):
    """Enumeration of all supported exercises."""
    SQUAT = "squat"
    PUSHUP = "pushup"
    LUNGE = "lunge"
    DEADLIFT = "deadlift"
    PLANK = "plank"
    BICEP_CURL = "bicep_curl"
    SHOULDER_PRESS = "shoulder_press"
    SITUP = "situp"
    JUMPING_JACK = "jumping_jack"
    HIGH_KNEES = "high_knees"
    MOUNTAIN_CLIMBER = "mountain_climber"
    WALL_SIT = "wall_sit"
    
    @classmethod
    def from_string(cls, name: str) -> 'ExerciseType':
        """Convert string to ExerciseType (case-insensitive)."""
        name_lower = name.lower().replace(' ', '_').replace('-', '_')
        for exercise in cls:
            if exercise.value == name_lower:
                return exercise
        raise ValueError(f"Unknown exercise: {name}. Available: {[e.value for e in cls]}")


# Configuration presets for different experience levels
CONFIG_PRESETS = {
    "beginner": {
        "squat": {"flexion_threshold": 110, "buffer": 10},
        "pushup": {"flexion_threshold": 100, "buffer": 20},  # Partial pushups
        "lunge": {"flexion_threshold": 90, "buffer": 15},    # Less deep
        "deadlift": {"flexion_threshold": 130, "buffer": 10}, # Less hinge
        "plank": {"min_hold_time": 5},                        # Shorter holds
        "wall_sit": {"min_hold_time": 5},                     # Shorter holds
    },
    "intermediate": {
        "squat": {"flexion_threshold": 90, "buffer": 10},
        "pushup": {"flexion_threshold": 90, "buffer": 15},
        "lunge": {"flexion_threshold": 80, "buffer": 10},
        "deadlift": {"flexion_threshold": 120, "buffer": 5},
        "plank": {"min_hold_time": 10, "good_hold_time": 30},
        "wall_sit": {"min_hold_time": 10, "good_hold_time": 30},
    },
    "advanced": {
        "squat": {"flexion_threshold": 80, "buffer": 5},      # Deep squats
        "pushup": {"flexion_threshold": 80, "buffer": 10},    # Chest to ground
        "lunge": {"flexion_threshold": 70, "buffer": 5},      # Deep lunges
        "deadlift": {"flexion_threshold": 110, "buffer": 5},  # Full hinge
        "plank": {"min_hold_time": 15, "excellent_hold_time": 90},
        "wall_sit": {"min_hold_time": 15, "excellent_hold_time": 90},
    }
}


class ExerciseDispatcher:
    """
    Dispatches pose data to the appropriate exercise counter.
    
    Usage:
        dispatcher = ExerciseDispatcher()
        dispatcher.set_exercise("squat", level="intermediate")
        
        # In main loop:
        count, state, angle = dispatcher.update(landmarks, confidence)
        feedback = dispatcher.get_feedback()
        progress = dispatcher.get_progress()
    """
    
    # Mapping of exercise types to their counter classes and config classes
    _COUNTER_MAP = {
        ExerciseType.SQUAT: (SquatCounter, SquatConfig),
        ExerciseType.PUSHUP: (PushUpCounter, PushUpConfig),
        ExerciseType.LUNGE: (LungeCounter, LungeConfig),
        ExerciseType.DEADLIFT: (DeadliftCounter, DeadliftConfig),
        ExerciseType.PLANK: (PlankCounter, PlankConfig),
        ExerciseType.BICEP_CURL: (BicepCurlCounter, BicepCurlConfig),
        ExerciseType.SHOULDER_PRESS: (ShoulderPressCounter, ShoulderPressConfig),
        ExerciseType.SITUP: (SitUpCounter, SitUpConfig),
        ExerciseType.JUMPING_JACK: (JumpingJackCounter, JumpingJackConfig),
        ExerciseType.HIGH_KNEES: (HighKneesCounter, HighKneesConfig),
        ExerciseType.MOUNTAIN_CLIMBER: (MountainClimberCounter, MountainClimberConfig),
        ExerciseType.WALL_SIT: (WallSitCounter, WallSitConfig),
    }
    
    def __init__(self):
        """Initialize dispatcher with no active exercise."""
        self._current_exercise: Optional[ExerciseType] = None
        self._counter: Optional[ExerciseCounter] = None
        self._available_exercises = [ex.value for ex in ExerciseType]
    
    def set_exercise(self, exercise_name: str, level: str = "intermediate", custom_config: Optional[Dict] = None) -> None:
        """
        Set the current exercise and initialize its counter.
        
        Args:
            exercise_name: Name of exercise (e.g., "squat", "pushup")
            level: Experience level - "beginner", "intermediate", or "advanced"
            custom_config: Optional custom configuration overrides
        
        Raises:
            ValueError: If exercise name or level is invalid
        """
        # Parse exercise type
        try:
            exercise_type = ExerciseType.from_string(exercise_name)
        except ValueError as e:
            raise ValueError(f"Invalid exercise: {exercise_name}. Available: {self._available_exercises}")
        
        # Validate level
        if level not in CONFIG_PRESETS:
            raise ValueError(f"Invalid level: {level}. Available: {list(CONFIG_PRESETS.keys())}")
        
        # Get counter class and config class
        counter_class, config_class = self._COUNTER_MAP[exercise_type]
        
        config_kwargs = {
        'min_confidence': 0.5  # Default value
    }
    
        # Apply preset if available
        if exercise_type.value in CONFIG_PRESETS[level]:
            config_kwargs.update(CONFIG_PRESETS[level][exercise_type.value])
    
        # Apply custom overrides
        if custom_config:
            config_kwargs.update(custom_config)
    
            # Create config instance
        config = config_class(**config_kwargs)
        
        
        # Initialize counter
        self._counter = counter_class(config)
        self._current_exercise = exercise_type
        
        print(f"[OK] Switched to {exercise_type.value} ({level} level)")
    
    def update(self, landmarks: Dict[str, Tuple[float, float]], confidence: float) -> Tuple[int, Enum, float]:
        """
        Update the current exercise counter with new frame data.
        
        Args:
            landmarks: Dictionary mapping joint names to (x, y) coordinates
            confidence: Overall confidence score of pose detection
            
        Returns:
            Tuple of (count, state, primary_angle)
            
        Raises:
            RuntimeError: If no exercise has been set
        """
        if self._counter is None:
            raise RuntimeError("No exercise set. Call set_exercise() first.")
        
        return self._counter.update(landmarks, confidence)
    
    def reset(self) -> None:
        """Reset the current exercise counter."""
        if self._counter:
            self._counter.reset()
    
    def get_feedback(self) -> Optional[str]:
        """Get form feedback for current exercise."""
        if self._counter:
            return self._counter.get_feedback()
        return None
    
    def get_progress(self) -> float:
        """Get progress of current rep/hold (0.0 to 1.0)."""
        if self._counter:
            return self._counter.get_progress()
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for current exercise session."""
        if self._counter:
            return self._counter.get_summary()
        return {}
    
    def get_rep_metrics(self) -> List[Dict[str, Any]]:
        """Get detailed metrics for each rep/hold."""
        if self._counter and hasattr(self._counter, 'get_rep_metrics'):
            return self._counter.get_rep_metrics()
        return []
    
    def get_current_exercise(self) -> Optional[str]:
        """Get name of current exercise."""
        return self._current_exercise.value if self._current_exercise else None
    
    def get_required_landmarks(self) -> List[str]:
        """
        Get list of landmarks required for current exercise.
        Useful for filtering which landmarks to extract.
        """
        if self._counter:
            return self._counter.required_landmarks
        return []
    
    def is_static_exercise(self) -> bool:
        """Check if current exercise is static (hold-based) vs dynamic (rep-based)."""
        if self._current_exercise in [ExerciseType.PLANK, ExerciseType.WALL_SIT]:
            return True
        return False
    
    def export_session_data(self) -> Dict[str, Any]:
        """Export all session data for logging/saving."""
        if not self._counter:
            return {}
        
        return {
            'exercise': self.get_current_exercise(),
            'summary': self.get_summary(),
            'rep_metrics': self.get_rep_metrics(),
            'counter_state': self._counter.to_dict() if hasattr(self._counter, 'to_dict') else {}
        }
    
    def list_available_exercises(self) -> List[str]:
        """Get list of all available exercises."""
        return self._available_exercises.copy()
    
    def get_exercise_info(self, exercise_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about an exercise.
        
        Args:
            exercise_name: Name of exercise (uses current if None)
            
        Returns:
            Dictionary with exercise information
        """
        if exercise_name is None:
            if self._current_exercise is None:
                return {}
            exercise_type = self._current_exercise
        else:
            try:
                exercise_type = ExerciseType.from_string(exercise_name)
            except ValueError:
                return {}
        
        # Basic info for all exercises
        info = {
            'name': exercise_type.value,
            'type': 'static' if exercise_type in [ExerciseType.PLANK, ExerciseType.WALL_SIT] else 'dynamic',
            'primary_joint': self._get_primary_joint(exercise_type),
        }
        
        # Add exercise-specific info
        if exercise_type == ExerciseType.SQUAT:
            info.update({
                'description': 'Knee flexion-extension cycle',
                'target_muscles': ['quadriceps', 'glutes', 'hamstrings'],
                'thresholds': 'Knee angle: 170° (up) → 90° (down)',
            })
        elif exercise_type == ExerciseType.PUSHUP:
            info.update({
                'description': 'Elbow flexion-extension in plank position',
                'target_muscles': ['chest', 'shoulders', 'triceps'],
                'thresholds': 'Elbow angle: 160° (up) → 90° (down)',
            })
        elif exercise_type == ExerciseType.LUNGE:
            info.update({
                'description': 'Alternating forward steps with knee flexion',
                'target_muscles': ['quadriceps', 'glutes', 'hamstrings'],
                'thresholds': 'Front knee: 160° (up) → 80° (down)',
            })
        elif exercise_type == ExerciseType.DEADLIFT:
            info.update({
                'description': 'Hip hinge movement',
                'target_muscles': ['hamstrings', 'glutes', 'lower back'],
                'thresholds': 'Hip angle: 165° (up) → 120° (down)',
            })
        elif exercise_type == ExerciseType.PLANK:
            info.update({
                'description': 'Static hold with straight body',
                'target_muscles': ['core', 'shoulders', 'back'],
                'thresholds': 'Body angle: >170° for proper form',
            })
        elif exercise_type == ExerciseType.BICEP_CURL:
            info.update({
                'description': 'Elbow flexion with palms up',
                'target_muscles': ['biceps', 'forearms'],
                'thresholds': 'Elbow angle: 160° (down) → 45° (up)',
            })
        elif exercise_type == ExerciseType.SHOULDER_PRESS:
            info.update({
                'description': 'Overhead press with dumbbells',
                'target_muscles': ['shoulders', 'triceps'],
                'thresholds': 'Elbow angle: 90° (down) → 170° (up)',
            })
        elif exercise_type == ExerciseType.SITUP:
            info.update({
                'description': 'Torso curl from lying to sitting',
                'target_muscles': ['abdominals', 'hip flexors'],
                'thresholds': 'Torso angle: 20° (down) → 80° (up)',
            })
        elif exercise_type == ExerciseType.JUMPING_JACK:
            info.update({
                'description': 'Jump to star position and back',
                'target_muscles': ['full body', 'cardiovascular'],
                'thresholds': 'Arms overhead, feet apart in open position',
            })
        elif exercise_type == ExerciseType.HIGH_KNEES:
            info.update({
                'description': 'Running in place with high knee drive',
                'target_muscles': ['hip flexors', 'quadriceps', 'core'],
                'thresholds': 'Knees should reach waist height',
            })
        elif exercise_type == ExerciseType.MOUNTAIN_CLIMBER:
            info.update({
                'description': 'Alternating knee drives in plank position',
                'target_muscles': ['core', 'shoulders', 'hip flexors'],
                'thresholds': 'Maintain plank while driving knees',
            })
        elif exercise_type == ExerciseType.WALL_SIT:
            info.update({
                'description': 'Static hold with back against wall',
                'target_muscles': ['quadriceps', 'glutes'],
                'thresholds': 'Knees at 90°, back against wall',
            })
        
        return info
    
    def _get_primary_joint(self, exercise_type: ExerciseType) -> str:
        """Get primary joint being tracked for an exercise."""
        joint_map = {
            ExerciseType.SQUAT: 'knee',
            ExerciseType.PUSHUP: 'elbow',
            ExerciseType.LUNGE: 'knee',
            ExerciseType.DEADLIFT: 'hip',
            ExerciseType.PLANK: 'spine',
            ExerciseType.BICEP_CURL: 'elbow',
            ExerciseType.SHOULDER_PRESS: 'shoulder',
            ExerciseType.SITUP: 'hip',
            ExerciseType.JUMPING_JACK: 'multiple',
            ExerciseType.HIGH_KNEES: 'hip',
            ExerciseType.MOUNTAIN_CLIMBER: 'hip',
            ExerciseType.WALL_SIT: 'knee',
        }
        return joint_map.get(exercise_type, 'unknown')


# Create a singleton instance for easy import
_default_dispatcher = ExerciseDispatcher()


def get_dispatcher() -> ExerciseDispatcher:
    """Get the default dispatcher instance."""
    return _default_dispatcher


def set_exercise(exercise_name: str, level: str = "intermediate", custom_config: Optional[Dict] = None) -> None:
    """Convenience function to set exercise on default dispatcher."""
    _default_dispatcher.set_exercise(exercise_name, level, custom_config)


def update_counter(landmarks: Dict[str, Tuple[float, float]], confidence: float) -> Tuple[int, Enum, float]:
    """Convenience function to update default dispatcher."""
    return _default_dispatcher.update(landmarks, confidence)


def get_feedback() -> Optional[str]:
    """Convenience function to get feedback from default dispatcher."""
    return _default_dispatcher.get_feedback()


def reset_counter() -> None:
    """Convenience function to reset default dispatcher."""
    _default_dispatcher.reset()


# Export all public interfaces
__all__ = [
    'ExerciseDispatcher',
    'ExerciseType',
    'get_dispatcher',
    'set_exercise',
    'update_counter',
    'get_feedback',
    'reset_counter',
    'CONFIG_PRESETS',
]