"""
Deadlift counter using a finite state machine with hysteresis.
Tracks hip angle (hip hinge) and knee angle to count repetitions and provide form feedback.
Focuses on maintaining a straight back and proper hip drive.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np

from ..utils.geometry import calculate_angle, calculate_distance
from .interface import ExerciseCounter, RepCounterConfig


class DeadliftState(Enum):
    """Deadlift FSM states."""
    IDLE = "IDLE"           # No valid deadlift pose detected
    UP = "UP"               # Standing position (hips extended)
    DESCENDING = "DESCENDING"  # In the descent phase (lowering the bar)
    DOWN = "DOWN"           # Bottom position (bar on ground, hips low)
    ASCENDING = "ASCENDING"    # In the ascent phase (lifting the bar)


@dataclass
class DeadliftConfig(RepCounterConfig):
    """
    Configuration for deadlift counter.
    
    Extends base config with deadlift-specific thresholds.
    """
    # Angle thresholds (in degrees)
    extension_threshold: float = 165.0  # Angle for standing position (hips extended)
    flexion_threshold: float = 120.0    # Angle for bottom position (hips hinged)
    
    # Hysteresis buffer (prevents false transitions)
    buffer: float = 5.0
    
    # Smoothing settings
    angle_buffer_size: int = 3
    
    # Back angle thresholds (for straight back)
    back_angle_target: float = 180.0    # Ideal straight back
    back_angle_tolerance: float = 15.0  # Acceptable deviation from straight
    
    # Knee angle thresholds (should be slightly bent, not fully extended)
    knee_angle_min: float = 150.0       # Minimum knee angle (don't lock out)
    knee_angle_max: float = 175.0       # Maximum knee angle (don't over-bend)
    
    # Hip height ratio (for checking if hips rise too fast)
    hip_rise_threshold: float = 0.1     # Normalized hip rise per frame
    
    # Bar path tracking (approximated by wrist position)
    bar_path_tolerance: float = 0.1     # How much bar path can deviate
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
            
        if self.extension_threshold <= self.flexion_threshold:
            raise ValueError("extension_threshold must be greater than flexion_threshold")
        if self.buffer < 0:
            raise ValueError("buffer must be non-negative")

class DeadliftCounter(ExerciseCounter):
    """
    Deadlift counter using finite state machine with dual thresholds.
    
    Tracks hip angle through a complete deadlift cycle:
    UP -> DESCENDING -> DOWN -> ASCENDING -> UP (count increments)
    
    Focuses on:
    - Maintaining straight back
    - Proper hip hinge movement
    - Controlled bar path
    - No rounding of spine
    """
    
    def __init__(self, config: Optional[DeadliftConfig] = None):
        """
        Initialize deadlift counter.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or DeadliftConfig()
        
        # FSM state
        self.state = DeadliftState.IDLE
        self.count = 0
        
        # Internal tracking
        self._last_hip_angle = 0.0
        self._last_knee_angle = 0.0
        self._last_back_angle = 0.0
        self._min_angle_in_rep = float('inf')
        self._max_angle_in_rep = 0.0
        self._rep_start_time: Optional[float] = None
        self._rep_end_time: Optional[float] = None
        self._rep_metrics: List[Dict[str, Any]] = []
        
        # Position tracking for form checks
        self._initial_hip_height: Optional[float] = None
        self._last_hip_height: Optional[float] = None
        self._initial_wrist_pos: Optional[Tuple[float, float]] = None
        self._max_bar_deviation = 0.0
        
        # Smoothing buffers
        self._hip_buffer = []
        self._knee_buffer = []
        self._back_buffer = []
        
        # Performance tracking
        self._total_valid_frames = 0
        self._total_frames = 0
        
        # Form warnings
        self._form_warnings: List[str] = []
        
        # Back rounding detection
        self._back_rounding_detected = False
    
    @property
    def name(self) -> str:
        """Return the name of the exercise."""
        return "deadlift"
    
    @property
    def required_landmarks(self) -> list:
        """
        Return list of landmark names required for deadlift.
        
        Returns:
            List of strings matching the keys expected in update() landmarks dict
        """
        return [
            # Shoulders (for back angle)
            'left_shoulder', 'right_shoulder',
            # Hips (primary tracking)
            'left_hip', 'right_hip',
            # Knees
            'left_knee', 'right_knee',
            # Ankles
            'left_ankle', 'right_ankle',
            # Wrists (for bar path approximation)
            'left_wrist', 'right_wrist'
        ]
    
    def _calculate_hip_angle(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate hip angle (shoulder-hip-knee).
        This is the primary angle for deadlift tracking.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Hip angle in degrees
        """
        # Use average of both sides for robustness
        left_angle = calculate_angle(
            landmarks['left_shoulder'],
            landmarks['left_hip'],
            landmarks['left_knee']
        )
        
        right_angle = calculate_angle(
            landmarks['right_shoulder'],
            landmarks['right_hip'],
            landmarks['right_knee']
        )
        
        return (left_angle + right_angle) / 2.0
    
    def _calculate_back_angle(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate back angle (shoulder-hip-ankle).
        Used to check if back is straight.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Back angle in degrees (180° = straight)
        """
        left_angle = calculate_angle(
            landmarks['left_shoulder'],
            landmarks['left_hip'],
            landmarks['left_ankle']
        )
        
        right_angle = calculate_angle(
            landmarks['right_shoulder'],
            landmarks['right_hip'],
            landmarks['right_ankle']
        )
        
        return (left_angle + right_angle) / 2.0
    
    def _calculate_knee_angle(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate knee angle (hip-knee-ankle).
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Knee angle in degrees
        """
        left_angle = calculate_angle(
            landmarks['left_hip'],
            landmarks['left_knee'],
            landmarks['left_ankle']
        )
        
        right_angle = calculate_angle(
            landmarks['right_hip'],
            landmarks['right_knee'],
            landmarks['right_ankle']
        )
        
        return (left_angle + right_angle) / 2.0
    
    def _check_back_angle(self, back_angle: float) -> bool:
        """
        Check if back is straight enough.
        
        Args:
            back_angle: Current back angle
            
        Returns:
            True if back angle is acceptable
        """
        cfg = self.config
        deviation = abs(back_angle - cfg.back_angle_target)
        
        if deviation > cfg.back_angle_tolerance:
            if back_angle < cfg.back_angle_target:
                if deviation > cfg.back_angle_tolerance * 1.5:
                    self._form_warnings.append("ROUNDING BACK! Keep spine straight")
                    self._back_rounding_detected = True
                else:
                    self._form_warnings.append("Straighten your back - chest out")
            else:
                self._form_warnings.append("Don't over-extend your back")
            return False
        
        return True
    
    def _check_knee_angle(self, knee_angle: float) -> bool:
        """
        Check if knee angle is within acceptable range.
        
        Args:
            knee_angle: Current knee angle
            
        Returns:
            True if knee angle is acceptable
        """
        cfg = self.config
        
        if knee_angle < cfg.knee_angle_min:
            self._form_warnings.append("Knees too bent - straighten slightly")
            return False
        elif knee_angle > cfg.knee_angle_max:
            self._form_warnings.append("Don't lock your knees")
            return False
        
        return True
    
    def _check_hip_rise(self, hip_height: float) -> bool:
        """
        Check if hips are rising too fast (common form error).
        
        Args:
            hip_height: Current normalized hip height (y-coordinate)
            
        Returns:
            True if hip rise is controlled
        """
        if self._last_hip_height is None:
            self._last_hip_height = hip_height
            return True
        
        # In image coordinates, y increases downward
        # So rising hips = decreasing y
        hip_rise = self._last_hip_height - hip_height
        
        if hip_rise > self.config.hip_rise_threshold:
            self._form_warnings.append("Hips rising too fast - keep chest down")
            return False
        
        self._last_hip_height = hip_height
        return True
    
    def _check_bar_path(self, wrist_pos: Tuple[float, float]) -> bool:
        """
        Check if bar path (approximated by wrists) is straight.
        
        Args:
            wrist_pos: Current wrist position
            
        Returns:
            True if bar path is acceptable
        """
        if self._initial_wrist_pos is None:
            self._initial_wrist_pos = wrist_pos
            return True
        
        # Calculate horizontal deviation from initial position
        deviation = abs(wrist_pos[0] - self._initial_wrist_pos[0])
        self._max_bar_deviation = max(self._max_bar_deviation, deviation)
        
        if deviation > self.config.bar_path_tolerance:
            if deviation > self.config.bar_path_tolerance * 1.5:
                self._form_warnings.append("Bar path not straight - keep bar close to body")
            else:
                self._form_warnings.append("Pull the bar in a straight line")
            return False
        
        return True
    
    def update(self, landmarks: Dict[str, Tuple[float, float]], confidence: float) -> Tuple[int, Enum, float]:
        """
        Update counter state with new frame data.
        
        Args:
            landmarks: Dictionary containing required landmarks
            confidence: Overall confidence of pose detection
            
        Returns:
            Tuple of (count, state, hip_angle)
        """
        self._total_frames += 1
        self._form_warnings = []
        
        # Reset if confidence is too low
        if confidence < self.config.min_confidence:
            if self.state != DeadliftState.IDLE:
                self.state = DeadliftState.IDLE
            return self.count, self.state, self._last_hip_angle
        
        # Extract required landmarks
        try:
            required_landmarks = [
                'left_shoulder', 'right_shoulder',
                'left_hip', 'right_hip',
                'left_knee', 'right_knee',
                'left_ankle', 'right_ankle',
                'left_wrist', 'right_wrist'
            ]
            for lm in required_landmarks:
                if lm not in landmarks:
                    raise KeyError(f"Missing landmark: {lm}")
                    
        except KeyError as e:
            raise ValueError(f"Missing required landmark: {e}")
        
        # Calculate all angles
        hip_angle = self._calculate_hip_angle(landmarks)
        back_angle = self._calculate_back_angle(landmarks)
        knee_angle = self._calculate_knee_angle(landmarks)
        
        # Use average wrist position as bar approximation
        wrist_pos = (
            (landmarks['left_wrist'][0] + landmarks['right_wrist'][0]) / 2,
            (landmarks['left_wrist'][1] + landmarks['right_wrist'][1]) / 2
        )
        
        # Get hip height (y-coordinate) for hip rise check
        hip_height = (landmarks['left_hip'][1] + landmarks['right_hip'][1]) / 2
        
        # Smooth angles with simple moving average
        self._hip_buffer.append(hip_angle)
        self._knee_buffer.append(knee_angle)
        self._back_buffer.append(back_angle)
        
        for buffer in [self._hip_buffer, self._knee_buffer, self._back_buffer]:
            if len(buffer) > self.config.angle_buffer_size:
                buffer.pop(0)
        
        smoothed_hip = np.mean(self._hip_buffer)
        smoothed_knee = np.mean(self._knee_buffer)
        smoothed_back = np.mean(self._back_buffer)
        
        self._last_hip_angle = smoothed_hip
        self._last_knee_angle = smoothed_knee
        self._last_back_angle = smoothed_back
        
        # Check form (collect warnings but don't block state transition)
        back_ok = self._check_back_angle(smoothed_back)
        knee_ok = self._check_knee_angle(smoothed_knee)
        
        # Only check hip rise and bar path during movement
        if self.state in [DeadliftState.DESCENDING, DeadliftState.ASCENDING]:
            self._check_hip_rise(hip_height)
            self._check_bar_path(wrist_pos)
        else:
            # Reset tracking for next rep
            self._initial_hip_height = None
            self._last_hip_height = None
            self._initial_wrist_pos = None
        
        # Update FSM state
        self._update_state(smoothed_hip, hip_height, wrist_pos)
        
        return self.count, self.state, smoothed_hip
    
    def _update_state(self, angle: float, hip_height: float, wrist_pos: Tuple[float, float]) -> None:
        """
        Update FSM based on current hip angle.
        
        Args:
            angle: Current smoothed hip angle
            hip_height: Current hip height (for tracking)
            wrist_pos: Current wrist position (for bar path)
        """
        cfg = self.config
        
        # State transitions with hysteresis
        if self.state == DeadliftState.IDLE:
            # IDLE -> UP: Person standing with hips extended
            if angle > cfg.extension_threshold:
                self.state = DeadliftState.UP
                self._total_valid_frames += 1
        
        elif self.state == DeadliftState.UP:
            # UP -> DESCENDING: Start deadlift (hip angle decreases)
            if angle < cfg.extension_threshold - cfg.buffer:
                self.state = DeadliftState.DESCENDING
                self._rep_start_time = time.time()
                self._min_angle_in_rep = angle
                self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
                self._initial_hip_height = hip_height
                self._last_hip_height = hip_height
                self._initial_wrist_pos = wrist_pos
                self._max_bar_deviation = 0.0
                self._back_rounding_detected = False
        
        elif self.state == DeadliftState.DESCENDING:
            # Update min angle (more flexed = smaller angle)
            self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            # DESCENDING -> DOWN: Reached bottom position
            if angle < cfg.flexion_threshold:
                self.state = DeadliftState.DOWN
            
            # DESCENDING -> UP: Aborted descent (stood back up)
            elif angle > cfg.extension_threshold:
                self.state = DeadliftState.UP
                self._reset_rep_tracking()
        
        elif self.state == DeadliftState.DOWN:
            # Update min angle
            self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            # DOWN -> ASCENDING: Started ascent
            if angle > cfg.flexion_threshold + cfg.buffer:
                self.state = DeadliftState.ASCENDING
        
        elif self.state == DeadliftState.ASCENDING:
            # Update max angle
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            # ASCENDING -> UP: Completed rep
            if angle > cfg.extension_threshold:
                self.count += 1
                self.state = DeadliftState.UP
                self._rep_end_time = time.time()
                self._store_rep_metrics()
                self._reset_rep_tracking()
            
            # ASCENDING -> DOWN: User descended again without completing rep
            elif angle < cfg.flexion_threshold:
                self.state = DeadliftState.DOWN
        
        self._total_valid_frames += 1
    
    def _store_rep_metrics(self) -> None:
        """Store metrics for the completed repetition."""
        if self._rep_start_time and self._rep_end_time:
            duration = self._rep_end_time - self._rep_start_time
        else:
            duration = 0.0
        
        # Calculate average back angle during rep
        avg_back_angle = self._last_back_angle
        
        self._rep_metrics.append({
            'rep_number': len(self._rep_metrics) + 1,
            'min_hip_angle': self._min_angle_in_rep,
            'max_hip_angle': self._max_angle_in_rep,
            'range_of_motion': self._max_angle_in_rep - self._min_angle_in_rep,
            'avg_back_angle': avg_back_angle,
            'avg_knee_angle': self._last_knee_angle,
            'duration': duration,
            'max_bar_deviation': self._max_bar_deviation,
            'back_rounding_detected': self._back_rounding_detected,
            'depth_quality': self._classify_depth_quality(),
            'form_issues': self._form_warnings.copy() if self._form_warnings else []
        })
    
    def _classify_depth_quality(self) -> str:
        """Classify the quality of deadlift depth."""
        if self._min_angle_in_rep <= self.config.flexion_threshold:
            return "good"  # Reached proper depth
        elif self._min_angle_in_rep <= self.config.flexion_threshold + 10:
            return "shallow"  # Slightly above proper depth
        else:
            return "too_shallow"  # Not going low enough
    
    def _reset_rep_tracking(self) -> None:
        """Reset tracking variables for a new rep."""
        self._min_angle_in_rep = float('inf')
        self._max_angle_in_rep = 0.0
        self._rep_start_time = None
        self._rep_end_time = None
        self._initial_hip_height = None
        self._last_hip_height = None
        self._initial_wrist_pos = None
        self._max_bar_deviation = 0.0
        self._back_rounding_detected = False
    
    def reset(self) -> None:
        """Reset counter to initial state."""
        self.state = DeadliftState.IDLE
        self.count = 0
        self._last_hip_angle = 0.0
        self._last_knee_angle = 0.0
        self._last_back_angle = 0.0
        self._hip_buffer.clear()
        self._knee_buffer.clear()
        self._back_buffer.clear()
        self._reset_rep_tracking()
        self._rep_metrics = []
        self._form_warnings = []
        self._total_valid_frames = 0
        self._total_frames = 0
    
    def get_progress(self) -> float:
        """
        Get normalized progress of current repetition (0.0 to 1.0).
        
        Returns:
            0.0 = standing / just started
            0.25 = descending
            0.5 = at bottom
            0.75 = ascending
            1.0 = completed (but returns 0.0 after completion)
        """
        cfg = self.config
        angle = self._last_hip_angle
        
        if self.state == DeadliftState.IDLE or self.state == DeadliftState.UP:
            return 0.0
        
        elif self.state == DeadliftState.DESCENDING:
            # Progress from 0.0 to 0.5 as angle decreases
            span = cfg.extension_threshold - cfg.flexion_threshold
            if span <= 0:
                return 0.25
            prog = (cfg.extension_threshold - angle) / span
            return min(0.5, max(0.0, prog * 0.5))
        
        elif self.state == DeadliftState.DOWN:
            return 0.5
        
        elif self.state == DeadliftState.ASCENDING:
            # Progress from 0.5 to 1.0 as angle increases
            span = cfg.extension_threshold - cfg.flexion_threshold
            if span <= 0:
                return 0.75
            prog = (angle - cfg.flexion_threshold) / span
            return min(1.0, max(0.5, 0.5 + prog * 0.5))
        
        return 0.0
    
    def get_feedback(self) -> Optional[str]:
        """
        Get form feedback based on current state and angles.
        
        Returns:
            String with feedback message or None if form is correct
        """
        # Return form warnings first (higher priority)
        if self._form_warnings:
            return self._form_warnings[0]
        
        angle = self._last_hip_angle
        back_angle = self._last_back_angle
        cfg = self.config
        
        # No feedback in IDLE state
        if self.state == DeadliftState.IDLE:
            return "Stand with bar over mid-foot"
        
        # Feedback for UP state
        if self.state == DeadliftState.UP:
            if angle < cfg.extension_threshold - 5:
                return "Fully extend hips at the top"
            return "Ready - hinge at hips to lower"
        
        # Feedback for DESCENDING
        if self.state == DeadliftState.DESCENDING:
            if angle < cfg.flexion_threshold + 15:
                return "Almost to the ground - keep back straight"
            elif angle > cfg.extension_threshold - 20:
                return "Push hips back - don't just bend knees"
            
            # Back angle feedback during descent
            if back_angle < cfg.back_angle_target - cfg.back_angle_tolerance:
                return "Keep your back straight - chest up"
            
            return "Control the descent"
        
        # Feedback for DOWN
        if self.state == DeadliftState.DOWN:
            if angle > cfg.flexion_threshold + cfg.buffer:
                return "Lower the weight to the ground"
            elif angle < cfg.flexion_threshold - 10:
                return "Don't round your back at the bottom"
            
            # Check back angle at bottom
            if back_angle < cfg.back_angle_target - cfg.back_angle_tolerance:
                return "CHEST UP! Straighten your back before lifting"
            
            return "Good position - drive through heels"
        
        # Feedback for ASCENDING
        if self.state == DeadliftState.ASCENDING:
            if angle < cfg.flexion_threshold + 20:
                return "Drive hips forward - push the ground away"
            elif angle > cfg.extension_threshold - 15:
                return "Almost up - squeeze glutes"
            
            # Back angle feedback during ascent
            if back_angle < cfg.back_angle_target - cfg.back_angle_tolerance:
                return "Don't let your back round as you lift"
            
            return "Keep bar close to body"
        
        return None
    
    def get_rep_metrics(self) -> List[Dict[str, Any]]:
        """
        Get metrics for all completed repetitions.
        
        Returns:
            List of dictionaries with per-rep metrics
        """
        return self._rep_metrics.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the session.
        
        Returns:
            Dictionary with session summary
        """
        if not self._rep_metrics:
            return {
                'exercise': 'deadlift',
                'total_reps': 0,
                'avg_range_of_motion': 0,
                'avg_duration': 0,
                'avg_back_angle': 0,
                'good_reps': 0,
                'shallow_reps': 0,
                'too_shallow_reps': 0,
                'reps_with_rounding': 0,
                'reps_with_form_issues': 0,
                'detection_quality': 0
            }
        
        # Count quality categories
        quality_counts = {'good': 0, 'shallow': 0, 'too_shallow': 0}
        reps_with_rounding = 0
        reps_with_issues = 0
        
        for rep in self._rep_metrics:
            quality_counts[rep['depth_quality']] += 1
            if rep['back_rounding_detected']:
                reps_with_rounding += 1
            if rep['form_issues']:
                reps_with_issues += 1
        
        # Calculate averages
        avg_rom = np.mean([r['range_of_motion'] for r in self._rep_metrics])
        avg_duration = np.mean([r['duration'] for r in self._rep_metrics])
        avg_back_angle = np.mean([r['avg_back_angle'] for r in self._rep_metrics])
        avg_bar_dev = np.mean([r['max_bar_deviation'] for r in self._rep_metrics])
        
        # Detection quality (percentage of frames with valid pose)
        detection_quality = (self._total_valid_frames / max(1, self._total_frames)) * 100
        
        return {
            'exercise': 'deadlift',
            'total_reps': len(self._rep_metrics),
            'avg_range_of_motion': round(avg_rom, 1),
            'avg_duration': round(avg_duration, 2),
            'avg_back_angle': round(avg_back_angle, 1),
            'avg_bar_deviation': round(avg_bar_dev, 3),
            'good_reps': quality_counts['good'],
            'shallow_reps': quality_counts['shallow'],
            'too_shallow_reps': quality_counts['too_shallow'],
            'reps_with_rounding': reps_with_rounding,
            'reps_with_form_issues': reps_with_issues,
            'detection_quality': round(detection_quality, 1)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export counter state to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'last_hip_angle': self._last_hip_angle,
            'last_knee_angle': self._last_knee_angle,
            'last_back_angle': self._last_back_angle,
            'min_angle_in_rep': self._min_angle_in_rep if self._min_angle_in_rep != float('inf') else None,
            'max_angle_in_rep': self._max_angle_in_rep,
            'max_bar_deviation': self._max_bar_deviation,
            'back_rounding_detected': self._back_rounding_detected,
            'form_warnings': self._form_warnings,
            'config': self.config.to_dict(),
            'rep_metrics': self._rep_metrics
        })
        return base_dict