"""
Push-up counter using a finite state machine with hysteresis.
Tracks elbow angle to count repetitions and provide form feedback.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np

from ..utils.geometry import calculate_angle, AngleBuffer
from .interface import ExerciseCounter, RepCounterConfig


class PushUpState(Enum):
    """Push-up FSM states."""
    IDLE = "IDLE"           # No valid push-up pose detected
    UP = "UP"               # Arms extended (top position)
    DESCENDING = "DESCENDING"  # In the descent phase
    DOWN = "DOWN"           # Bottom of push-up (chest near ground)
    ASCENDING = "ASCENDING"    # In the ascent phase


@dataclass
class PushUpConfig(RepCounterConfig):
    """
    Configuration for push-up counter.
    
    Extends base config with push-up specific thresholds.
    """
    # Angle thresholds (in degrees)
    extension_threshold: float = 160.0  # Angle for arms fully extended
    flexion_threshold: float = 90.0     # Angle for bottom position (elbow at 90°)
    
    # Hysteresis buffer (prevents false transitions)
    buffer: float = 15.0
    
    # Smoothing settings
    angle_buffer_size: int = 3
    
    # Feedback thresholds
    too_shallow_angle: float = 100.0    # Above this is too shallow
    too_deep_angle: float = 70.0        # Below this might be too deep
    
    # Body alignment thresholds (for back straightness)
    body_angle_threshold: float = 160.0  # Minimum shoulder-hip-ankle angle
    hip_drop_threshold: float = 15.0     # Max hip drop relative to shoulders
    
    # Which arm to track (left, right, or average)
    use_both_arms: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
            
        if self.extension_threshold <= self.flexion_threshold:
            raise ValueError("extension_threshold must be greater than flexion_threshold")
        if self.buffer < 0:
            raise ValueError("buffer must be non-negative")
        if self.angle_buffer_size < 1:
            raise ValueError("angle_buffer_size must be at least 1")

class PushUpCounter(ExerciseCounter):
    """
    Push-up counter using finite state machine with dual thresholds.
    
    Tracks elbow angle through a complete push-up cycle:
    UP -> DESCENDING -> DOWN -> ASCENDING -> UP (count increments)
    
    Also monitors body alignment to ensure proper form.
    """
    
    def __init__(self, config: Optional[PushUpConfig] = None):
        """
        Initialize push-up counter.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or PushUpConfig()
        
        # FSM state
        self.state = PushUpState.IDLE
        self.count = 0
        
        # Internal tracking
        self._last_elbow_angle = 0.0
        self._last_body_angle = 0.0
        self._min_angle_in_rep = float('inf')
        self._max_angle_in_rep = 0.0
        self._rep_start_time: Optional[float] = None
        self._rep_end_time: Optional[float] = None
        self._rep_metrics: List[Dict[str, Any]] = []
        
        # Smoothing buffers
        self._elbow_buffer = AngleBuffer(size=self.config.angle_buffer_size)
        self._body_buffer = AngleBuffer(size=self.config.angle_buffer_size)
        
        # Performance tracking
        self._total_valid_frames = 0
        self._total_frames = 0
        
        # Form warnings
        self._form_warnings: List[str] = []
    
    @property
    def name(self) -> str:
        """Return the name of the exercise."""
        return "pushup"
    
    @property
    def required_landmarks(self) -> list:
        """
        Return list of landmark names required for push-up.
        
        Returns:
            List of strings matching the keys expected in update() landmarks dict
        """
        return [
            # Shoulders
            'left_shoulder', 'right_shoulder',
            # Elbows
            'left_elbow', 'right_elbow',
            # Wrists
            'left_wrist', 'right_wrist',
            # Hips
            'left_hip', 'right_hip',
            # Ankles (for body alignment)
            'left_ankle', 'right_ankle'
        ]
    
    def update(self, landmarks: Dict[str, Tuple[float, float]], confidence: float) -> Tuple[int, Enum, float]:
        """
        Update counter state with new frame data.
        
        Args:
            landmarks: Dictionary containing required landmarks
            confidence: Overall confidence of pose detection
            
        Returns:
            Tuple of (count, state, elbow_angle)
        """
        self._total_frames += 1
        self._form_warnings = []
        
        # Reset if confidence is too low
        if confidence < self.config.min_confidence:
            if self.state != PushUpState.IDLE:
                self.state = PushUpState.IDLE
            return self.count, self.state, self._last_elbow_angle
        
        # Extract required landmarks
        try:
            # Shoulders
            left_shoulder = landmarks['left_shoulder']
            right_shoulder = landmarks['right_shoulder']
            
            # Elbows
            left_elbow = landmarks['left_elbow']
            right_elbow = landmarks['right_elbow']
            
            # Wrists
            left_wrist = landmarks['left_wrist']
            right_wrist = landmarks['right_wrist']
            
            # Hips
            left_hip = landmarks['left_hip']
            right_hip = landmarks['right_hip']
            
            # Ankles
            left_ankle = landmarks['left_ankle']
            right_ankle = landmarks['right_ankle']
            
        except KeyError as e:
            raise ValueError(f"Missing required landmark: {e}")
        
        # Calculate elbow angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Calculate body angles (shoulder-hip-ankle) for form checking
        left_body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
        right_body_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
        body_angle = (left_body_angle + right_body_angle) / 2.0
        
        # Combine elbow angles based on configuration
        if self.config.use_both_arms:
            elbow_angle = (left_elbow_angle + right_elbow_angle) / 2.0
        else:
            # Use the more flexed elbow (smaller angle) - more conservative
            elbow_angle = min(left_elbow_angle, right_elbow_angle)
        
        # Add to smoothing buffers
        self._elbow_buffer.add(elbow_angle)
        self._body_buffer.add(body_angle)
        
        smoothed_elbow = self._elbow_buffer.get_smoothed()
        smoothed_body = self._body_buffer.get_smoothed()
        
        if smoothed_elbow is not None:
            elbow_angle = smoothed_elbow
        if smoothed_body is not None:
            body_angle = smoothed_body
        
        self._last_elbow_angle = elbow_angle
        self._last_body_angle = body_angle
        
        # Check form before updating state
        self._check_form(body_angle, elbow_angle)
        
        # Update FSM state
        self._update_state(elbow_angle)
        
        return self.count, self.state, elbow_angle
    
    def _check_form(self, body_angle: float, elbow_angle: float) -> None:
        """
        Check for form errors and generate warnings.
        
        Args:
            body_angle: Current shoulder-hip-ankle angle
            elbow_angle: Current elbow angle
        """
        cfg = self.config
        
        # Check if back is straight
        if body_angle < cfg.body_angle_threshold:
            if body_angle < cfg.body_angle_threshold - 20:
                self._form_warnings.append("Hips too high or low - keep back straight")
            else:
                self._form_warnings.append("Straighten your back")
        
        # Check depth during descent/bottom
        if self.state in [PushUpState.DESCENDING, PushUpState.DOWN]:
            if elbow_angle > cfg.too_shallow_angle:
                self._form_warnings.append("Go deeper - chest to ground")
        
        # Check for locking out elbows at top
        if self.state == PushUpState.UP:
            if elbow_angle < cfg.extension_threshold - 10:
                self._form_warnings.append("Fully extend your arms at the top")
    
    def _update_state(self, angle: float) -> None:
        """
        Update FSM based on current elbow angle.
        
        Args:
            angle: Current smoothed elbow angle
        """
        cfg = self.config
        
        # State transitions with hysteresis
        if self.state == PushUpState.IDLE:
            # IDLE -> UP: Person in push-up position with arms extended
            if angle > cfg.extension_threshold:
                self.state = PushUpState.UP
                self._total_valid_frames += 1
        
        elif self.state == PushUpState.UP:
            # UP -> DESCENDING: Start push-up (cross below extension threshold minus buffer)
            if angle < cfg.extension_threshold - cfg.buffer:
                self.state = PushUpState.DESCENDING
                self._rep_start_time = time.time()
                self._min_angle_in_rep = angle
                self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
        
        elif self.state == PushUpState.DESCENDING:
            # Update min angle
            self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            # DESCENDING -> DOWN: Reached bottom position
            if angle < cfg.flexion_threshold:
                self.state = PushUpState.DOWN
            
            # DESCENDING -> UP: Aborted descent (pushed back up)
            elif angle > cfg.extension_threshold:
                self.state = PushUpState.UP
                self._reset_rep_tracking()
        
        elif self.state == PushUpState.DOWN:
            # Update min angle
            self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            # DOWN -> ASCENDING: Started ascent
            if angle > cfg.flexion_threshold + cfg.buffer:
                self.state = PushUpState.ASCENDING
        
        elif self.state == PushUpState.ASCENDING:
            # Update max angle
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            # ASCENDING -> UP: Completed rep
            if angle > cfg.extension_threshold:
                self.count += 1
                self.state = PushUpState.UP
                self._rep_end_time = time.time()
                self._store_rep_metrics()
                self._reset_rep_tracking()
            
            # ASCENDING -> DOWN: User descended again without completing rep
            elif angle < cfg.flexion_threshold:
                self.state = PushUpState.DOWN
        
        self._total_valid_frames += 1
    
    def _store_rep_metrics(self) -> None:
        """Store metrics for the completed repetition."""
        if self._rep_start_time and self._rep_end_time:
            duration = self._rep_end_time - self._rep_start_time
        else:
            duration = 0.0
        
        # Calculate average body angle during rep (for form assessment)
        avg_body_angle = self._last_body_angle  # Simplified - could store history
        
        self._rep_metrics.append({
            'rep_number': len(self._rep_metrics) + 1,
            'min_elbow_angle': self._min_angle_in_rep,
            'max_elbow_angle': self._max_angle_in_rep,
            'range_of_motion': self._max_angle_in_rep - self._min_angle_in_rep,
            'duration': duration,
            'avg_body_angle': avg_body_angle,
            'depth_quality': self._classify_depth_quality(),
            'form_issues': self._form_warnings.copy() if self._form_warnings else []
        })
    
    def _classify_depth_quality(self) -> str:
        """Classify the quality of push-up depth."""
        if self._min_angle_in_rep <= self.config.flexion_threshold:
            return "good"
        elif self._min_angle_in_rep <= self.config.too_shallow_angle:
            return "shallow"
        else:
            return "very_shallow"
    
    def _reset_rep_tracking(self) -> None:
        """Reset tracking variables for a new rep."""
        self._min_angle_in_rep = float('inf')
        self._max_angle_in_rep = 0.0
        self._rep_start_time = None
        self._rep_end_time = None
    
    def reset(self) -> None:
        """Reset counter to initial state."""
        self.state = PushUpState.IDLE
        self.count = 0
        self._last_elbow_angle = 0.0
        self._last_body_angle = 0.0
        self._elbow_buffer.clear()
        self._body_buffer.clear()
        self._reset_rep_tracking()
        self._rep_metrics = []
        self._form_warnings = []
        self._total_valid_frames = 0
        self._total_frames = 0
    
    def get_progress(self) -> float:
        """
        Get normalized progress of current repetition (0.0 to 1.0).
        
        Returns:
            0.0 = up position / just started
            0.25 = descending
            0.5 = at bottom
            0.75 = ascending
            1.0 = completed (but returns 0.0 after completion)
        """
        cfg = self.config
        angle = self._last_elbow_angle
        
        if self.state == PushUpState.IDLE or self.state == PushUpState.UP:
            return 0.0
        
        elif self.state == PushUpState.DESCENDING:
            # Progress from 0.0 to 0.5 as angle decreases
            span = cfg.extension_threshold - cfg.flexion_threshold
            if span <= 0:
                return 0.25
            prog = (cfg.extension_threshold - angle) / span
            return min(0.5, max(0.0, prog * 0.5))
        
        elif self.state == PushUpState.DOWN:
            return 0.5
        
        elif self.state == PushUpState.ASCENDING:
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
        
        angle = self._last_elbow_angle
        cfg = self.config
        
        # No feedback in IDLE state
        if self.state == PushUpState.IDLE:
            return "Get into push-up position"
        
        # Feedback for UP state
        if self.state == PushUpState.UP:
            if angle < cfg.extension_threshold - 10:
                return "Lock your arms at the top"
            return "Ready - lower yourself down"
        
        # Feedback for DESCENDING
        if self.state == PushUpState.DESCENDING:
            if angle < cfg.flexion_threshold + 15:
                return "Almost there - go down further"
            elif angle > cfg.extension_threshold - 20:
                return "Bend your elbows more"
            return "Control your descent"
        
        # Feedback for DOWN
        if self.state == PushUpState.DOWN:
            if angle > cfg.flexion_threshold + cfg.buffer:
                return "Go a bit deeper"
            elif angle < cfg.too_deep_angle:
                return "Don't let your chest touch the ground"
            return "Good depth - now push up"
        
        # Feedback for ASCENDING
        if self.state == PushUpState.ASCENDING:
            if angle < cfg.flexion_threshold + 20:
                return "Push through your palms"
            elif angle > cfg.extension_threshold - 15:
                return "Almost up - lockout"
            return "Keep pushing"
        
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
                'exercise': 'pushup',
                'total_reps': 0,
                'avg_range_of_motion': 0,
                'avg_duration': 0,
                'good_reps': 0,
                'shallow_reps': 0,
                'very_shallow_reps': 0,
                'reps_with_form_issues': 0,
                'detection_quality': 0
            }
        
        # Count quality categories
        quality_counts = {'good': 0, 'shallow': 0, 'very_shallow': 0}
        reps_with_issues = 0
        
        for rep in self._rep_metrics:
            quality_counts[rep['depth_quality']] += 1
            if rep['form_issues']:
                reps_with_issues += 1
        
        # Calculate averages
        avg_rom = np.mean([r['range_of_motion'] for r in self._rep_metrics])
        avg_duration = np.mean([r['duration'] for r in self._rep_metrics])
        
        # Detection quality (percentage of frames with valid pose)
        detection_quality = (self._total_valid_frames / max(1, self._total_frames)) * 100
        
        return {
            'exercise': 'pushup',
            'total_reps': len(self._rep_metrics),
            'avg_range_of_motion': round(avg_rom, 1),
            'avg_duration': round(avg_duration, 2),
            'good_reps': quality_counts['good'],
            'shallow_reps': quality_counts['shallow'],
            'very_shallow_reps': quality_counts['very_shallow'],
            'reps_with_form_issues': reps_with_issues,
            'detection_quality': round(detection_quality, 1)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export counter state to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'last_elbow_angle': self._last_elbow_angle,
            'last_body_angle': self._last_body_angle,
            'min_angle_in_rep': self._min_angle_in_rep if self._min_angle_in_rep != float('inf') else None,
            'max_angle_in_rep': self._max_angle_in_rep,
            'form_warnings': self._form_warnings,
            'config': self.config.to_dict(),
            'rep_metrics': self._rep_metrics
        })
        return base_dict