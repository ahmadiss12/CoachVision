"""
Bicep curl counter using a finite state machine with hysteresis.
Tracks elbow angle to count repetitions and checks for proper form
(keeping elbows pinned to sides, no swinging).
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np

from ..utils.geometry import calculate_angle, calculate_distance
from .interface import ExerciseCounter, RepCounterConfig


class BicepCurlState(Enum):
    """Bicep curl FSM states."""
    IDLE = "IDLE"           # No valid curl pose detected
    DOWN = "DOWN"           # Arms extended (bottom position)
    ASCENDING = "ASCENDING" # In the ascent phase (curling up)
    UP = "UP"               # Arms flexed (top position)
    DESCENDING = "DESCENDING" # In the descent phase (lowering down)


@dataclass
class BicepCurlConfig(RepCounterConfig):
    """
    Configuration for bicep curl counter.
    
    Extends base config with bicep curl specific thresholds.
    """
    # Angle thresholds (in degrees)
    extension_threshold: float = 160.0  # Angle for arms extended (nearly straight)
    flexion_threshold: float = 45.0     # Angle for fully curled
    
    # Hysteresis buffer (prevents false transitions)
    buffer: float = 15.0
    
    # Smoothing settings
    angle_buffer_size: int = 3
    
    # Form thresholds
    elbow_movement_threshold: float = 0.05  # Max elbow movement from body (normalized)
    shoulder_movement_threshold: float = 0.03  # Max shoulder movement (no swinging)
    
    # Which arm to track (left, right, or average)
    track_both_arms: bool = True
    require_alternating: bool = False  # Whether to alternate arms
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
            
        if self.extension_threshold <= self.flexion_threshold:
            raise ValueError("extension_threshold must be greater than flexion_threshold")
        if self.buffer < 0:
            raise ValueError("buffer must be non-negative")

class BicepCurlCounter(ExerciseCounter):
    """
    Bicep curl counter using finite state machine with dual thresholds.
    
    Tracks elbow angle through a complete curl cycle:
    DOWN -> ASCENDING -> UP -> DESCENDING -> DOWN (count increments)
    
    Checks for proper form:
    - Elbows pinned to sides
    - No shoulder swinging
    - Full range of motion
    """
    
    def __init__(self, config: Optional[BicepCurlConfig] = None):
        """
        Initialize bicep curl counter.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or BicepCurlConfig()
        
        # FSM state
        self.state = BicepCurlState.IDLE
        self.count = 0
        
        # Arm tracking
        self.active_arm = "right"  # Default, will detect based on movement
        self.last_arm = "right"
        self.last_arm_time = 0.0
        
        # Internal tracking
        self._last_elbow_angle = 0.0
        self._last_shoulder_pos: Optional[Tuple[float, float]] = None
        self._last_elbow_pos: Optional[Tuple[float, float]] = None
        self._initial_shoulder_pos: Optional[Tuple[float, float]] = None
        
        self._min_angle_in_rep = float('inf')
        self._max_angle_in_rep = 0.0
        self._rep_start_time: Optional[float] = None
        self._rep_end_time: Optional[float] = None
        self._rep_metrics: List[Dict[str, Any]] = []
        
        # Smoothing buffers
        self._left_buffer = []
        self._right_buffer = []
        
        # Performance tracking
        self._total_valid_frames = 0
        self._total_frames = 0
        
        # Form warnings
        self._form_warnings: List[str] = []
        
        # Movement tracking
        self._max_elbow_movement = 0.0
        self._max_shoulder_movement = 0.0
    
    @property
    def name(self) -> str:
        """Return the name of the exercise."""
        return "bicep_curl"
    
    @property
    def required_landmarks(self) -> list:
        """
        Return list of landmark names required for bicep curl.
        
        Returns:
            List of strings matching the keys expected in update() landmarks dict
        """
        return [
            # Shoulders (for stability check)
            'left_shoulder', 'right_shoulder',
            # Elbows (primary tracking)
            'left_elbow', 'right_elbow',
            # Wrists
            'left_wrist', 'right_wrist'
        ]
    
    def _detect_active_arm(self, left_angle: float, right_angle: float) -> str:
        """
        Detect which arm is actively curling based on angle change.
        
        Args:
            left_angle: Current left elbow angle
            right_angle: Current right elbow angle
            
        Returns:
            "left", "right", or "both"
        """
        if not hasattr(self, '_prev_left'):
            self._prev_left = left_angle
            self._prev_right = right_angle
            return "right"  # Default
        
        # Calculate rate of change
        left_change = abs(left_angle - self._prev_left)
        right_change = abs(right_angle - self._prev_right)
        
        self._prev_left = left_angle
        self._prev_right = right_angle
        
        # Detect which arm is moving more
        if left_change > right_change + 5:  # Left moving significantly more
            return "left"
        elif right_change > left_change + 5:  # Right moving significantly more
            return "right"
        else:
            return "both"
    
    def _check_elbow_position(self, elbow_pos: Tuple[float, float], shoulder_pos: Tuple[float, float]) -> bool:
        """
        Check if elbow is staying close to body (not flaring out).
        
        Args:
            elbow_pos: Current elbow position
            shoulder_pos: Current shoulder position
            
        Returns:
            True if elbow position is acceptable
        """
        if self._last_elbow_pos is None:
            self._last_elbow_pos = elbow_pos
            return True
        
        # Calculate horizontal distance from shoulder
        # In normalized coordinates, elbow should be roughly under shoulder
        horizontal_distance = abs(elbow_pos[0] - shoulder_pos[0])
        
        if horizontal_distance > self.config.elbow_movement_threshold:
            if horizontal_distance > self.config.elbow_movement_threshold * 2:
                self._form_warnings.append("Keep elbows pinned to your sides")
            else:
                self._form_warnings.append("Elbows drifting - keep them stable")
            return False
        
        # Track maximum movement during rep
        if self._initial_shoulder_pos is not None:
            movement = calculate_distance(elbow_pos, self._last_elbow_pos)
            self._max_elbow_movement = max(self._max_elbow_movement, movement)
        
        self._last_elbow_pos = elbow_pos
        return True
    
    def _check_shoulder_stability(self, shoulder_pos: Tuple[float, float]) -> bool:
        """
        Check if shoulders are stable (no swinging).
        
        Args:
            shoulder_pos: Current shoulder position
            
        Returns:
            True if shoulder movement is acceptable
        """
        if self._last_shoulder_pos is None:
            self._last_shoulder_pos = shoulder_pos
            self._initial_shoulder_pos = shoulder_pos
            return True
        
        # Calculate movement from last frame
        movement = calculate_distance(shoulder_pos, self._last_shoulder_pos)
        self._max_shoulder_movement = max(self._max_shoulder_movement, movement)
        
        # Check against threshold
        if movement > self.config.shoulder_movement_threshold:
            if movement > self.config.shoulder_movement_threshold * 2:
                self._form_warnings.append("Don't swing your shoulders - use only arms")
            else:
                self._form_warnings.append("Minimize shoulder movement")
            return False
        
        self._last_shoulder_pos = shoulder_pos
        return True
    
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
            if self.state != BicepCurlState.IDLE:
                self.state = BicepCurlState.IDLE
            return self.count, self.state, self._last_elbow_angle
        
        # Extract required landmarks
        try:
            left_shoulder = landmarks['left_shoulder']
            right_shoulder = landmarks['right_shoulder']
            left_elbow = landmarks['left_elbow']
            right_elbow = landmarks['right_elbow']
            left_wrist = landmarks['left_wrist']
            right_wrist = landmarks['right_wrist']
            
        except KeyError as e:
            raise ValueError(f"Missing required landmark: {e}")
        
        # Calculate elbow angles
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Smooth angles
        self._left_buffer.append(left_angle)
        self._right_buffer.append(right_angle)
        
        if len(self._left_buffer) > self.config.angle_buffer_size:
            self._left_buffer.pop(0)
        if len(self._right_buffer) > self.config.angle_buffer_size:
            self._right_buffer.pop(0)
        
        smoothed_left = np.mean(self._left_buffer)
        smoothed_right = np.mean(self._right_buffer)
        
        # Detect active arm
        active = self._detect_active_arm(smoothed_left, smoothed_right)
        
        # Determine which angle to use for counting
        if self.config.track_both_arms:
            # Use average if both arms are moving similarly
            if active == "both":
                elbow_angle = (smoothed_left + smoothed_right) / 2.0
                self.active_arm = "both"
                shoulder_pos = ((left_shoulder[0] + right_shoulder[0]) / 2,
                               (left_shoulder[1] + right_shoulder[1]) / 2)
                elbow_pos = ((left_elbow[0] + right_elbow[0]) / 2,
                            (left_elbow[1] + right_elbow[1]) / 2)
            else:
                # Track the more active arm
                self.active_arm = active
                if active == "left":
                    elbow_angle = smoothed_left
                    shoulder_pos = left_shoulder
                    elbow_pos = left_elbow
                else:  # right or default
                    elbow_angle = smoothed_right
                    shoulder_pos = right_shoulder
                    elbow_pos = right_elbow
        else:
            # Track specific arm (default right)
            self.active_arm = "right"
            elbow_angle = smoothed_right
            shoulder_pos = right_shoulder
            elbow_pos = right_elbow
        
        self._last_elbow_angle = elbow_angle
        
        # Check form
        self._check_elbow_position(elbow_pos, shoulder_pos)
        self._check_shoulder_stability(shoulder_pos)
        
        # Update FSM state
        self._update_state(elbow_angle)
        
        return self.count, self.state, elbow_angle
    
    def _update_state(self, angle: float) -> None:
        """
        Update FSM based on current elbow angle.
        
        Args:
            angle: Current smoothed elbow angle
        """
        cfg = self.config
        
        # State transitions with hysteresis
        if self.state == BicepCurlState.IDLE:
            # IDLE -> DOWN: Arms extended
            if angle > cfg.extension_threshold:
                self.state = BicepCurlState.DOWN
                self._total_valid_frames += 1
        
        elif self.state == BicepCurlState.DOWN:
            # DOWN -> ASCENDING: Starting curl
            if angle < cfg.extension_threshold - cfg.buffer:
                self.state = BicepCurlState.ASCENDING
                self._rep_start_time = time.time()
                self._min_angle_in_rep = angle
                self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
                self._max_elbow_movement = 0.0
                self._max_shoulder_movement = 0.0
        
        elif self.state == BicepCurlState.ASCENDING:
            # Update min angle (more flexed = smaller angle)
            self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            # ASCENDING -> UP: Reached top position
            if angle < cfg.flexion_threshold:
                self.state = BicepCurlState.UP
            
            # ASCENDING -> DOWN: Aborted ascent
            elif angle > cfg.extension_threshold:
                self.state = BicepCurlState.DOWN
                self._reset_rep_tracking()
        
        elif self.state == BicepCurlState.UP:
            # Update min angle
            self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            # UP -> DESCENDING: Starting descent
            if angle > cfg.flexion_threshold + cfg.buffer:
                self.state = BicepCurlState.DESCENDING
        
        elif self.state == BicepCurlState.DESCENDING:
            # Update max angle
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            # DESCENDING -> DOWN: Completed rep
            if angle > cfg.extension_threshold:
                self.count += 1
                self.state = BicepCurlState.DOWN
                self._rep_end_time = time.time()
                self._store_rep_metrics()
                self._reset_rep_tracking()
            
            # DESCENDING -> UP: Aborted descent
            elif angle < cfg.flexion_threshold:
                self.state = BicepCurlState.UP
        
        self._total_valid_frames += 1
    
    def _store_rep_metrics(self) -> None:
        """Store metrics for the completed repetition."""
        if self._rep_start_time and self._rep_end_time:
            duration = self._rep_end_time - self._rep_start_time
        else:
            duration = 0.0
        
        self._rep_metrics.append({
            'rep_number': len(self._rep_metrics) + 1,
            'arm_used': self.active_arm,
            'min_angle': self._min_angle_in_rep,
            'max_angle': self._max_angle_in_rep,
            'range_of_motion': self._max_angle_in_rep - self._min_angle_in_rep,
            'duration': duration,
            'max_elbow_movement': self._max_elbow_movement,
            'max_shoulder_movement': self._max_shoulder_movement,
            'depth_quality': self._classify_depth_quality(),
            'form_issues': self._form_warnings.copy() if self._form_warnings else []
        })
    
    def _classify_depth_quality(self) -> str:
        """Classify the quality of curl depth."""
        if self._min_angle_in_rep <= self.config.flexion_threshold:
            return "full_range"
        elif self._min_angle_in_rep <= self.config.flexion_threshold + 20:
            return "partial"
        else:
            return "very_partial"
    
    def _reset_rep_tracking(self) -> None:
        """Reset tracking variables for a new rep."""
        self._min_angle_in_rep = float('inf')
        self._max_angle_in_rep = 0.0
        self._rep_start_time = None
        self._rep_end_time = None
        self._last_shoulder_pos = None
        self._last_elbow_pos = None
        self._initial_shoulder_pos = None
        self._max_elbow_movement = 0.0
        self._max_shoulder_movement = 0.0
    
    def reset(self) -> None:
        """Reset counter to initial state."""
        self.state = BicepCurlState.IDLE
        self.count = 0
        self._last_elbow_angle = 0.0
        self._left_buffer.clear()
        self._right_buffer.clear()
        self._reset_rep_tracking()
        self._rep_metrics = []
        self._form_warnings = []
        self._total_valid_frames = 0
        self._total_frames = 0
        
        if hasattr(self, '_prev_left'):
            delattr(self, '_prev_left')
        if hasattr(self, '_prev_right'):
            delattr(self, '_prev_right')
    
    def get_progress(self) -> float:
        """
        Get normalized progress of current repetition (0.0 to 1.0).
        
        Returns:
            0.0 = down position
            0.25 = ascending
            0.5 = at top
            0.75 = descending
            1.0 = back to down (completed)
        """
        cfg = self.config
        angle = self._last_elbow_angle
        
        if self.state == BicepCurlState.IDLE or self.state == BicepCurlState.DOWN:
            return 0.0
        
        elif self.state == BicepCurlState.ASCENDING:
            # Progress from 0.0 to 0.5 as angle decreases
            span = cfg.extension_threshold - cfg.flexion_threshold
            if span <= 0:
                return 0.25
            prog = (cfg.extension_threshold - angle) / span
            return min(0.5, max(0.0, prog * 0.5))
        
        elif self.state == BicepCurlState.UP:
            return 0.5
        
        elif self.state == BicepCurlState.DESCENDING:
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
        if self.state == BicepCurlState.IDLE:
            return "Stand with arms extended"
        
        # Feedback for DOWN state
        if self.state == BicepCurlState.DOWN:
            if angle < cfg.extension_threshold - 10:
                return "Fully extend your arms at the bottom"
            return "Ready - curl the weight up"
        
        # Feedback for ASCENDING
        if self.state == BicepCurlState.ASCENDING:
            if angle < cfg.flexion_threshold + 15:
                return "Squeeze at the top"
            return "Curl up - keep elbows pinned"
        
        # Feedback for UP
        if self.state == BicepCurlState.UP:
            if angle > cfg.flexion_threshold + 10:
                return "Curl higher - full range of motion"
            return "Good - now lower slowly"
        
        # Feedback for DESCENDING
        if self.state == BicepCurlState.DESCENDING:
            if angle < cfg.extension_threshold - 20:
                return "Control the descent"
            return "Lower all the way down"
        
        return None
    
    def get_rep_metrics(self) -> List[Dict[str, Any]]:
        """Get metrics for all completed repetitions."""
        return self._rep_metrics.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the session."""
        if not self._rep_metrics:
            return {
                'exercise': 'bicep_curl',
                'total_reps': 0,
                'left_arm_reps': 0,
                'right_arm_reps': 0,
                'both_arms_reps': 0,
                'avg_range_of_motion': 0,
                'avg_duration': 0,
                'full_range_reps': 0,
                'partial_reps': 0,
                'very_partial_reps': 0,
                'reps_with_form_issues': 0,
                'detection_quality': 0
            }
        
        # Count by arm
        arm_counts = {'left': 0, 'right': 0, 'both': 0}
        quality_counts = {'full_range': 0, 'partial': 0, 'very_partial': 0}
        reps_with_issues = 0
        
        for rep in self._rep_metrics:
            arm_counts[rep['arm_used']] += 1
            quality_counts[rep['depth_quality']] += 1
            if rep['form_issues']:
                reps_with_issues += 1
        
        # Calculate averages
        avg_rom = np.mean([r['range_of_motion'] for r in self._rep_metrics])
        avg_duration = np.mean([r['duration'] for r in self._rep_metrics])
        
        detection_quality = (self._total_valid_frames / max(1, self._total_frames)) * 100
        
        return {
            'exercise': 'bicep_curl',
            'total_reps': len(self._rep_metrics),
            'left_arm_reps': arm_counts['left'],
            'right_arm_reps': arm_counts['right'],
            'both_arms_reps': arm_counts['both'],
            'avg_range_of_motion': round(avg_rom, 1),
            'avg_duration': round(avg_duration, 2),
            'full_range_reps': quality_counts['full_range'],
            'partial_reps': quality_counts['partial'],
            'very_partial_reps': quality_counts['very_partial'],
            'reps_with_form_issues': reps_with_issues,
            'detection_quality': round(detection_quality, 1)
        }