"""
Plank form checker for static hold exercise.
Tracks body alignment (shoulder-hip-ankle angle) to ensure straight back.
Monitors hold duration and detects form breaks.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np

from ..utils.geometry import calculate_angle, calculate_distance
from .interface import ExerciseCounter, RepCounterConfig


class PlankState(Enum):
    """Plank FSM states."""
    IDLE = "IDLE"           # No valid plank pose detected
    HOLDING = "HOLDING"     # Proper plank position maintained
    BROKEN = "BROKEN"       # Form broken (hips too high/low)


@dataclass
class PlankConfig(RepCounterConfig):
    """
    Configuration for plank form checker.
    
    Extends base config with plank-specific thresholds.
    """
    # Angle thresholds (in degrees)
    angle_threshold: float = 170.0      # Minimum angle for straight back
    angle_warning_threshold: float = 160.0  # Angle that triggers warning
    
    # Hold time thresholds (in seconds)
    min_hold_time: float = 5.0          # Minimum hold for a "valid" plank
    good_hold_time: float = 30.0        # Time for "good" plank
    excellent_hold_time: float = 60.0   # Time for "excellent" plank
    
    # Body alignment checks
    hip_height_tolerance: float = 0.1   # Normalized hip height variation allowed
    max_hip_drop: float = 0.15          # Maximum hip drop from ideal
    max_hip_raise: float = 0.15         # Maximum hip raise from ideal
    
    # Smoothing settings
    angle_buffer_size: int = 5
    
    # Rest time after form break
    recovery_time: float = 2.0           # Time needed to reset after form break
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
            
        if self.angle_threshold <= self.angle_warning_threshold:
            raise ValueError("angle_threshold must be greater than angle_warning_threshold")
        if self.min_hold_time < 0:
            raise ValueError("min_hold_time must be non-negative")

class PlankCounter(ExerciseCounter):
    """
    Plank form checker for static hold.
    
    Unlike rep-based counters, this tracks:
    - Hold duration with proper form
    - Form breaks and recovery
    - Body alignment (straight line from shoulders to ankles)
    - Hip height consistency
    
    The counter doesn't count "reps" but instead tracks hold time
    and provides real-time feedback on form quality.
    """
    
    def __init__(self, config: Optional[PlankConfig] = None):
        """
        Initialize plank counter.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or PlankConfig()
        
        # FSM state
        self.state = PlankState.IDLE
        self.hold_count = 0  # Number of holds (sessions of continuous good form)
        
        # Timing tracking
        self._session_start_time: Optional[float] = None
        self._hold_start_time: Optional[float] = None
        self._break_start_time: Optional[float] = None
        self._total_hold_time = 0.0
        self._current_hold_duration = 0.0
        
        # Angle tracking
        self._last_body_angle = 0.0
        self._angle_history = []
        
        # Hip position tracking (for consistency)
        self._initial_hip_height: Optional[float] = None
        self._hip_height_history = []
        self._max_hip_deviation = 0.0
        
        # Smoothing buffer
        self._angle_buffer = []
        
        # Performance tracking
        self._total_valid_frames = 0
        self._total_frames = 0
        
        # Form warnings
        self._form_warnings: List[str] = []
        
        # Hold sessions tracking
        self._hold_sessions: List[Dict[str, Any]] = []
        
        # Best hold of session
        self._best_hold_duration = 0.0
    
    @property
    def name(self) -> str:
        """Return the name of the exercise."""
        return "plank"
    
    @property
    def required_landmarks(self) -> list:
        """
        Return list of landmark names required for plank.
        
        Returns:
            List of strings matching the keys expected in update() landmarks dict
        """
        return [
            # Shoulders
            'left_shoulder', 'right_shoulder',
            # Hips (primary for alignment)
            'left_hip', 'right_hip',
            # Ankles
            'left_ankle', 'right_ankle',
            # Optional: elbows/wrists for arm position (if needed)
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist'
        ]
    
    def _calculate_body_angle(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate body angle (shoulder-hip-ankle).
        For a proper plank, this should be close to 180°.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Body angle in degrees
        """
        # Use average of both sides for robustness
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
    
    def _calculate_hip_height(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate normalized hip height (average y-coordinate).
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Average hip y-coordinate
        """
        left_hip_y = landmarks['left_hip'][1]
        right_hip_y = landmarks['right_hip'][1]
        return (left_hip_y + right_hip_y) / 2.0
    
    def _check_hip_alignment(self, hip_height: float) -> bool:
        """
        Check if hips are at consistent height (not sagging or piking).
        
        Args:
            hip_height: Current normalized hip height
            
        Returns:
            True if hip position is acceptable
        """
        if self._initial_hip_height is None:
            self._initial_hip_height = hip_height
            return True
        
        # Track deviation
        deviation = hip_height - self._initial_hip_height
        abs_deviation = abs(deviation)
        self._max_hip_deviation = max(self._max_hip_deviation, abs_deviation)
        
        # Store in history for trend analysis
        self._hip_height_history.append(hip_height)
        if len(self._hip_height_history) > 30:  # Keep last ~1 second at 30fps
            self._hip_height_history.pop(0)
        
        # Check for sagging hips (too low) - y increases downward
        if deviation > self.config.max_hip_drop:
            self._form_warnings.append("Hips too low - engage core and glutes")
            return False
        
        # Check for piked hips (too high)
        if deviation < -self.config.max_hip_raise:
            self._form_warnings.append("Hips too high - lower to straight line")
            return False
        
        # Check for excessive wobble (variance in recent history)
        if len(self._hip_height_history) > 10:
            variance = np.var(self._hip_height_history[-10:])
            if variance > 0.001:  # Empirically determined threshold
                self._form_warnings.append("Keep hips stable - less wobble")
                # Not returning False because form isn't broken, just unstable
        
        return True
    
    def _get_form_quality(self, angle: float) -> str:
        """
        Get qualitative assessment of current form.
        
        Args:
            angle: Current body angle
            
        Returns:
            String: "excellent", "good", "warning", or "broken"
        """
        if angle >= self.config.angle_threshold:
            return "excellent"
        elif angle >= self.config.angle_threshold - 5:
            return "good"
        elif angle >= self.config.angle_warning_threshold:
            return "warning"
        else:
            return "broken"
    
    def update(self, landmarks: Dict[str, Tuple[float, float]], confidence: float) -> Tuple[int, Enum, float]:
        """
        Update counter state with new frame data.
        
        Args:
            landmarks: Dictionary containing required landmarks
            confidence: Overall confidence of pose detection
            
        Returns:
            Tuple of (hold_count, state, body_angle)
            Note: hold_count is number of completed hold sessions, not reps
        """
        self._total_frames += 1
        self._form_warnings = []
        
        current_time = time.time()
        
        # Reset if confidence is too low
        if confidence < self.config.min_confidence:
            if self.state != PlankState.IDLE:
                self._end_hold_session(current_time)
                self.state = PlankState.IDLE
            return self.hold_count, self.state, self._last_body_angle
        
        # Extract required landmarks
        try:
            required_landmarks = [
                'left_shoulder', 'right_shoulder',
                'left_hip', 'right_hip',
                'left_ankle', 'right_ankle'
            ]
            for lm in required_landmarks:
                if lm not in landmarks:
                    raise KeyError(f"Missing landmark: {lm}")
                    
        except KeyError as e:
            raise ValueError(f"Missing required landmark: {e}")
        
        # Calculate body angle
        body_angle = self._calculate_body_angle(landmarks)
        
        # Get hip height
        hip_height = self._calculate_hip_height(landmarks)
        
        # Smooth angle
        self._angle_buffer.append(body_angle)
        if len(self._angle_buffer) > self.config.angle_buffer_size:
            self._angle_buffer.pop(0)
        
        smoothed_angle = np.mean(self._angle_buffer)
        self._last_body_angle = smoothed_angle
        
        # Track angle history for trend analysis
        self._angle_history.append(smoothed_angle)
        if len(self._angle_history) > 30:  # Keep last ~1 second
            self._angle_history.pop(0)
        
        # Check form
        hip_ok = self._check_hip_alignment(hip_height)
        form_quality = self._get_form_quality(smoothed_angle)
        
        # Update FSM state
        self._update_state(smoothed_angle, form_quality, hip_ok, current_time)
        
        # Update timing if in HOLDING state
        if self.state == PlankState.HOLDING:
            if self._hold_start_time is None:
                self._hold_start_time = current_time
            else:
                self._current_hold_duration = current_time - self._hold_start_time
                self._total_hold_time += (1.0 / 30)  # Approximate per frame
                
                # Update best hold
                if self._current_hold_duration > self._best_hold_duration:
                    self._best_hold_duration = self._current_hold_duration
        
        elif self.state == PlankState.BROKEN:
            # Track break time for recovery
            if self._break_start_time is None:
                self._break_start_time = current_time
        
        self._total_valid_frames += 1
        
        return self.hold_count, self.state, smoothed_angle
    
    def _update_state(self, angle: float, form_quality: str, hip_ok: bool, current_time: float) -> None:
        """
        Update FSM based on current form.
        
        Args:
            angle: Current body angle
            form_quality: Qualitative form assessment
            hip_ok: Whether hip position is acceptable
            current_time: Current timestamp
        """
        
        if self.state == PlankState.IDLE:
            # IDLE -> HOLDING: Person in good plank position
            if form_quality in ["excellent", "good"] and hip_ok:
                self.state = PlankState.HOLDING
                self._session_start_time = current_time
                self._hold_start_time = current_time
                self._initial_hip_height = None  # Will be set in first check
                self._total_valid_frames += 1
        
        elif self.state == PlankState.HOLDING:
            # Check if form is still good
            if form_quality == "broken" or not hip_ok:
                # Form broken - end this hold session
                self.state = PlankState.BROKEN
                self._end_hold_session(current_time)
                self._break_start_time = current_time
            elif form_quality == "warning":
                # Warning but not broken - add feedback but keep holding
                if angle < self.config.angle_threshold:
                    self._form_warnings.append("Straighten your body - engage core")
        
        elif self.state == PlankState.BROKEN:
            # Check if form has recovered
            if form_quality in ["excellent", "good"] and hip_ok:
                # Check if enough recovery time has passed
                if self._break_start_time:
                    recovery_time = current_time - self._break_start_time
                    if recovery_time >= self.config.recovery_time:
                        # Recovered - start new hold
                        self.state = PlankState.HOLDING
                        self._hold_start_time = current_time
                        self._break_start_time = None
            else:
                # Still broken, provide feedback
                if angle < self.config.angle_warning_threshold:
                    if angle < self.config.angle_warning_threshold - 20:
                        self._form_warnings.append("Body position way off - reset your plank")
                    else:
                        if hip_ok:
                            self._form_warnings.append("Straighten your body - hips too high/low")
                        else:
                            self._form_warnings.append("Fix your hip position")
    
    def _end_hold_session(self, end_time: float) -> None:
        """
        End current hold session and record metrics.
        
        Args:
            end_time: Time when hold ended
        """
        if self._hold_start_time is not None:
            duration = end_time - self._hold_start_time
            
            # Only count holds that meet minimum duration
            if duration >= self.config.min_hold_time:
                self.hold_count += 1
                
                # Calculate average angle during hold
                avg_angle = np.mean(self._angle_history) if self._angle_history else 0
                
                # Classify hold quality
                if duration >= self.config.excellent_hold_time:
                    quality = "excellent"
                elif duration >= self.config.good_hold_time:
                    quality = "good"
                else:
                    quality = "basic"
                
                # Store session data
                self._hold_sessions.append({
                    'session_number': len(self._hold_sessions) + 1,
                    'duration': duration,
                    'avg_body_angle': avg_angle,
                    'max_hip_deviation': self._max_hip_deviation,
                    'quality': quality,
                    'form_issues': self._form_warnings.copy() if self._form_warnings else []
                })
            
            # Reset tracking for next hold
            self._hold_start_time = None
            self._initial_hip_height = None
            self._hip_height_history = []
            self._max_hip_deviation = 0.0
            self._angle_history = []
    
    def reset(self) -> None:
        """Reset counter to initial state."""
        current_time = time.time()
        
        # End any ongoing hold session
        if self.state == PlankState.HOLDING and self._hold_start_time is not None:
            self._end_hold_session(current_time)
        
        self.state = PlankState.IDLE
        self.hold_count = 0
        self._last_body_angle = 0.0
        self._session_start_time = None
        self._hold_start_time = None
        self._break_start_time = None
        self._total_hold_time = 0.0
        self._current_hold_duration = 0.0
        self._angle_buffer.clear()
        self._angle_history.clear()
        self._hip_height_history.clear()
        self._initial_hip_height = None
        self._max_hip_deviation = 0.0
        self._form_warnings = []
        self._hold_sessions = []
        self._best_hold_duration = 0.0
        self._total_valid_frames = 0
        self._total_frames = 0
    
    def get_progress(self) -> float:
        """
        Get normalized progress of current hold (0.0 to 1.0).
        
        For plank, progress is based on hold duration relative to goals.
        
        Returns:
            Float between 0.0 and 1.0:
            0.0 = just started hold
            0.5 = reached min_hold_time
            1.0 = reached excellent_hold_time
        """
        if self.state != PlankState.HOLDING or self._hold_start_time is None:
            return 0.0
        
        current_duration = time.time() - self._hold_start_time
        
        # Progress to min hold time (0 to 0.5)
        if current_duration < self.config.min_hold_time:
            return (current_duration / self.config.min_hold_time) * 0.5
        
        # Progress from min to good (0.5 to 0.75)
        elif current_duration < self.config.good_hold_time:
            progress = (current_duration - self.config.min_hold_time) / (self.config.good_hold_time - self.config.min_hold_time)
            return 0.5 + (progress * 0.25)
        
        # Progress from good to excellent (0.75 to 1.0)
        elif current_duration < self.config.excellent_hold_time:
            progress = (current_duration - self.config.good_hold_time) / (self.config.excellent_hold_time - self.config.good_hold_time)
            return 0.75 + (progress * 0.25)
        
        # Beyond excellent
        return 1.0
    
    def get_feedback(self) -> Optional[str]:
        """
        Get form feedback based on current state and angles.
        
        Returns:
            String with feedback message or None if form is correct
        """
        # Return form warnings first (higher priority)
        if self._form_warnings:
            return self._form_warnings[0]
        
        angle = self._last_body_angle
        
        # State-specific feedback
        if self.state == PlankState.IDLE:
            return "Get into plank position (on elbows or hands)"
        
        if self.state == PlankState.HOLDING:
            # Provide timing feedback
            if self._hold_start_time:
                duration = time.time() - self._hold_start_time
                
                if duration < self.config.min_hold_time:
                    remaining = self.config.min_hold_time - duration
                    return f"Hold for {remaining:.1f} more seconds"
                elif duration < self.config.good_hold_time:
                    return "Good form! Keep holding"
                elif duration < self.config.excellent_hold_time:
                    return "Excellent! Push for the full minute"
                else:
                    return "Amazing hold! You're crushing it"
            
            # Form-specific feedback
            if angle < self.config.angle_threshold:
                if angle < self.config.angle_threshold - 10:
                    return "Straighten your body - don't sag"
                else:
                    return "Keep your body in a straight line"
            
            return None
        
        if self.state == PlankState.BROKEN:
            if angle < self.config.angle_warning_threshold - 20:
                return "Reset your position - body not straight"
            else:
                return "Fix your form and hold again"
        
        return None
    
    def get_current_hold_duration(self) -> float:
        """
        Get duration of current hold.
        
        Returns:
            Current hold duration in seconds (0 if not holding)
        """
        if self.state == PlankState.HOLDING and self._hold_start_time:
            return time.time() - self._hold_start_time
        return 0.0
    
    def get_total_hold_time(self) -> float:
        """
        Get total accumulated hold time for session.
        
        Returns:
            Total hold time in seconds
        """
        return self._total_hold_time
    
    def get_hold_sessions(self) -> List[Dict[str, Any]]:
        """
        Get data for all completed hold sessions.
        
        Returns:
            List of dictionaries with per-session metrics
        """
        return self._hold_sessions.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the session.
        
        Returns:
            Dictionary with session summary
        """
        if not self._hold_sessions:
            return {
                'exercise': 'plank',
                'total_holds': 0,
                'total_hold_time': round(self._total_hold_time, 1),
                'best_hold': round(self._best_hold_duration, 1),
                'avg_hold_duration': 0,
                'avg_body_angle': 0,
                'excellent_holds': 0,
                'good_holds': 0,
                'basic_holds': 0,
                'holds_with_issues': 0,
                'detection_quality': 0
            }
        
        # Count quality categories
        quality_counts = {'excellent': 0, 'good': 0, 'basic': 0}
        holds_with_issues = 0
        
        for session in self._hold_sessions:
            quality_counts[session['quality']] += 1
            if session['form_issues']:
                holds_with_issues += 1
        
        # Calculate averages
        avg_duration = np.mean([s['duration'] for s in self._hold_sessions])
        avg_angle = np.mean([s['avg_body_angle'] for s in self._hold_sessions])
        
        # Detection quality (percentage of frames with valid pose)
        detection_quality = (self._total_valid_frames / max(1, self._total_frames)) * 100
        
        return {
            'exercise': 'plank',
            'total_holds': len(self._hold_sessions),
            'total_hold_time': round(self._total_hold_time, 1),
            'best_hold': round(self._best_hold_duration, 1),
            'avg_hold_duration': round(avg_duration, 1),
            'avg_body_angle': round(avg_angle, 1),
            'excellent_holds': quality_counts['excellent'],
            'good_holds': quality_counts['good'],
            'basic_holds': quality_counts['basic'],
            'holds_with_issues': holds_with_issues,
            'detection_quality': round(detection_quality, 1)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export counter state to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'last_body_angle': self._last_body_angle,
            'current_hold_duration': self.get_current_hold_duration(),
            'total_hold_time': self._total_hold_time,
            'best_hold_duration': self._best_hold_duration,
            'max_hip_deviation': self._max_hip_deviation,
            'form_warnings': self._form_warnings,
            'config': self.config.to_dict(),
            'hold_sessions': self._hold_sessions
        })
        return base_dict