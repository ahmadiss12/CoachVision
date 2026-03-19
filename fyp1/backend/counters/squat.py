"""
Squat counter using a finite state machine with hysteresis.
Tracks knee angle to count repetitions and provide form feedback.
"""


from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np

from ..utils.geometry import calculate_angle, AngleBuffer
from .interface import ExerciseCounter, RepCounterConfig


class SquatState(Enum):
    """Squat FSM states.
    
    STATE MACHINE LOGIC:
    The squat counter uses a finite state machine (FSM) to robustly track repetitions.
    State flow for a complete squat:
        IDLE -> UP -> DESCENDING -> DOWN -> ASCENDING -> UP (count += 1)
    
    This approach ensures:
    - Reps are only counted when a complete squat is performed
    - Partial movements don't trigger false counts
    - States provide feedback on user's current position in the movement
    """
    IDLE = "IDLE"           # No valid squat pose detected (confidence too low)
    UP = "UP"               # Standing position (knees extended > extension_threshold)
    DESCENDING = "DESCENDING"  # In the descent phase (crossing below extension_threshold - buffer)
    DOWN = "DOWN"           # At bottom of squat (knees flexed below flexion_threshold)
    ASCENDING = "ASCENDING"    # In the ascent phase (crossing above flexion_threshold + buffer)


@dataclass
class SquatConfig(RepCounterConfig):
    """
    Configuration for squat counter.
    
    Extends base config with squat-specific thresholds.
    
    THRESHOLDS AND BUFFERS:
    These parameters control what is considered a valid squat and prevent false positive reps.
    
    Key Concepts:
    1. Angle Thresholds: Define the knee angle ranges for different squat phases
       - extension_threshold: Knee angle when standing up (typically 170°)
       - flexion_threshold: Knee angle at squat depth (typically 90° for parallel squat)
    
    2. Hysteresis Buffer: Prevents bouncing/false transitions near thresholds
       - When transitioning UP->DESCENDING, angle must drop below (extension - buffer)
       - When transitioning DOWN->ASCENDING, angle must rise above (flexion + buffer)
       - This "dead zone" of 10° prevents noise from triggering multiple transitions
       - Without buffers, slight angle fluctuations could cause state flip-flops
    
    3. Smoothing: Reduces jitter from pose estimation by averaging consecutive frames
       - angle_buffer_size determines how many frames to average
       - Larger buffer = smoother but more delayed response
    
    4. Quality Assessment: Distinguishes between good form and improper form
       - shallow_warning_angle: Below 100° is considered shallow (poor form)
       - deep_warning_angle: Below 80° is considered too deep (safety concern)
    """
    # ===== ANGLE THRESHOLDS (in degrees) =====
    extension_threshold: float = 150.0  # Knee angle when fully extended (standing up) - lowered for shallow squats
    flexion_threshold: float = 115.0    # Knee angle at bottom of squat (parallel or below) - increased for shallow squats
    
    # ===== HYSTERESIS BUFFER =====
    # Prevents false state transitions from pose estimation noise
    # When angle hovers near threshold, buffer requires stronger movement to transition
    buffer: float = 10.0  # Dead zone in degrees around thresholds
    
    # ===== SMOOTHING SETTINGS =====
    # Reduces jitter from pose estimation by temporal averaging
    angle_buffer_size: int = 3  # Number of frames to average (3 frame window)
    
    # ===== FEEDBACK THRESHOLDS =====
    # Used to classify rep quality and give user feedback
    shallow_warning_angle: float = 100.0  # Angle above this is too shallow (poor form)
    deep_warning_angle: float = 80.0      # Angle below this is too deep (safety concern)
    
    # ===== ANGLE COMBINATION METHOD =====
    # How to handle bilateral (two-sided) measurements
    use_both_legs: bool = True  # True: average both knee angles (more robust)
                                 # False: use minimum angle (more conservative)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Call parent's __post_init__ if it exists
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
            
        if self.extension_threshold <= self.flexion_threshold:
            raise ValueError("extension_threshold must be greater than flexion_threshold")
        if self.buffer < 0:
            raise ValueError("buffer must be non-negative")
        if self.angle_buffer_size < 1:
            raise ValueError("angle_buffer_size must be at least 1")


class SquatCounter(ExerciseCounter):
    """
    Squat counter using finite state machine with dual thresholds.
    
    Tracks knee angle through a complete squat cycle:
    UP -> DESCENDING -> DOWN -> ASCENDING -> UP (count increments)
    
    Uses hysteresis to prevent false counts when hovering near thresholds.
    """
    
    def __init__(self, config: Optional[SquatConfig] = None):
        """
        Initialize squat counter.
        
        Args:
            config: Configuration object (uses defaults if None)
        
        INTERNAL STATE TRACKING:
        The counter maintains multiple variables to track the current state and metrics.
        """
        self.config = config or SquatConfig()
        
        # ===== FSM STATE TRACKING =====
        self.state = SquatState.IDLE      # Current FSM state
        self.count = 0                    # Number of completed reps
        
        # ===== ANGLE AND POSITION TRACKING =====
        self._last_angle = 0.0            # Most recent knee angle (for feedback)
        self._min_angle_in_rep = float('inf')  # Minimum angle reached in current rep (deepest point)
        self._max_angle_in_rep = 0.0      # Maximum angle reached in current rep (highest point)
        self._rep_start_time: Optional[float] = None      # When current rep started (for duration)
        self._rep_end_time: Optional[float] = None        # When current rep completed
        self._rep_metrics: List[Dict[str, Any]] = []      # Historical data for all completed reps
        
        # ===== ANGLE SMOOTHING BUFFER =====
        # Reduces pose estimation jitter by applying temporal averaging
        self._angle_buffer = AngleBuffer(size=self.config.angle_buffer_size)
        
        # ===== PERFORMANCE/QUALITY TRACKING =====
        # Used to calculate detection quality percentage
        self._total_valid_frames = 0      # Frames where pose confidence was sufficient
        self._total_frames = 0            # Total frames processed
    
    @property
    def name(self) -> str:
        """Return the name of the exercise."""
        return "squat"
    
    @property
    def required_landmarks(self) -> list:
        """
        Return list of landmark names required for squat.
        
        Returns:
            List of strings matching the keys expected in update() landmarks dict
        """
        return [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle'
        ]
    
    def update(self, landmarks: Dict[str, Tuple[float, float]], confidence: float) -> Tuple[int, Enum, float]:
        """
        Update counter state with new frame data.
        
        Args:
            landmarks: Dictionary containing at least:
                - left_hip, left_knee, left_ankle (left leg 2D positions)
                - right_hip, right_knee, right_ankle (right leg 2D positions)
                Each landmark is a tuple of (x, y) coordinates in pixel/normalized space
            confidence: Overall confidence of pose detection (0-1 scale)
            
        Returns:
            Tuple of (count, state, knee_angle)
        
        ANGLE CALCULATION PROCESS:
        1. Extract 3D joint positions from landmarks dictionary
        2. Calculate knee angle for each leg using geometry helpers
           - Angle is formed by: hip -> knee <- ankle
           - 180° = fully extended, 90° = bent to parallel
        3. Combine both leg angles (average or minimum) for robustness
        4. Apply temporal smoothing to reduce jitter
        5. Update FSM state based on smoothed angle
        """
        self._total_frames += 1
        
        # Reset if confidence is too low (lost pose tracking)
        if confidence < self.config.min_confidence:
            if self.state != SquatState.IDLE:
                self.state = SquatState.IDLE
            return self.count, self.state, self._last_angle
        
        # ===== EXTRACT LANDMARKS =====
        # Get x,y coordinates for both legs from pose estimator
        try:
            left_hip = landmarks['left_hip']
            left_knee = landmarks['left_knee']
            left_ankle = landmarks['left_ankle']
            right_hip = landmarks['right_hip']
            right_knee = landmarks['right_knee']
            right_ankle = landmarks['right_ankle']
        except KeyError as e:
            raise ValueError(f"Missing required landmark: {e}")
        
        # ===== CALCULATE KNEE ANGLES =====
        # Use geometry helper to compute angle at knee joint
        # Angle = arccos((BA · AC) / (|BA| * |AC|))
        # where B=hip, A=knee, C=ankle (angle at A)
        left_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # ===== COMBINE BILATERAL MEASUREMENTS =====
        # Handle both sides symmetrically for better robustness
        if self.config.use_both_legs:
            # Average both sides: robust to single-side errors
            knee_angle = left_angle 
        else:
            # Use minimum angle: conservative (assumes worst form)
            # One knee bending less = incomplete movement
            knee_angle = min(left_angle, right_angle)
        
        # ===== TEMPORAL SMOOTHING =====
        # Reduce jitter from pose estimation by averaging over recent frames
        # Moving average window reduces noise while preserving peak angles
        self._angle_buffer.add(knee_angle)
        smoothed_angle = self._angle_buffer.get_smoothed()
        if smoothed_angle is not None:
            knee_angle = smoothed_angle
        
        self._last_angle = knee_angle
        
        # ===== UPDATE STATE MACHINE =====
        # Feed smoothed angle into FSM to determine state and increment rep count
        self._update_state(knee_angle)
        
        return self.count, self.state, knee_angle
    
    def _update_state(self, angle: float) -> None:
        """
        Update FSM based on current knee angle.
        
        STATE MACHINE LOGIC - DETAILED EXPLANATION:
        
        The FSM uses angle thresholds with hysteresis (buffer) to create stable state transitions.
        This prevents bouncing between states due to measurement noise.
        
        STATE TRANSITIONS:
        
        IDLE State (lost pose or low confidence):
            - Transition to UP when: angle > extension_threshold (170°)
            - Used to: Wait for valid pose detection before tracking
        
        UP State (standing, knees extended):
            - Transition to DESCENDING when: angle < extension_threshold - buffer (170 - 10 = 160°)
            - The buffer ensures the person is actually descending, not just temporary noise
            - Tracks rep start time and initializes min/max angle trackers
        
        DESCENDING State (moving down, bending knees):
            - Transition to DOWN when: angle < flexion_threshold (90°)
            - Transition back to UP if: angle > extension_threshold (user stands back up)
            - If transitioning back UP: discard this rep (incomplete descent)
            - Tracks minimum angle achieved during descent
        
        DOWN State (bottom of squat, knees maximally flexed):
            - Transition to ASCENDING when: angle > flexion_threshold + buffer (90 + 10 = 100°)
            - The buffer prevents false state changes if person pauses at bottom
            - Continues tracking minimum angle
        
        ASCENDING State (moving up, extending knees):
            - Transition to UP when: angle > extension_threshold (170°)
              * At this point: INCREMENT REP COUNT (one complete squat cycle!)
              * Store rep metrics (depth, duration, quality)
            - Transition back to DOWN if: angle < flexion_threshold (user squats again)
            - Tracks maximum angle during ascent
        
        WHY BUFFERS ARE ESSENTIAL:
        Without buffers, the FSM would oscillate between states if angle hovers near a threshold.
        Example without buffer:
            - Person at 165° (near 170° extension threshold)
            - Measurement noise: 168° -> 172° -> 168° -> 172°
            - FSM would toggle: UP -> DESCENDING -> UP -> DESCENDING (false history!)
        
        With 10° buffer:
            - Must drop below 160° to transition DOWN (more deliberate)
            - Must rise above 100° to transition UP again (clear commitment)
            - Noise around 165° doesn't cause false transitions
        
        Args:
            angle: Current smoothed knee angle
        """
        cfg = self.config
        
        # ===== STATE: IDLE =====
        if self.state == SquatState.IDLE:
            # Wait for person to stand up before tracking
            if angle >= cfg.extension_threshold:
                self.state = SquatState.UP
                self._total_valid_frames += 1
        
        # ===== STATE: UP (Standing) =====
        elif self.state == SquatState.UP:
            # Person starts descending (crosses below extension threshold minus buffer)
            # Buffer ensures this is intentional descent, not noise
            if angle < cfg.extension_threshold - cfg.buffer:
                self.state = SquatState.DESCENDING
                self._rep_start_time = self._rep_start_time or time.time()  # Mark rep start
                self._min_angle_in_rep = angle  # Initialize minimum angle
                self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
        
        # ===== STATE: DESCENDING (Moving Down) =====
        elif self.state == SquatState.DESCENDING:
            # Track deepest point reached
            self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            # Reached bottom of squat (valid depth)
            if angle <= cfg.flexion_threshold:
                self.state = SquatState.DOWN
            
            # User stood back up without reaching bottom (incomplete rep)
            # Discard this incomplete descent
            elif angle >= cfg.extension_threshold:
                self.state = SquatState.UP
                self._reset_rep_tracking()  # Throw away this attempt
        
        # ===== STATE: DOWN (Bottom of Squat) =====
        elif self.state == SquatState.DOWN:
            # Continue tracking depth extremes while paused at bottom
            self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            # Person starts ascending (crosses above flexion threshold plus buffer)
            # Buffer prevents false ascent if person momentarily relaxes at bottom
            if angle > cfg.flexion_threshold + cfg.buffer:
                self.state = SquatState.ASCENDING
        
        # ===== STATE: ASCENDING (Moving Up) =====
        elif self.state == SquatState.ASCENDING:
            # Track upward motion
            self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
            
            # Person returned to standing position - REP COMPLETED!
            if angle >= cfg.extension_threshold:
                self.count += 1  # *** INCREMENT REP COUNT ***
                self.state = SquatState.UP
                self._rep_end_time = time.time()  # Mark rep completion time
                self._store_rep_metrics()  # Save metrics for this rep
                self._reset_rep_tracking()  # Prepare for next rep
            
            # User descended again before completing rep
            # This is valid (multiple partial reps can count)
            elif angle <= cfg.flexion_threshold:
                self.state = SquatState.DOWN
        
        self._total_valid_frames += 1
    
    def _store_rep_metrics(self) -> None:
        """
        Store metrics for the completed repetition.
        
        FEEDBACK AND METRICS: Collecting Rep Data
        
        This method is called when a rep is completed and saves comprehensive data about the rep.
        This data is used for:
        - Providing immediate feedback to the user about form quality
        - Analyzing workout patterns and progress over time
        - Generating session summaries and statistics
        
        Metrics Collected Per Rep:
        
        1. rep_number: Sequential number of this rep in the session
           - Used for tracking progression and ordering reps chronologically
        
        2. min_angle: Deepest knee angle achieved during the rep
           - Lower values = deeper squat = better form (typically)
           - Used to assess squat depth vs. configured thresholds
        
        3. max_angle: Highest knee angle achieved during the rep
           - Should be > extension_threshold (170°) for complete rep
           - Indicates how fully user extended at the top
        
        4. range_of_motion (ROM): max_angle - min_angle
           - Measures full movement range in this rep
           - Consistent ROM indicates stable, controlled form
           - Decreasing ROM over time may indicate fatigue
        
        5. duration: Time from start of rep to completion (in seconds)
           - Slow reps (~2-3 sec) indicate controlled tempo
           - Very fast reps may indicate bouncing or poor control
           - Shows workout intensity and pace
        
        6. depth_quality: Classification of squat depth
           - "good": Reached or passed parallel (min_angle ≤ flexion_threshold)
           - "shallow": Between parallel and warning angle (100°)
           - "very_shallow": Above 100° angle (poor depth)
        """
        if self._rep_start_time and self._rep_end_time:
            duration = self._rep_end_time - self._rep_start_time
        else:
            duration = 0.0
        
        self._rep_metrics.append({
            'rep_number': len(self._rep_metrics) + 1,
            'min_angle': self._min_angle_in_rep,
            'max_angle': self._max_angle_in_rep,
            'range_of_motion': self._max_angle_in_rep - self._min_angle_in_rep,
            'duration': duration,
            'depth_quality': self._classify_depth_quality()
        })
    
    def _reset_rep_tracking(self) -> None:
        """
        Reset tracking variables for a new rep.
        
        Called after every completed rep to prepare for the next one.
        Clears all per-rep state variables to their initial values.
        """
        self._min_angle_in_rep = float('inf')  # Ready to track new minimum
        self._max_angle_in_rep = 0.0            # Ready to track new maximum
        self._rep_start_time = None             # Clear previous rep timing
        self._rep_end_time = None
    
    def _classify_depth_quality(self) -> str:
        """
        Classify the quality of squat depth.
        
        FEEDBACK: Form Quality Classification
        
        Compares the deepest knee angle achieved to configured thresholds
        to determine if the squat met depth requirements.
        
        Return Values:
        
        "good": min_angle ≤ flexion_threshold (≤ 90°)
            - User reached or exceeded parallel (parallel squat)
            - Indicates full range of motion and good form
            - Most effective for strength/hypertrophy development
        
        "shallow": shallow_warning_angle (100°) ≥ min_angle > flexion_threshold
            - User descended but didn't reach parallel
            - Between 90° and 100°
            - Less effective ROM but still useful work
            - Feedback: "Go deeper for better activation"
        
        "very_shallow": min_angle > shallow_warning_angle (> 100°)
            - Minimal depth, barely descended
            - Above 100° angle
            - Minimal ROM, may not count as full rep in competitive lifting
            - Feedback: "Squat deeper for full range"
        """
        if self._min_angle_in_rep <= self.config.flexion_threshold:
            return "good"
        elif self._min_angle_in_rep <= self.config.shallow_warning_angle:
            return "shallow"
        else:
            return "very_shallow"
    
    def reset(self) -> None:
        """
        Reset counter to initial state.
        
        RESET AND EXPORT: Resetting the Counter
        
        This method clears all tracking data and returns the counter to its initial state.
        Useful for starting a new workout session or clearing previous workout data.
        
        Resets:
        1. FSM State -> IDLE (wait for pose detection)
        2. Rep Count -> 0 (start counting from zero)
        3. Current Angle Tracking -> 0 (clear last measured angle)
        4. Angle Buffer -> clear (remove smoothing history)
        5. Rep Metrics -> empty list (clear all history of completed reps)
        6. Frame Counters -> 0 (reset detection quality calculation)
        
        After calling reset(), the counter is ready for a fresh workout session.
        All historical data (rep metrics) is lost - call get_summary() first if needed.
        """
        self.state = SquatState.IDLE
        self.count = 0
        self._last_angle = 0.0
        self._angle_buffer.clear()
        self._reset_rep_tracking()
        self._rep_metrics = []
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
        angle = self._last_angle
        
        if self.state == SquatState.IDLE or self.state == SquatState.UP:
            return 0.0
        
        elif self.state == SquatState.DESCENDING:
            # Progress from 0.0 to 0.5 as angle decreases
            span = cfg.extension_threshold - cfg.flexion_threshold
            if span <= 0:
                return 0.25
            prog = (cfg.extension_threshold - angle) / span
            return min(0.5, max(0.0, prog * 0.5))
        
        elif self.state == SquatState.DOWN:
            return 0.5
        
        elif self.state == SquatState.ASCENDING:
            # Progress from 0.5 to 1.0 as angle increases
            span = cfg.extension_threshold - cfg.flexion_threshold
            if span <= 0:
                return 0.75
            prog = (angle - cfg.flexion_threshold) / span
            return min(1.0, max(0.5, 0.5 + prog * 0.5))
        
        return 0.0
    
    def get_feedback(self) -> Optional[str]:
        """
        Get form feedback based on current state and angle.
        
        FEEDBACK AND METRICS: Real-Time Form Feedback
        
        This method provides context-aware, real-time feedback to guide the user
        through proper squat form. Feedback is state-specific and angle-specific:
        
        IDLE State:
            - Message: "Stand in frame to begin"
            - User hasn't been detected yet or confidence dropped
            - Action: Have user stand and ensure visible in camera
        
        UP State (standing, ready to squat):
            - Message: "Ready to squat" (when angle drops below threshold minus buffer)
            - None: When already standing idle
            - Action: User can start descending
        
        DESCENDING State (moving downward):
            - "Bend your knees more" (if angle > extension - 15°, stiff knees)
            - "Almost there - go deeper" (if angle < flexion + 10°, near bottom)
            - None: During smooth descent at proper speed
        
        DOWN State (at bottom of squat):
            - "Go a bit deeper" (if angle > flexion + buffer, not deep enough)
            - "Don't go too deep" (if angle < deep_warning_angle, 80°, too low)
            - "Good depth - now push up" (proper depth, prompt to ascend)
        
        ASCENDING State (moving upward):
            - "Push through your heels" (if angle < flexion + 15°, still descending)
            - "Almost up - lockout" (if angle > extension - 10°, nearly locked)
            - None: During smooth controlled ascent
        
        Returns:
            String message for user or None if form is correct
        """
        angle = self._last_angle
        cfg = self.config
        
        # No feedback in IDLE state
        if self.state == SquatState.IDLE:
            return "Stand in frame to begin"
        
        # Feedback for UP state
        if self.state == SquatState.UP:
            if angle < cfg.extension_threshold - cfg.buffer:
                return "Ready to squat"
            return None
        
        # Feedback for DESCENDING
        if self.state == SquatState.DESCENDING:
            if angle < cfg.flexion_threshold + 10:
                return "Almost there - go deeper"
            elif angle > cfg.extension_threshold - 15:
                return "Bend your knees more"
            return None
        
        # Feedback for DOWN
        if self.state == SquatState.DOWN:
            if angle > cfg.flexion_threshold + cfg.buffer:
                return "Go a bit deeper"
            elif angle < cfg.deep_warning_angle:
                return "Don't go too deep"
            return "Good depth - now push up"
        
        # Feedback for ASCENDING
        if self.state == SquatState.ASCENDING:
            if angle < cfg.flexion_threshold + 15:
                return "Push through your heels"
            elif angle > cfg.extension_threshold - 10:
                return "Almost up - lockout"
            return None
        
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
                'exercise': 'squat',
                'total_reps': 0,
                'avg_range_of_motion': 0,
                'avg_duration': 0,
                'good_reps': 0,
                'shallow_reps': 0,
                'very_shallow_reps': 0,
                'detection_quality': 0
            }
        
        # Count quality categories
        quality_counts = {'good': 0, 'shallow': 0, 'very_shallow': 0}
        for rep in self._rep_metrics:
            quality_counts[rep['depth_quality']] += 1
        
        # Calculate averages
        avg_rom = np.mean([r['range_of_motion'] for r in self._rep_metrics])
        avg_duration = np.mean([r['duration'] for r in self._rep_metrics])
        
        # Detection quality (percentage of frames with valid pose)
        detection_quality = (self._total_valid_frames / max(1, self._total_frames)) * 100
        
        return {
            'exercise': 'squat',
            'total_reps': len(self._rep_metrics),
            'avg_range_of_motion': round(avg_rom, 1),
            'avg_duration': round(avg_duration, 2),
            'good_reps': quality_counts['good'],
            'shallow_reps': quality_counts['shallow'],
            'very_shallow_reps': quality_counts['very_shallow'],
            'detection_quality': round(detection_quality, 1)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export counter state to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'min_angle_in_rep': self._min_angle_in_rep if self._min_angle_in_rep != float('inf') else None,
            'max_angle_in_rep': self._max_angle_in_rep,
            'config': self.config.to_dict(),
            'rep_metrics': self._rep_metrics
        })
        return base_dict 