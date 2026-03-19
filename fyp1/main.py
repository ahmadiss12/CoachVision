"""
Main pipeline for Intelligent Exercise Form Correction System.
Integrates MediaPipe pose detection with exercise-specific counters.
Provides real-time visualization and feedback.
"""

import cv2
import time
import argparse
import os
from typing import Tuple, Optional, Dict, List
import numpy as np
from enum import Enum

# MediaPipe imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Local imports
from backend.counters.dispatcher import (
    ExerciseDispatcher, ExerciseType, set_exercise, 
    update_counter, get_feedback, reset_counter,
    CONFIG_PRESETS
)
from backend.utils.geometry import convert_to_pixel_coordinates, LandmarkFilter
from backend.utils.one_euro import OneEuroFilter
from backend.voice import FeedbackPolicy, VoiceCoach, FeedbackLabel
from backend.squat_form_analyzer import SquatFormAnalyzer


class DisplayMode(Enum):
    """Display modes for visualization."""
    FULL = "full"           # Show all info
    MINIMAL = "minimal"     # Show only essential info
    DEBUG = "debug"         # Show technical details


class ExerciseApp:
    """
    Main application class for exercise monitoring.
    Handles video capture, pose detection, and visualization.
     Usage:
         app = ExerciseApp(exercise_name='squat', difficulty='beginner')
         app.run()
       """
    
    def __init__(self, 
                 exercise_name: str = "squat",
                 difficulty: str = "intermediate",
                 video_source: Optional[str] = None,
                 camera_id: int = 0,
                 display_mode: str = "full",
                 save_output: bool = False,
                 show_fps: bool = True):
        """
        Initialize the exercise application.
        
        Args:
            exercise_name: Name of exercise to perform
            difficulty: Beginner/intermediate/advanced
            video_source: Path to video file (None for webcam)
            camera_id: Camera ID for webcam
            display_mode: Display mode (full/minimal/debug)
            save_output: Whether to save output video
            show_fps: Whether to display FPS counter
            """
        self.exercise_name = exercise_name
        self.difficulty = difficulty
        self.display_mode = DisplayMode(display_mode)
        self.save_output = save_output
        self.show_fps = show_fps
        
        # Initialize video capture
        if video_source:
            self.cap = cv2.VideoCapture(video_source)
            self.is_video_file = True
        else:
            self.cap = cv2.VideoCapture(camera_id)
            self.is_video_file = False
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video source")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:  # Webcam might return 0
            self.fps = 30
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[VIDEO] Video source: {self.width}x{self.height}, {self.fps} FPS")
        
        # Initialize MediaPipe Pose using Tasks API
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "backend", "pose_landmarker.task")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. "
                "Download from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            )
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        self.frame_count = 0  # For video timestamp tracking
        
        # Define pose connections for skeleton drawing (MediaPipe standard)
        self.pose_connections = [
            (11, 13), (13, 15),  # Right arm
            (12, 14), (14, 16),  # Left arm
            (11, 12),            # Shoulders
            (5, 6),              # Face
            (5, 7), (7, 9),      # Left arm (face coords)
            (6, 8), (8, 10),     # Right arm (face coords)
            (11, 23), (12, 24),  # Hips
            (23, 24),            # Hip line
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
        ]
        
        # Define important landmarks for extraction
        landmark_indices = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_foot_index': 31, 'right_foot_index': 32,
        }
        self.important_landmarks = landmark_indices
        
        # Initialize landmark filters for smoothing
        self.landmark_filters = {}
        for name in self.important_landmarks.keys():
            self.landmark_filters[name] = LandmarkFilter(min_cutoff=0.5, beta=0.01)
        
        # Initialize exercise counter via dispatcher
        try:
            set_exercise(exercise_name, level=difficulty)
            print(f"[OK] Initialized {exercise_name} counter ({difficulty} level)")
        except Exception as e:
            print(f"[ERROR] Failed to initialize exercise: {e}")
            raise
        # Initialize form classifier
        self.form_analyzer = SquatFormAnalyzer()
        print("[OK] Form analyzer loaded")
        # Voice: single-label policy + coach (no queue, instant stop)
        self._last_voice_label = None
        self._last_voice_time = 0.0
        self.voice_coach = None
        self.voice_policy = None
        try:
            self.voice_policy = FeedbackPolicy(exercise_name, difficulty)
            self.voice_coach = VoiceCoach(exercise_name, difficulty)
            self.voice_coach.pregenerate()
            print("[OK] Voice feedback initialized (single-slot, no delay)")
        except Exception as e:
            print(f"[WARNING] Voice feedback unavailable: {e}")
        
        # Get exercise info for display
        self.exercise_info = ExerciseDispatcher().get_exercise_info(exercise_name)
        
        # Performance tracking
        self.frame_count = 0
        self.prev_time = time.time()
        self.fps_history = []
        
        # Visualization settings
        self.colors = {
            'good': (0, 255, 0),      # Green
            'warning': (0, 255, 255),  # Yellow
            'bad': (0, 0, 255),        # Red
            'neutral': (255, 255, 255), # White
            'info': (255, 200, 0),      # Orange
            'bg': (0, 0, 0),            # Black
            'skeleton': (255, 255, 255) # White skeleton
        }
        
        # UI layout
        self.ui_margin = 10
        self.line_height = 25
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_PLAIN
        
        # Initialize video writer if saving
        if self.save_output:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"output_{exercise_name}_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
            print(f"💾 Saving output to: {output_path}")
    
    def extract_landmarks(self, landmarks) -> Dict[str, Tuple[float, float]]:
        """
        Extract and filter landmarks from MediaPipe results.
        
        Args:
            landmarks: List of MediaPipe NormalizedLandmark objects
            
        Returns:
            Dictionary mapping landmark names to (x, y) coordinates
        """
        result = {}
        timestamp = time.time()
        
        for name, idx in self.important_landmarks.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                # Apply filtering
                filtered = self.landmark_filters[name].filter(lm.x, lm.y, timestamp)
                result[name] = (filtered[0], filtered[1])
        
        return result
    
    def draw_skeleton(self, frame: np.ndarray, landmarks) -> None:
        """
        Draw pose skeleton on frame.
        
        Args:
            frame: Video frame
            landmarks: List of MediaPipe NormalizedLandmark objects
        """
        # Draw skeleton connections
        for connection in self.pose_connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                # Only draw if both landmarks are reasonably visible
                if start.presence > 0.5 and end.presence > 0.5:
                    start_pos = (int(start.x * self.width), int(start.y * self.height))
                    end_pos = (int(end.x * self.width), int(end.y * self.height))
                    cv2.line(frame, start_pos, end_pos, (255, 255, 255), 2)
        
        # Draw landmarks as circles
        for landmark in landmarks:
            if landmark.presence > 0.5:
                pos = (int(landmark.x * self.width), int(landmark.y * self.height))
                cv2.circle(frame, pos, 3, (0, 255, 0), -1)
    
    def draw_info_panel(self, 
                       frame: np.ndarray,
                       count: int,
                       state: Enum,
                       angle: float,
                       feedback: Optional[str],
                       progress: float) -> None:
        """
        Draw information panel on frame.
        
        Args:
            frame: Video frame
            count: Current rep count / hold count
            state: Current FSM state
            angle: Primary angle being tracked
            feedback: Form feedback message
            progress: Progress of current rep/hold
        """
        h, w = frame.shape[:2]
        y_offset = self.ui_margin
        
        # Background panel (semi-transparent)
        panel_width = 300
        panel_height = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (w - panel_width - self.ui_margin, self.ui_margin),
                     (w - self.ui_margin, panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Exercise title
        title = f"{self.exercise_name.upper()} - {self.difficulty}"
        cv2.putText(frame, title, 
                   (w - panel_width + self.ui_margin, y_offset + 20),
                   self.font, 0.7, self.colors['info'], 2)
        y_offset += 30
        
        # Rep count
        if self.exercise_info.get('type') == 'static':
            count_text = f"Hold Sessions: {count}"
        else:
            count_text = f"Reps: {count}"
        cv2.putText(frame, count_text,
                   (w - panel_width + self.ui_margin, y_offset + 20),
                   self.font, 0.6, self.colors['neutral'], 1)
        y_offset += 25
        
        # State
        state_text = f"State: {state.value}"
        cv2.putText(frame, state_text,
                   (w - panel_width + self.ui_margin, y_offset + 20),
                   self.font, 0.6, self.colors['neutral'], 1)
        y_offset += 25
        
        # Angle
        angle_text = f"Angle: {angle:.1f}°"
        cv2.putText(frame, angle_text,
                   (w - panel_width + self.ui_margin, y_offset + 20),
                   self.font, 0.6, self.colors['neutral'], 1)
        y_offset += 30
        
        # Progress bar
        bar_x = w - panel_width + self.ui_margin
        bar_y = y_offset
        bar_width = panel_width - 2 * self.ui_margin
        bar_height = 15
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        
        # Progress fill
        fill_width = int(bar_width * progress)
        if progress < 0.5:
            color = self.colors['warning']
        else:
            color = self.colors['good']
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + fill_width, bar_y + bar_height),
                     color, -1)
        
        # Progress text
        progress_text = f"{progress*100:.0f}%"
        cv2.putText(frame, progress_text,
                   (bar_x + bar_width//2 - 20, bar_y - 5),
                   self.font_small, 1, self.colors['neutral'], 1)
        y_offset += 30
        
        # Feedback message (highlighted)
        if feedback:
            # Create colored background for feedback
            fb_y = h - 60
            cv2.rectangle(frame, 
                         (self.ui_margin, fb_y - 5),
                         (w - self.ui_margin, fb_y + 25),
                         (0, 0, 0), -1)
            
            # Choose color based on feedback severity
            if "!" in feedback or "don't" in feedback.lower():
                fb_color = self.colors['bad']
            elif "good" in feedback.lower() or "great" in feedback.lower():
                fb_color = self.colors['good']
            else:
                fb_color = self.colors['warning']
            
            cv2.putText(frame, f"💪 {feedback}", 
                       (self.ui_margin + 10, fb_y + 15),
                       self.font, 0.7, fb_color, 2)
    
    def draw_debug_info(self, frame: np.ndarray) -> None:
        """
        Draw debug information (FPS, frame count, etc.).
        
        Args:
            frame: Video frame
        """
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        self.prev_time = current_time
        
        # Frame info
        frame_text = f"Frame: {self.frame_count}"
        if self.total_frames > 0:
            frame_text += f"/{self.total_frames}"
        
        # FPS
        fps_text = f"FPS: {avg_fps:.1f}"
        
        # Draw in top-left corner
        cv2.putText(frame, fps_text, (10, 30),
                   self.font_small, 1, self.colors['info'], 1)
        cv2.putText(frame, frame_text, (10, 55),
                   self.font_small, 1, self.colors['info'], 1)
        
        # Exercise type
        ex_type = self.exercise_info.get('type', 'dynamic')
        cv2.putText(frame, f"Type: {ex_type}", (10, 80),
                   self.font_small, 1, self.colors['info'], 1)
    
    def draw_minimal_info(self, frame: np.ndarray, count: int, feedback: Optional[str]) -> None:
        """
        Draw minimal information (clean view).
        
        Args:
            frame: Video frame
            count: Current rep count
            feedback: Form feedback
        """
        h, w = frame.shape[:2]
        
        # Large rep count in top-left
        count_text = f"Reps: {count}"
        cv2.putText(frame, count_text, (30, 70),
                   self.font, 2, self.colors['neutral'], 3)
        
        # Feedback at bottom
        if feedback:
            cv2.putText(frame, feedback, (30, h - 50),
                       self.font, 1, self.colors['warning'], 2)
    
    def run(self) -> None:
        """Main application loop."""
        print(f">>> Starting {self.exercise_name} monitoring...")
        print("Controls: 'q' to quit | 'r' to reset | 'm' for display mode | 'v' to toggle voice")
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Prepare image for MediaPipe - convert to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                
                # Get timestamp in milliseconds for video mode
                timestamp_ms = int(self.frame_count / self.fps * 1000)
                
                # Detect pose landmarks
                detection_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                
                # Create a copy for drawing
                display_frame = frame.copy()
                
                if detection_result.pose_landmarks:
                    # Extract landmarks using the correct method
                    landmarks_dict = self.extract_landmarks(detection_result.pose_landmarks[0])
                    
                    # Draw skeleton
                    self.draw_skeleton(display_frame, detection_result.pose_landmarks[0])
                    
                    # Get overall confidence (average of landmark presences)
                    confidences = [lm.presence for lm in detection_result.pose_landmarks[0]]
                    confidence = np.mean(confidences) if confidences else 0
                    
                    # Update counter
                    count, state, angle = update_counter(landmarks_dict, confidence)
                    feedback = get_feedback()
                    progress = ExerciseDispatcher().get_progress()
                    
                    # Analyze squat form (new classifier)
                    form_name = None
                    if self.form_analyzer and self.form_analyzer.model:
                        try:
                            # Convert MediaPipe landmarks to numpy array format
                            landmarks_array = np.array([
                                [lm.x, lm.y, lm.z if hasattr(lm, 'z') else 0]
                                for lm in detection_result.pose_landmarks[0]
                            ])
                            
                            analysis = self.form_analyzer.analyze(landmarks_array)
                            if analysis.get('confidence', 0) > 0.7:  # Only if confident
                                form_name = analysis.get('form_name', 'unknown')
                                form_confidence = analysis.get('confidence', 0)
                                
                                # Only print periodically to avoid spam
                                if self.frame_count % 50 == 0:
                                    print(f"[FORM] {form_name} ({form_confidence:.0%})")
                        except Exception as e:
                            if self.frame_count % 100 == 0:
                                print(f"[FORM] Error: {e}")
                    
                    # DEBUG: Print counter updates
                    if self.frame_count % 50 == 0:
                        print(f"[FRAME {self.frame_count}] Count: {count}, State: {state.value}, Angle: {angle:.1f}°")
                    
                    # Voice: one label per frame, cooldown 1s
                    if self.voice_coach and self.voice_policy:
                        label = self.voice_policy.decide(count, state, angle, form_name=form_name)
                        if label != FeedbackLabel.NONE:
                            now = time.time()
                            if label != self._last_voice_label or (now - self._last_voice_time) >= 1.0:
                                self.voice_coach.say(label)
                                self._last_voice_label = label
                                self._last_voice_time = now
                    
                    # Draw based on display mode
                    if self.display_mode == DisplayMode.FULL:
                        self.draw_info_panel(display_frame, count, state, angle, feedback, progress)
                        self.draw_debug_info(display_frame)
                    elif self.display_mode == DisplayMode.MINIMAL:
                        self.draw_minimal_info(display_frame, count, feedback)
                    else:  # DEBUG mode
                        self.draw_info_panel(display_frame, count, state, angle, feedback, progress)
                        self.draw_debug_info(display_frame)
                        
                        # Draw additional debug info
                        y_offset = 150
                        for name, (x, y) in list(landmarks_dict.items())[:5]:  # Show first 5 landmarks
                            pixel_x, pixel_y = convert_to_pixel_coordinates((x, y), self.width, self.height)
                            cv2.putText(display_frame, f"{name}: ({pixel_x},{pixel_y})", 
                                      (10, y_offset), self.font_small, 0.5, self.colors['info'], 1)
                            y_offset += 20
                else:
                    # No pose detected
                    cv2.putText(display_frame, "No person detected", 
                               (self.width//2 - 100, self.height//2),
                               self.font, 1, self.colors['bad'], 2)
                    
                    # Still update debug info
                    if self.display_mode != DisplayMode.MINIMAL:
                        self.draw_debug_info(display_frame)
                
                # Show frame
                cv2.imshow('Exercise Form Correction', display_frame)
                
                # Save frame if requested
                if self.save_output:
                    self.out.write(display_frame)
                
                # Handle key presses
                # For video files, add frame delay based on FPS; for webcam use short delay
                delay_ms = int(100 / self.fps) if self.is_video_file else 1
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord('q'):
                    print("👋 Quitting...")
                    break
                elif key == ord('r'):
                    reset_counter()
                    print("🔄 Counter reset")
                elif key == ord('m'):
                    # Cycle through display modes
                    modes = list(DisplayMode)
                    current_idx = modes.index(self.display_mode)
                    next_idx = (current_idx + 1) % len(modes)
                    self.display_mode = modes[next_idx]
                    print(f"📺 Display mode: {self.display_mode.value}")
                elif key == ord('v'):
                    if self.voice_coach:
                        new_volume = 0.0 if self.voice_coach.volume > 0.4 else 0.8
                        self.voice_coach.set_volume(new_volume)
                        print(f"Voice: {'On' if new_volume > 0 else 'Off'}")
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.voice_coach:
            self.voice_coach.stop()
        
        self.cap.release()
        if self.save_output:
            self.out.release()
        cv2.destroyAllWindows()
        
        # Print session summary
        summary = ExerciseDispatcher().get_summary()
        if summary:
            print("\n[SUMMARY] Session Summary:")
            for key, value in summary.items():
                if key != 'exercise':
                    print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("[OK] Application closed")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Exercise Form Correction System')
    
    parser.add_argument('--exercise', '-e', type=str, default='squat',
                       choices=['squat', 'pushup', 'lunge', 'deadlift', 'plank',
                               'bicep_curl', 'shoulder_press', 'situp', 
                               'jumping_jack', 'high_knees', 'mountain_climber', 'wall_sit'],
                       help='Exercise to monitor')
    
    parser.add_argument('--difficulty', '-d', type=str, default='intermediate',
                       choices=['beginner', 'intermediate', 'advanced'],
                       help='Difficulty level')
    
    parser.add_argument('--video', '-v', type=str, default=None,
                       help='Path to video file (default: use webcam)')
    
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera ID for webcam')
    
    parser.add_argument('--display', type=str, default='full',
                       choices=['full', 'minimal', 'debug'],
                       help='Display mode')
    
    parser.add_argument('--save', '-s', action='store_true',
                       help='Save output video')
    
    parser.add_argument('--no-fps', action='store_true',
                       help='Hide FPS counter')
    
    return parser.parse_args()


def list_exercises():
    """Print list of available exercises."""
    exercises = ExerciseDispatcher().list_available_exercises()
    print("\n📋 Available Exercises:")
    for ex in exercises:
        info = ExerciseDispatcher().get_exercise_info(ex)
        ex_type = info.get('type', 'dynamic')
        muscles = info.get('target_muscles', [])
        muscle_str = ', '.join(muscles[:3]) + ('...' if len(muscles) > 3 else '')
        print(f"  • {ex}: {ex_type} - targets {muscle_str}")


if __name__ == "__main__":
    args = parse_arguments()
    
    # Show available exercises if requested
    if args.exercise == 'list':
        list_exercises()
        exit(0)
    
    # Run application
    app = ExerciseApp(
        exercise_name=args.exercise,
        difficulty=args.difficulty,
        video_source=args.video,
        camera_id=args.camera,
        display_mode=args.display,
        save_output=args.save,
        show_fps=not args.no_fps
    )
    
    app.run()