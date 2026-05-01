#!/usr/bin/env python3
"""Debug script to test squat counter with 2.mp4 video."""

import cv2
import numpy as np
from backend.counters.dispatcher import set_exercise, update_counter, reset_counter, CONFIG_PRESETS
from backend.utils.geometry import convert_to_pixel_coordinates
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
from mediapipe import ImageFormat
import os

# Initialize MediaPipe pose detection
task_path = "backend/pose_landmarker.task"
base_options = mp.tasks.BaseOptions(model_asset_path=task_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=False
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# Open video
video_path = "2.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Setup counter
set_exercise('squat', 'beginner')

frame_idx = 0
angles = []
states = []
counts = []

print(f"[VIDEO] {video_path}: {frame_count_total} frames at {fps} FPS")
print(f"\n[SQUAT CONFIG]")
print(f"  extension_threshold: 150°")
print(f"  flexion_threshold: 115°")
print(f"  buffer: 10°")
print(f"\n[PROCESSING]\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect pose
    mp_image = mp.Image(image_format=ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(frame_idx / fps * 1000)
    detection_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
    
    if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
        frame_idx += 1
        continue
    
    landmarks_list = detection_result.pose_landmarks[0]
    confidences = [lm.presence for lm in landmarks_list]
    confidence = np.mean(confidences)
    
    # Extract landmark positions
    landmarks = {}
    for i, lm in enumerate(landmarks_list):
        if i == 11:  # left_shoulder
            landmarks['left_shoulder'] = (lm.x, lm.y)
        elif i == 12:  # right_shoulder
            landmarks['right_shoulder'] = (lm.x, lm.y)
        elif i == 23:  # left_hip
            landmarks['left_hip'] = (lm.x, lm.y)
        elif i == 24:  # right_hip
            landmarks['right_hip'] = (lm.x, lm.y)
        elif i == 25:  # left_knee
            landmarks['left_knee'] = (lm.x, lm.y)
        elif i == 26:  # right_knee
            landmarks['right_knee'] = (lm.x, lm.y)
        elif i == 27:  # left_ankle
            landmarks['left_ankle'] = (lm.x, lm.y)
        elif i == 28:  # right_ankle
            landmarks['right_ankle'] = (lm.x, lm.y)
    
    # Update counter
    if confidence > 0.5:  # Only if enough confidence
        count, state, angle = update_counter(landmarks, confidence)
        angles.append(angle)
        states.append(state.value)
        counts.append(count)
        
        # Print every 50 frames
        if frame_idx % 50 == 0 or count != counts[-2] if len(counts) > 1 else False:
            print(f"Frame {frame_idx:3d}: angle={angle:6.1f}° | state={state.value:12s} | count={count}")
    
    frame_idx += 1

cap.release()

print(f"\n[RESULTS]")
print(f"Final count: {counts[-1] if counts else 0}")
print(f"Angle range: {min(angles):.1f}° to {max(angles):.1f}°" if angles else "No angles detected")
print(f"Average angle: {np.mean(angles):.1f}°" if angles else "No angles detected")
print(f"Unique states: {set(states)}")
