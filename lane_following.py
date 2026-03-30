#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Following System - TEST VERSION with Synthetic Frames
Use this to test lane-following logic when Picamera2 camera is unavailable.
Same algorithm as lane_following.py but generates test road patterns.
"""

import cv2
import numpy as np
import time
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from collections import deque
from typing import Tuple, Optional

# ==================== Configuration ====================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
HEADLESS = True
BLUR_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150
ROI_TOP_PERCENT = 0.5
ROI_BOTTOM_PERCENT = 1.0
KP = 0.5
SMOOTHING_WINDOW = 5
BASE_SPEED = 255
LEFT_MOTOR_PIN = 12
RIGHT_MOTOR_PIN = 13
PWM_FREQ = 50
STOP_VAL = 1500
DRIVE_SPEED_RANGE = 300


# ==================== Test Frame Generation ====================
def generate_test_frame(t: float) -> np.ndarray:
    """Generate synthetic road frame with lane"""
    frame = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) * 80  # Dark asphalt
    
    # Simulate moving lane center
    lane_center = FRAME_WIDTH // 2 + int(80 * np.sin(t * 0.5))
    lane_width = 60
    
    # Draw lane markings (white)
    cv2.line(frame, (lane_center - lane_width, 0), 
             (lane_center - lane_width, FRAME_HEIGHT), (255, 255, 255), 3)
    cv2.line(frame, (lane_center + lane_width, 0), 
             (lane_center + lane_width, FRAME_HEIGHT), (255, 255, 255), 3)
    
    # Add some texture variation
    noise = np.random.randint(-10, 10, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return frame


# ==================== Image Processing ====================
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Grayscale -> Blur -> Canny"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    return edges


def extract_roi(edges: np.ndarray) -> np.ndarray:
    """Extract bottom half of frame"""
    h = edges.shape[0]
    roi_top = int(h * ROI_TOP_PERCENT)
    return edges[roi_top:, :]


def detect_lane(roi_edges: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Detect left and right lane boundaries using edge clustering"""
    h, w = roi_edges.shape
    if h == 0:
        return None, None
    
    # Find edge pixels
    edge_pixels = np.where(roi_edges > 0)
    if len(edge_pixels[1]) == 0:  # No edges found
        return None, None
    
    x_coords = edge_pixels[1]
    threshold = w // 2
    
    # Split edges into left and right halves
    left_edges = x_coords[x_coords < threshold]
    right_edges = x_coords[x_coords >= threshold]
    
    left_x = np.median(left_edges) if len(left_edges) > 0 else None
    right_x = np.median(right_edges) if len(right_edges) > 0 else None
    
    return left_x, right_x


def compute_lane_center(left_x: Optional[float], right_x: Optional[float], 
                       width: int) -> float:
    """Compute lane center from boundaries"""
    if left_x is not None and right_x is not None:
        return (left_x + right_x) / 2
    elif left_x is not None:
        return left_x + width // 4
    elif right_x is not None:
        return right_x - width // 4
    else:
        return width / 2  # Center if no lanes detected


def compute_error(lane_center: float, frame_width: int) -> float:
    """Compute steering error (deviation from center)"""
    return lane_center - frame_width / 2


# ==================== Motor Control ====================
class MotorController:
    """Simulated motor controller with proportional feedback"""
    
    def __init__(self, kp: float = KP, base_speed: int = BASE_SPEED):
        self.kp = kp
        self.base_speed = base_speed
        self.error_history = deque(maxlen=SMOOTHING_WINDOW)
    
    def smooth_error(self, error: float) -> float:
        """Apply moving average to error"""
        self.error_history.append(error)
        return float(np.mean(self.error_history))
    
    def compute_speeds(self, error: float) -> Tuple[int, int]:
        """Compute motor speeds based on error"""
        smoothed_error = self.smooth_error(error)
        correction = self.kp * smoothed_error
        correction = np.clip(correction, -DRIVE_SPEED_RANGE, DRIVE_SPEED_RANGE)
        
        left_speed = int(STOP_VAL - DRIVE_SPEED_RANGE + correction)
        right_speed = int(STOP_VAL + DRIVE_SPEED_RANGE - correction)
        
        left_speed = np.clip(left_speed, 1000, 2000)
        right_speed = np.clip(right_speed, 1000, 2000)
        
        return left_speed, right_speed


# ==================== Visualization ====================
def draw_diagnostics(frame: np.ndarray, edges: np.ndarray, roi_edges: np.ndarray,
                    lane_center: float, error: float, 
                    left_speed: int, right_speed: int) -> np.ndarray:
    """Draw diagnostics on frame"""
    h, w = frame.shape[:2]
    roi_top = int(h * ROI_TOP_PERCENT)
    
    # Create display frame
    display = frame.copy()
    
    # Draw ROI box
    cv2.rectangle(display, (0, roi_top), (w, h), (0, 255, 0), 2)
    
    # Draw lane center line
    cv2.line(display, (int(lane_center), roi_top), (int(lane_center), h), (0, 255, 255), 2)
    
    # Draw center line
    cv2.line(display, (w // 2, roi_top), (w // 2, h), (255, 0, 0), 1)
    
    # Add text
    cv2.putText(display, f"Error: {error:+.1f}px", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, f"L:{left_speed} R:{right_speed}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return display


# ==================== Main Loop ====================
def main():
    """Main loop for lane following with synthetic frames"""
    print("🚀 Lane Following System - TEST MODE (Synthetic Frames)")
    print("=" * 50)
    print("ℹ Using generated test road patterns")
    print()
    
    motor_controller = MotorController(kp=KP, base_speed=BASE_SPEED)
    
    print("Press Ctrl+C to exit")
    print("-" * 50)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            t = time.time() - start_time
            
            # Generate synthetic frame
            frame = generate_test_frame(t)
            
            # Process
            edges = preprocess_frame(frame)
            roi_edges = extract_roi(edges)
            left_x, right_x = detect_lane(roi_edges)
            lane_center = compute_lane_center(left_x, right_x, FRAME_WIDTH)
            error = compute_error(lane_center, FRAME_WIDTH)
            
            # Control
            left_speed, right_speed = motor_controller.compute_speeds(error)
            
            # Logging
            frame_count += 1
            if frame_count % 5 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                status = "🟢" if abs(error) < 50 else "🟡"
                print(f"[{fps:5.1f} FPS] {status} Error: {error:+7.1f}px | "
                      f"L:{left_speed:4d} R:{right_speed:4d} | Lane:{lane_center:6.1f}")
            
            # Throttle
            time.sleep(1.0 / FPS)
    
    except KeyboardInterrupt:
        print("\n\n✓ Exiting...")


if __name__ == "__main__":
    main()
