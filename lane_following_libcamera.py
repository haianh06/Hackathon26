#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Following System for Raspberry Pi 5 - Using libcamera (rpicam-still)
Real-time lane following using edge detection and proportional control
"""

import cv2
import numpy as np
import time
import os
import subprocess
import tempfile
from collections import deque
from typing import Tuple, Optional

# ==================== Configuration ====================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 10  # Lower FPS for rpicam-still approach
DISPLAY = True   # Enable display
HEADLESS = False # Disable headless mode

# Preprocessing
BLUR_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150

# ROI
ROI_TOP_PERCENT = 0.5
ROI_BOTTOM_PERCENT = 1.0

# Control
KP = 0.5
SMOOTHING_WINDOW = 5
BASE_SPEED = 255  # Not used in this version, but kept for compatibility

# Motor pins
LEFT_MOTOR_PIN = 12
RIGHT_MOTOR_PIN = 13
PWM_FREQ = 50
STOP_VAL = 1500
DRIVE_SPEED_RANGE = 300


# ==================== Motor Control ====================
class MotorController:
    """Simple motor controller with proportional feedback"""

    def __init__(self, kp: float = KP, base_speed: int = BASE_SPEED):
        self.kp = kp
        self.base_speed = base_speed
        self.error_history = deque(maxlen=SMOOTHING_WINDOW)

    def smooth_error(self, error: float) -> float:
        """Apply moving average to error"""
        self.error_history.append(error)
        return float(np.mean(self.error_history))

    def compute_speeds(self, error: float) -> Tuple[int, int]:
        """
        Compute motor speeds based on error
        left_speed = base - correction
        right_speed = base + correction
        """
        smoothed_error = self.smooth_error(error)
        correction = self.kp * smoothed_error

        # Clamp correction to avoid excessive steering
        correction = np.clip(correction, -DRIVE_SPEED_RANGE, DRIVE_SPEED_RANGE)

        left_speed = int(STOP_VAL - DRIVE_SPEED_RANGE + correction)
        right_speed = int(STOP_VAL + DRIVE_SPEED_RANGE - correction)

        left_speed = np.clip(left_speed, 1000, 2000)
        right_speed = np.clip(right_speed, 1000, 2000)

        return left_speed, right_speed


# ==================== Camera Setup (libcamera) ====================
def init_camera() -> bool:
    """Initialize camera using libcamera (rpicam-still)"""
    try:
        # Test camera with a quick capture
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp:
            cmd = [
                'rpicam-still',
                '--width', str(FRAME_WIDTH),
                '--height', str(FRAME_HEIGHT),
                '--output', tmp.name,
                '--timeout', '1000',
                '--nopreview'
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=5)
            if result.returncode == 0:
                print(f"✓ libcamera initialized ({FRAME_WIDTH}x{FRAME_HEIGHT})")
                return True
            else:
                print(f"✗ libcamera test failed: {result.stderr.decode()}")
                return False

    except Exception as e:
        print(f"✗ libcamera init failed: {type(e).__name__}: {e}")
        return False


def get_frame() -> Optional[np.ndarray]:
    """Capture frame using rpicam-still"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name

        # Capture frame
        cmd = [
            'rpicam-still',
            '--width', str(FRAME_WIDTH),
            '--height', str(FRAME_HEIGHT),
            '--output', tmp_path,
            '--timeout', '100',  # Very short timeout
            '--nopreview'
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=2)
        if result.returncode != 0:
            os.unlink(tmp_path)
            return None

        # Read image
        frame = cv2.imread(tmp_path)
        os.unlink(tmp_path)

        if frame is not None and frame.size > 0:
            # Convert BGR to RGB for consistency
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame

    except Exception as e:
        # Clean up temp file if it exists
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except:
            pass
        return None

    return None


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
    """Main loop for lane following with libcamera"""
    # Disable Qt backend for headless mode
    if HEADLESS:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    print("🚀 Lane Following System - Using libcamera")
    print("=" * 50)
    if HEADLESS:
        print("ℹ Running in HEADLESS mode (no display)")
    else:
        print("ℹ Running with display enabled")
    print("-" * 50)

    motor_controller = MotorController(kp=KP)
    
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Capture frame
            frame = get_frame()
            if frame is None:
                print("⚠ Failed to read frame")
                time.sleep(0.1)
                continue

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
            if frame_count % 3 == 0:  # Log every 3 frames (slower due to capture time)
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                status = "🟢" if abs(error) < 50 else "🟡" if abs(error) < 100 else "🔴"
                print(f"[{fps:4.1f} FPS] {status} Error: {error:+6.1f}px | "
                      f"L:{left_speed:4d} R:{right_speed:4d} | Lane:{lane_center:6.1f}")

            # Display
            if DISPLAY:
                display = draw_diagnostics(frame, edges, roi_edges,
                                          lane_center, error, left_speed, right_speed)
                cv2.imshow("Lane Following (libcamera)", display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹ Exiting...")
                    break

            # Throttle to target FPS
            time.sleep(max(0, 1.0 / FPS - 0.1))  # Account for capture time

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")

    finally:
        print("Cleaning up...")
        if not HEADLESS:
            cv2.destroyAllWindows()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    main()