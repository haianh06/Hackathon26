#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Following + Motor Control Integration for Raspberry Pi 5
Extends lane_following.py with actual motor control using lgpio
"""

import cv2
import numpy as np
import time
import os
import lgpio
from collections import deque
from typing import Tuple, Optional

# ==================== Configuration ====================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
DISPLAY = False  # Changed to False for headless mode
HEADLESS = True  # Enable headless (no X11)

# Preprocessing
BLUR_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150

# ROI
ROI_TOP_PERCENT = 0.5
ROI_BOTTOM_PERCENT = 1.0

# Control
BASE_SPEED = 255
KP = 0.5  # Tune this for your robot
SMOOTHING_WINDOW = 5

# Motor pins (adjust to your wiring)
LEFT_MOTOR_PIN = 12
RIGHT_MOTOR_PIN = 13
PWM_FREQ = 50
STOP_VAL = 1500
DRIVE_SPEED_RANGE = 300


# ==================== GPIO & Motor Setup ====================
class RPiMotorController:
    """Motor controller using lgpio for Raspberry Pi"""
    
    def __init__(self, left_pin: int = LEFT_MOTOR_PIN, 
                 right_pin: int = RIGHT_MOTOR_PIN,
                 freq: int = PWM_FREQ):
        self.h = lgpio.gpiochip_open(0)
        self.left_pin = left_pin
        self.right_pin = right_pin
        self.freq = freq
        
        # Claim pins
        lgpio.gpio_claim_output(self.h, self.left_pin)
        lgpio.gpio_claim_output(self.h, self.right_pin)
        
        self.error_history = deque(maxlen=SMOOTHING_WINDOW)
        self.kp = KP
        print(f"✓ Motor controller initialized (pins: L={left_pin}, R={right_pin})")
    
    def smooth_error(self, error: float) -> float:
        """Apply moving average to error"""
        self.error_history.append(error)
        return float(np.mean(self.error_history))
    
    def compute_speeds(self, error: float) -> Tuple[int, int]:
        """Compute motor speeds based on proportional error"""
        smoothed_error = self.smooth_error(error)
        correction = self.kp * smoothed_error
        correction = np.clip(correction, -DRIVE_SPEED_RANGE, DRIVE_SPEED_RANGE)
        
        left_speed = int(STOP_VAL - DRIVE_SPEED_RANGE + correction)
        right_speed = int(STOP_VAL + DRIVE_SPEED_RANGE - correction)
        
        left_speed = np.clip(left_speed, 1000, 2000)
        right_speed = np.clip(right_speed, 1000, 2000)
        
        return left_speed, right_speed
    
    def set_motor_speeds(self, left_speed: int, right_speed: int):
        """Set motor speeds using PWM"""
        try:
            left_duty = (left_speed / 20000) * 100
            right_duty = (right_speed / 20000) * 100
            
            lgpio.tx_pwm(self.h, self.left_pin, self.freq, left_duty)
            lgpio.tx_pwm(self.h, self.right_pin, self.freq, right_duty)
        except Exception as e:
            print(f"⚠ Motor control error: {e}")
    
    def stop(self):
        """Stop motors"""
        try:
            lgpio.tx_pwm(self.h, self.left_pin, self.freq, 0)
            lgpio.tx_pwm(self.h, self.right_pin, self.freq, 0)
            print("🛑 Motors stopped")
        except Exception as e:
            print(f"⚠ Error stopping motors: {e}")
    
    def cleanup(self):
        """Release GPIO resources"""
        try:
            self.stop()
            lgpio.gpiochip_close(self.h)
            print("✓ GPIO cleanup complete")
        except Exception as e:
            print(f"⚠ Cleanup error: {e}")


# ==================== Camera Setup ====================
def init_camera() -> Optional:
    """Initialize camera using Picamera2 context manager"""
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        
        # Just return the object - configure happens on first capture
        print(f"✓ Picamera2 initialized ({FRAME_WIDTH}x{FRAME_HEIGHT})")
        return picam2
    
    except ImportError:
        print("✗ Picamera2 not available")
    except Exception as e:
        print(f"✗ Picamera2 init failed: {type(e).__name__}: {str(e)[:80]}")
    
    return None


def get_frame(camera) -> Optional[np.ndarray]:
    """Capture frame, handling configuration lazily"""
    try:
        # Configure if not already done
        if not camera.is_open:
            config = camera.create_preview_configuration(
                main={'size': (FRAME_WIDTH, FRAME_HEIGHT), 'format': 'RGB888'}
            )
            camera.configure(config)
            # Don't start() - try capture_array() directly
        
        # Try to capture frame
        frame = camera.capture_array()
        if frame is not None and frame.size > 0:
            return frame
        return None
    except Exception as e:
        # Silently fail to avoid spam
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


# ==================== Lane Detection ====================
def detect_lane(roi_edges: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Detect left and right lane boundaries"""
    h, w = roi_edges.shape
    if h == 0:
        return None, None
    
    bottom_row = roi_edges[-1, :]
    edge_pixels = np.where(bottom_row > 0)[0]
    
    if len(edge_pixels) < 5:
        return None, None
    
    left_x = None
    right_x = None
    
    left_candidates = edge_pixels[edge_pixels < w // 2]
    if len(left_candidates) > 0:
        left_x = float(np.mean(left_candidates[-5:]))
    
    right_candidates = edge_pixels[edge_pixels >= w // 2]
    if len(right_candidates) > 0:
        right_x = float(np.mean(right_candidates[:5]))
    
    return left_x, right_x


def compute_lane_center(left_x: Optional[float], right_x: Optional[float]) -> Optional[float]:
    """Compute lane center"""
    if left_x is None or right_x is None:
        return None
    return (left_x + right_x) / 2.0


def compute_error(lane_center: Optional[float], frame_width: int) -> float:
    """Compute steering error"""
    if lane_center is None:
        return 0.0
    return lane_center - frame_width / 2.0


# ==================== Visualization ====================
def draw_diagnostics(frame: np.ndarray, edges: np.ndarray, 
                    lane_center: Optional[float], error: float,
                    left_speed: int, right_speed: int) -> np.ndarray:
    """Draw diagnostics on frame"""
    h, w = frame.shape[:2]
    display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # ROI boundary
    roi_top = int(h * ROI_TOP_PERCENT)
    cv2.line(display, (0, roi_top), (w, roi_top), (0, 255, 0), 2)
    
    # Image center
    image_center = w // 2
    cv2.circle(display, (image_center, h - 10), 5, (0, 255, 0), -1)
    
    # Lane center
    if lane_center is not None:
        cv2.circle(display, (int(lane_center), h - 10), 5, (0, 0, 255), -1)
    
    # Text overlays
    cv2.putText(display, f"Error: {error:+.1f}px", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display, f"L:{left_speed}  R:{right_speed}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(display, "Press 'q' to exit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Edge map
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return np.vstack([display, edges_colored])


# ==================== Main Loop ====================
def main():
    """Main loop with motor control integration"""
    # Disable Qt backend for headless mode
    if HEADLESS:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    print("🚀 Lane Following System with Motor Control")
    print("================================================")
    if HEADLESS:
        print("ℹ Running in HEADLESS mode (no display)")
    print("")
    
    camera = init_camera()
    if camera is None:
        print("✗ Failed to initialize camera")
        return
    
    motor = RPiMotorController()
    
    frame_count = 0
    start_time = time.time()
    
    print("Starting motor control loop... (Press 'q' to exit)\n")
    time.sleep(1)
    
    try:
        while True:
            frame = get_frame(camera)
            if frame is None:
                print("⚠ Frame read failed")
                break
            
            frame_count += 1
            
            # Process
            edges = preprocess_frame(frame)
            roi_edges = extract_roi(edges)
            left_x, right_x = detect_lane(roi_edges)
            lane_center = compute_lane_center(left_x, right_x)
            error = compute_error(lane_center, FRAME_WIDTH)
            
            # Control
            left_speed, right_speed = motor.compute_speeds(error)
            motor.set_motor_speeds(left_speed, right_speed)
            
            # Log every 10 frames
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                status = "🟢 Lane OK" if lane_center is not None else "🔴 No Lane"
                print(f"[{fps:.1f} FPS] {status} | "
                      f"Error: {error:+7.1f}px | "
                      f"Speeds: L={left_speed} R={right_speed}")
            
            # Display
            if DISPLAY:
                display = draw_diagnostics(frame, edges, lane_center, 
                                          error, left_speed, right_speed)
                cv2.imshow("Lane Following + Motors", display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹ Exiting...")
                    break
    
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
    
    finally:
        print("\nCleaning up...")
        motor.cleanup()
        if camera is not None:
            try:
                camera.stop()
            except Exception as e:
                print(f"⚠ Camera cleanup error: {e}")
        if not HEADLESS:
            cv2.destroyAllWindows()
        print("✓ Done")


if __name__ == "__main__":
    main()
