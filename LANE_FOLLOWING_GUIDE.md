# Lane Following System - Quick Start Guide

## 📋 Overview

Two implementations of a real-time lane-following system for Raspberry Pi 5:

1. **`lane_following.py`** - Computer vision only (no motor control)
2. **`lane_following_motor.py`** - Full integration with motor control via lgpio

## 🚀 Quick Start

### 1. Basic Test (Computer Vision Only)

```bash
source venv/bin/activate
python3 lane_following.py
```

This will:
- Capture camera stream
- Detect lane edges using Canny edge detection
- Compute steering error
- Display visualization
- Print FPS and telemetry

**Press 'q' to exit.**

### 2. With Motor Control

```bash
source venv/bin/activate
python3 lane_following_motor.py
```

This will:
- Everything above, PLUS
- Actually control left/right motor speeds
- Steer robot to follow lane in real-time

**Note:** Make sure motor pins are correct in your code (default: 12, 13).

## ⚙️ Configuration

Edit the top section of either file:

```python
# Preprocessing
CANNY_LOW = 50          # Lower = more edges
CANNY_HIGH = 150        # Higher = fewer, cleaner edges

# Control
KP = 0.5                # Proportional gain (tune this!)
SMOOTHING_WINDOW = 5    # Moving average smoothing

# Motor pins
LEFT_MOTOR_PIN = 12
RIGHT_MOTOR_PIN = 13
```

### Tuning KP (Most Important)

- **KP too low** (< 0.2): Robot understeers (doesn't correct enough)
- **KP optimal** (0.4-0.8): Smooth, responsive lane following
- **KP too high** (> 1.0): Robot oscillates (overcorrects)

Start with **KP = 0.5** and adjust based on behavior.

## 📊 Understanding the Output

```
FPS: 20.5 | Error: +42.3px | L-Speed: 1400 R-Speed: 1600 | Lane: 320.5
        |          |               |                    |         |
        |          |               |              Motor speeds   Lane center
        |          |          Steering correction applied        X position
        |    Steering error (pixels from center)
   Frames per second
```

- **Error > 0**: Lane is to the RIGHT → steer LEFT → increase left speed
- **Error < 0**: Lane is to the LEFT → steer RIGHT → increase right speed
- **Error ≈ 0**: Robot is centered in lane

## 🔍 Lane Detection Algorithm

1. **Grayscale** → Convert RGB to single channel
2. **Gaussian Blur** → Reduce noise (kernel: 5x5)
3. **Canny Edge Detection** → Find edges (50-150 threshold)
4. **ROI Extraction** → Use only bottom half of frame
5. **Edge Clustering** → Find left/right lane boundaries
6. **Center Calculation** → (left + right) / 2
7. **Error Computation** → center - image_center

## 🎮 Motor Control Logic

```
P-Controller:
  error = lane_center_x - image_center_x
  correction = Kp * error
  left_speed = base_speed - correction
  right_speed = base_speed + correction
```

This is a simple proportional (P) controller. For more stability, you can add:
- Integral term (I) - cumulative error
- Derivative term (D) - rate of error change
- But P-only is sufficient for lane following!

## 📷 Camera Setup

### Picamera2 (Raspberry Pi Official)
Default, does NOT require separate installation if using latest OS.

### OpenCV Fallback
Handles different camera devices (`/dev/video0`, `/dev/video1`, etc.)

## 🔧 Hardware Integration

### Motor Pins
Adjust these in the code:
```python
LEFT_MOTOR_PIN = 12
RIGHT_MOTOR_PIN = 13
```

### PWM Configuration
```python
PWM_FREQ = 50           # 50 Hz (typical servo frequency)
STOP_VAL = 1500         # Center position (microseconds)
DRIVE_SPEED_RANGE = 300 # ±300 from stop
```

Speed formula:
```
speed = STOP_VAL ± DRIVE_SPEED_RANGE
      = 1500 ± 300
      = 1200 (reverse) to 1800 (forward)
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No camera" | Check `/dev/video*`, enable CSI in raspi-config |
| "Motor not moving" | Verify pins (GPIO 12/13), check power supply |
| "Lane oscillates" | Reduce KP value |
| "Lane never corrected" | Increase KP value |
| "Slow FPS" | Reduce frame size or resolution in camera config |
| "Frame drops" | Reduce window size or increase CANNY thresholds |

## 📈 Performance Tips

1. **Reduce CANNY thresholds** if lane edges are faint
2. **Increase BLUR kernel** if image is noisy
3. **Adjust ROI percentage** to focus on lane area
4. **Tune motor speeds** to match robot capability
5. **Use moving average** (SMOOTHING_WINDOW) for stability

## 🎯 Advanced Tuning

### For Different Lane Types

**Wooden/Concrete Lane:**
```python
CANNY_LOW = 30
CANNY_HIGH = 100
KP = 0.6
```

**White Line on Dark Floor:**
```python
CANNY_LOW = 50
CANNY_HIGH = 150
KP = 0.5
```

**Dark Line on Light Floor:**
```python
CANNY_LOW = 100
CANNY_HIGH = 200
KP = 0.4
```

## 📝 Expected Behavior

1. Robot centers itself in lane
2. Correct steering left/right smoothly
3. Follow curves naturally
4. Recover from disturbances
5. Frame rate > 15 FPS on Raspberry Pi 5

## 🚨 Safety

- Always test in a safe area
- Start with low speed (DRIVE_SPEED_RANGE = 100)
- Increase gradually (100 → 200 → 300)
- Have manual stop mechanism ready
- Monitor motor temperatures

## 📚 Further Reading

- OpenCV Edge Detection: https://docs.opencv.org/master/da/d22/tutorial_py_canny.html
- Proportional Control: https://en.wikipedia.org/wiki/Proportional_control
- Raspberry Pi GPIO: https://www.raspberrypi.com/documentation/computers/os.html

---

**Happy lane following! 🚗💨**
