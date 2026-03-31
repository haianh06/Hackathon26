import cv2
import os
import time
from servo import _set_pwm, stop, LEFT_PIN, RIGHT_PIN
from gpio_handle import gpio_close
import numpy as np

# ============ CÁC THÔNG SỐ ĐIỀU KHIỂN ============
Kp = 0.8
Kd = 0.2
Ki = 0.0

BASE_SPEED = 120
TARGET_RIGHT = 300
TARGET_LEFT = 50
SCAN_Y_RATIO_left = 0.55
SCAN_Y_RATIO_right = 0.75

FRAME_WIDTH = 320
FRAME_HEIGHT = 240


def drive_motor(speed, steering):
    steering = max(-200, min(200, steering))
    left_us = 1500 - speed - steering
    right_us = 1500 + speed - steering
    _set_pwm(LEFT_PIN, left_us)
    _set_pwm(RIGHT_PIN, right_us)


def find_left_lane(edges):
    height, width = edges.shape
    y = int(height * SCAN_Y_RATIO_left)
    line = edges[y]
    mid = len(line) // 2
    left_line = line[:mid]
    if left_line.sum() == 0:
        return -1, y
    left_x = mid - np.argmax(left_line[::-1])
    if left_x <= 0:
        return -1, y
    return left_x, y


def find_right_lane(edges):
    height, width = edges.shape
    y = int(height * SCAN_Y_RATIO_right)
    line = edges[y]
    mid = len(line) // 2
    right_line = line[mid:]
    if right_line.sum() == 0:
        return -1, y
    right_x = mid + np.argmax(right_line)
    if right_x >= width - 1:
        return -1, y
    return right_x, y


def get_camera_frame(picam2, cap):
    if picam2:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cap.read()
        if not ret:
            return None
    return frame


def main():
    if "QT_QPA_PLATFORM" in os.environ:
        del os.environ["QT_QPA_PLATFORM"]

    picam2 = None
    cap = None

    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        sensor_res = picam2.sensor_resolution
        scale = 800 / sensor_res[0] if sensor_res[0] > 800 else 1.0
        config = picam2.create_preview_configuration(main={"size": (int(sensor_res[0]*scale), int(sensor_res[1]*scale)), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
    except Exception:
        pipeline = "libcamerasrc ! video/x-raw, width=800, height=600, framerate=30/1 ! videoconvert ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not picam2 and (not cap or not cap.isOpened()):
        print("❌ Cannot open camera")
        return

    running = True
    stop()

    print("📹 Bật preview camera. Nhấn 's' để bắt đầu bám làn, 'q' để thoát.")
    preview_mode = True

    while preview_mode and running:
        frame = get_camera_frame(picam2, cap)
        if frame is None:
            continue
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        cv2.putText(frame, "Press 's' to start", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Camera Preview", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            preview_mode = False
        elif key == ord('q'):
            running = False
            preview_mode = False

    cv2.destroyWindow("Camera Preview")

    if not running:
        print("❌ Kết thúc do người dùng yêu cầu.")
        return

    print("🚗 Bắt đầu bám làn kết hợp (right/left).")
    try:
        while running:
            frame = get_camera_frame(picam2, cap)
            if frame is None:
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            left_x, y_left = find_left_lane(edges)
            right_x, y_right = find_right_lane(edges)
            y = y_left

            cv2.line(frame, (0, y), (FRAME_WIDTH, y), (80, 80, 80), 1)
            center_x = FRAME_WIDTH // 2
            cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (255, 0, 0), 1)
            cv2.line(frame, (TARGET_LEFT, y-10), (TARGET_LEFT, y+10), (0, 255, 255), 2)
            cv2.putText(frame, "Target Left", (TARGET_LEFT-30, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.line(frame, (TARGET_RIGHT, y-10), (TARGET_RIGHT, y+10), (0, 255, 255), 2)
            cv2.putText(frame, "Target Right", (TARGET_RIGHT-40, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            mode = "No lane"
            steering = 0
            if right_x != -1:
                mode = "right"
                steering = - (right_x - TARGET_RIGHT) * 3
                cv2.circle(frame, (right_x, y), 5, (0, 255, 0), -1)
            elif left_x != -1:
                mode = "left"
                steering = - (left_x - TARGET_LEFT) * 3
                cv2.circle(frame, (left_x, y), 5, (0, 127, 255), -1)
            else:
                mode = "search"

            if mode in ["right", "left"]:
                drive_motor(BASE_SPEED, steering)
                cv2.putText(frame, f"Mode: {mode} lane", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Steer: {int(steering)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                drive_motor(80, 80)
                cv2.putText(frame, "Mode: SEARCH lane", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, "Searching for any lane", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                time.sleep(0.2)

            cv2.imshow("Lane following combined", frame)
            cv2.imshow("Edge", edges)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                running = False
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("🛑 Dừng xe và giải phóng tài nguyên...")
        stop()
        gpio_close()
        if picam2:
            picam2.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
