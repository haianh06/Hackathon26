import cv2
import os
import time
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from manual_control.servo import _set_pwm, stop, LEFT_PIN, RIGHT_PIN
from manual_control.gpio_handle import gpio_close
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
    """Tìm điểm trong (inner edge) của lane trái – điểm sát giữa nhất."""
    height, width = edges.shape
    y = int(height * SCAN_Y_RATIO_left)
    line = edges[y]
    mid = len(line) // 2
    left_half = line[:mid]
    if left_half.sum() == 0:
        return -1, y
    # Quét từ giữa ra ngoài → lấy điểm trong nhất
    inner_x = mid - 1 - np.argmax(left_half[::-1])
    if inner_x <= 0:
        return -1, y
    return inner_x, y


def find_right_lane(edges):
    """Tìm điểm trong (inner edge) của lane phải – điểm sát giữa nhất."""
    height, width = edges.shape
    y = int(height * SCAN_Y_RATIO_right)
    line = edges[y]
    mid = len(line) // 2
    right_half = line[mid:]
    if right_half.sum() == 0:
        return -1, y
    # Quét từ giữa ra ngoài → lấy điểm trong nhất
    inner_x = mid + np.argmax(right_half)
    if inner_x >= width - 1:
        return -1, y
    return inner_x, y


def draw_lane_overlay(frame, left_x, y_left, right_x, y_right):
    """Tô màu vùng lane giữa hai điểm detect được."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    bot = h  # kéo xuống đáy frame

    if left_x != -1 and right_x != -1:
        # Tô vùng lane giữa hai line
        pts = np.array([
            [left_x,  y_left],
            [right_x, y_right],
            [right_x, bot],
            [left_x,  bot],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 200, 80))   # xanh lá nhạt
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

        # Vẽ đường line trái (màu cam)
        cv2.line(frame, (left_x, y_left), (left_x, bot), (0, 140, 255), 2)
        # Vẽ đường line phải (màu xanh dương)
        cv2.line(frame, (right_x, y_right), (right_x, bot), (255, 80, 0), 2)

    elif left_x != -1:
        # Chỉ có lane trái
        cv2.line(frame, (left_x, y_left), (left_x, bot), (0, 140, 255), 2)
        pts = np.array([
            [left_x, y_left],
            [w // 2, y_left],
            [w // 2, bot],
            [left_x, bot],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 140, 255))
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    elif right_x != -1:
        # Chỉ có lane phải
        cv2.line(frame, (right_x, y_right), (right_x, bot), (255, 80, 0), 2)
        pts = np.array([
            [w // 2, y_right],
            [right_x, y_right],
            [right_x, bot],
            [w // 2, bot],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (255, 80, 0))
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)


def draw_orientation_mark(frame, steering, base_y=None):
    """Vẽ mũi tên hướng steering (orientation mark) ở dưới frame."""
    h, w = frame.shape[:2]
    if base_y is None:
        base_y = int(h * 0.82)
    cx = w // 2

    # Clamp và scale độ lệch thành độ dài mũi tên
    max_steer = 200
    clamped = max(-max_steer, min(max_steer, steering))
    arrow_len = int((clamped / max_steer) * (w // 4))
    tip_x = cx + arrow_len

    # Màu: xanh lá khi thẳng, vàng cam khi lệch nhiều
    ratio = abs(clamped) / max_steer
    color = (
        int(50 + ratio * 200),       # B
        int(230 - ratio * 150),      # G
        int(ratio * 255),            # R  → BGR
    )

    # Vẽ nền mờ cho HUD
    bar_h = 28
    cv2.rectangle(frame,
                  (cx - w // 4, base_y - bar_h // 2),
                  (cx + w // 4, base_y + bar_h // 2),
                  (30, 30, 30), -1)
    # Scale bar
    cv2.rectangle(frame,
                  (cx, base_y - bar_h // 4),
                  (tip_x, base_y + bar_h // 4),
                  color, -1)
    # Vẽ mũi tên
    if arrow_len != 0:
        cv2.arrowedLine(frame,
                        (cx, base_y),
                        (tip_x, base_y),
                        color, 2, tipLength=0.35)
    # Đường trung tâm
    cv2.line(frame, (cx, base_y - bar_h // 2), (cx, base_y + bar_h // 2),
             (200, 200, 200), 1)
    # Label
    label = f"Steer: {int(steering):+d}"
    cv2.putText(frame, label, (cx - w // 4, base_y - bar_h // 2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)


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
    last_lane = "right"  # Mặc định
    try:
        while running:
            frame = get_camera_frame(picam2, cap)
            if frame is None:
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 100, 200)

            left_x, y_left = find_left_lane(edges)
            right_x, y_right = find_right_lane(edges)
            y = y_left

            # --- Tô màu vùng lane và vẽ line ---
            draw_lane_overlay(frame, left_x, y_left, right_x, y_right)

            # --- Scan line & guide lines ---
            cv2.line(frame, (0, y_left), (FRAME_WIDTH, y_left), (80, 80, 80), 1)
            cv2.line(frame, (0, y_right), (FRAME_WIDTH, y_right), (60, 60, 60), 1)
            center_x = FRAME_WIDTH // 2
            cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (200, 200, 200), 1)

            # --- Target markers ---
            for tx, label in [(TARGET_LEFT, "TL"), (TARGET_RIGHT, "TR")]:
                cv2.line(frame, (tx, y-10), (tx, y+10), (0, 255, 255), 2)
                cv2.putText(frame, label, (tx-10, y-14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

            # --- Điểm inner edge (vẽ sau overlay để nổi bật) ---
            if left_x != -1:
                cv2.circle(frame, (left_x, y_left), 6, (0, 140, 255), -1)
                cv2.circle(frame, (left_x, y_left), 8, (255, 255, 255), 1)
                cv2.putText(frame, f"L:{left_x}", (left_x + 6, y_left - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)
            if right_x != -1:
                cv2.circle(frame, (right_x, y_right), 6, (255, 80, 0), -1)
                cv2.circle(frame, (right_x, y_right), 8, (255, 255, 255), 1)
                cv2.putText(frame, f"R:{right_x}", (right_x - 40, y_right - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 120, 0), 1)

            mode = "No lane"
            steering = 0
            if right_x != -1:
                mode = "right"
                last_lane = "right"
                steering = -(right_x - TARGET_RIGHT) * 3
            elif left_x != -1:
                mode = "left"
                last_lane = "left"
                steering = -(left_x - TARGET_LEFT) * 3
            else:
                mode = "search"

            # --- Orientation mark (steering arrow HUD) ---
            draw_orientation_mark(frame, steering)

            if mode in ["right", "left"]:
                drive_motor(BASE_SPEED, steering)
                cv2.putText(frame, f"Mode: {mode} lane", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 80), 1)
            else:
                spin_steering = 80 if last_lane == "right" else -80
                drive_motor(0, spin_steering)
                spin_dir = "left" if spin_steering > 0 else "right"
                cv2.putText(frame, f"SEARCH ({spin_dir})", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)
                time.sleep(0.1)

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
