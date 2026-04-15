import cv2
import os
import time
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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

# Dynamic scan line positions (closer together)
SCAN_Y_RATIO_left = 0.45   # 45% from top
SCAN_Y_RATIO_right = 0.62  # 62% from top (closer, only 17% gap)
SCAN_SEARCH_UP_ROWS = 35   # scan upward from the base scan line when lane is missing

# Camera angle adjustment (for better lane detection)
CAMERA_ROTATION = 0  # Adjust if needed: positive = rotate clockwise

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Bird's eye view parameters
BIRDS_EYE_WIDTH = 160
BIRDS_EYE_HEIGHT = 240


def get_perspective_transform_matrix(frame_width, frame_height):
    """
    Tính toán ma trận perspective transform để chuyển từ camera view sang bird's eye view.
    """
    # Định nghĩa 4 điểm ở camera view (hình thang)
    # Điểm từ dưới lên trên (gần đến xa)
    src_points = np.float32([
        [0, frame_height],           # bottom-left
        [frame_width, frame_height], # bottom-right
        [frame_width * 0.25, int(frame_height * 0.45)],  # top-left
        [frame_width * 0.75, int(frame_height * 0.45)],  # top-right
    ])

    # Định nghĩa 4 điểm ở bird's eye view (hình chữ nhật)
    dst_points = np.float32([
        [0, BIRDS_EYE_HEIGHT],           # bottom-left
        [BIRDS_EYE_WIDTH, BIRDS_EYE_HEIGHT], # bottom-right
        [0, 0],                           # top-left
        [BIRDS_EYE_WIDTH, 0],             # top-right
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return matrix


def warp_to_birds_eye(frame, matrix):
    """Chuyển frame sang bird's eye view."""
    warped = cv2.warpPerspective(frame, matrix, (BIRDS_EYE_WIDTH, BIRDS_EYE_HEIGHT))
    return warped


def draw_birds_eye_lanes(birds_eye_frame, left_x, y_left, right_x, y_right, matrix):
    """
    Vẽ các lane lines lên bird's eye view.
    Mapping tọa độ từ camera view sang bird's eye view.
    """
    h, w = birds_eye_frame.shape[:2]
    bot = h

    # Scale các tọa độ từ camera frame sang bird's eye frame
    # Left lane (camera x, y) → (bird's eye x, y)
    if left_x != -1:
        # Ánh xạ tọa độ: giã đơn vị từ camera (320x240) sang bird's eye (160x240)
        left_x_be = max(0, min(w - 1, int((left_x / FRAME_WIDTH) * w)))
        y_left_be = max(0, min(h - 1, int((y_left / FRAME_HEIGHT) * h)))
        cv2.line(birds_eye_frame, (left_x_be, y_left_be),
                 (left_x_be, bot), (0, 165, 255), 3)
        cv2.circle(birds_eye_frame, (left_x_be, y_left_be), 4, (0, 165, 255), -1)

    # Right lane (camera x, y) → (bird's eye x, y)
    if right_x != -1:
        right_x_be = max(0, min(w - 1, int((right_x / FRAME_WIDTH) * w)))
        y_right_be = max(0, min(h - 1, int((y_right / FRAME_HEIGHT) * h)))
        cv2.line(birds_eye_frame, (right_x_be, y_right_be),
                 (right_x_be, bot), (255, 100, 0), 3)
        cv2.circle(birds_eye_frame, (right_x_be, y_right_be), 4, (255, 100, 0), -1)


def drive_motor(speed, steering):
    steering = max(-200, min(200, steering))
    left_us = 1500 - speed - steering
    right_us = 1500 + speed - steering
    _set_pwm(LEFT_PIN, left_us)
    _set_pwm(RIGHT_PIN, right_us)


def _find_inner_lane_edge(line, side):
    mid = len(line) // 2
    if side == "left":
        left_half = line[:mid]
        if left_half.sum() == 0:
            return -1
        inner_x = mid - 1 - np.argmax(left_half[::-1])
        return inner_x if inner_x > 0 else -1

    right_half = line[mid:]
    if right_half.sum() == 0:
        return -1
    inner_x = mid + np.argmax(right_half)
    return inner_x if inner_x < len(line) - 1 else -1


def find_left_lane(edges):
    """Tìm điểm inner edge của lane trái – quét lên trên nếu không tìm thấy ở scan line gốc."""
    height, width = edges.shape
    base_y = int(height * SCAN_Y_RATIO_left)
    for offset in range(SCAN_SEARCH_UP_ROWS + 1):
        y = base_y - offset
        if y < 0:
            break
        left_x = _find_inner_lane_edge(edges[y], "left")
        if left_x != -1:
            return left_x, y
    return -1, base_y


def find_right_lane(edges):
    """Tìm điểm inner edge của lane phải – quét lên trên nếu không tìm thấy ở scan line gốc."""
    height, width = edges.shape
    base_y = int(height * SCAN_Y_RATIO_right)
    for offset in range(SCAN_SEARCH_UP_ROWS + 1):
        y = base_y - offset
        if y < 0:
            break
        right_x = _find_inner_lane_edge(edges[y], "right")
        if right_x != -1:
            return right_x, y
    return -1, base_y


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


def draw_edge_overlay(frame, edges):
    """Vẽ Canny edges dưới dạng màu sắc lên frame (thay vì window riêng)."""
    h, w = frame.shape[:2]

    # Tạo overlay màu từ edges - blue channel cho edges detected
    edge_overlay = np.zeros_like(frame)
    edge_overlay[:, :, 0] = edges  # Blue channel = edges
    edge_overlay[:, :, 2] = edges // 2  # Red channel = edges//2 (purple tint)

    # Blend edge overlay với frame
    cv2.addWeighted(edge_overlay, 0.15, frame, 0.85, 0, frame)


def create_edge_view_with_lanes(edges, left_x, y_left, right_x, y_right):
    """Tạo một view riêng hiển thị cạnh Canny với các lane lines được tô màu."""
    # Chuyển edges từ binary sang BGR để vẽ màu
    edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    h, w = edge_display.shape[:2]
    bot = h

    # Vẽ các lane lines lên edge view với màu sắc rõ ràng
    if left_x != -1:
        # Vẽ lane trái (cam/yellow)
        cv2.line(edge_display, (left_x, y_left), (left_x, bot), (0, 165, 255), 3)
        cv2.circle(edge_display, (left_x, y_left), 5, (0, 165, 255), -1)

    if right_x != -1:
        # Vẽ lane phải (xanh dương)
        cv2.line(edge_display, (right_x, y_right), (right_x, bot), (255, 100, 0), 3)
        cv2.circle(edge_display, (right_x, y_right), 5, (255, 100, 0), -1)

    # Vẽ scan lines
    y_left_line = int(h * SCAN_Y_RATIO_left)
    y_right_line = int(h * SCAN_Y_RATIO_right)
    cv2.line(edge_display, (0, y_left_line), (w, y_left_line), (100, 255, 100), 1)
    cv2.line(edge_display, (0, y_right_line), (w, y_right_line), (100, 255, 100), 1)

    # Vẽ center line
    center_x = w // 2
    cv2.line(edge_display, (center_x, 0), (center_x, h), (200, 200, 200), 1)

    return edge_display


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
        try:
            if "LensPosition" in picam2.controls:
                picam2.set_controls({"LensPosition": 0.5})
        except: pass
        picam2.start()
    except Exception as e:
        print(f"⚠️  PiCamera2 unavailable: {e}, trying V4L2...")
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
        cv2.putText(frame, "Press 's' to START | 'q' to EXIT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Mode: Preview", (10, FRAME_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.imshow("Camera Preview", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            preview_mode = False
            print("✅ Bắt đầu bám làn...")
        elif key == ord('q'):
            running = False
            preview_mode = False
            print("❌ Thoát chế độ preview")

    cv2.destroyWindow("Camera Preview")

    if not running:
        print("❌ Kết thúc do người dùng yêu cầu.")
        return

    print("🚗 Bắt đầu bám làn - Nhấn 'q' để dừng")
    last_lane = "right"  # Mặc định
    last_left_x = -1
    last_right_x = -1

    # Tính toán ma trận perspective transform cho bird's eye view
    perspective_matrix = get_perspective_transform_matrix(FRAME_WIDTH, FRAME_HEIGHT)

    try:
        while running:
            frame = get_camera_frame(picam2, cap)
            if frame is None:
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 100, 200)

            # Vẽ visualization của Canny edges lên frame chính
            draw_edge_overlay(frame, edges)

            left_x, y_left = find_left_lane(edges)
            right_x, y_right = find_right_lane(edges)
            y = y_left

            # Improvement: Use last known lane position if current detection is missing
            # This helps with smooth tracking when one lane is momentarily lost
            if left_x != -1:
                last_left_x = left_x
            elif last_left_x != -1:
                left_x = last_left_x

            if right_x != -1:
                last_right_x = right_x
            elif last_right_x != -1:
                right_x = last_right_x

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
                cv2.putText(frame, f"Mode: {mode} lane | Speed: {BASE_SPEED}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 80), 1)
            else:
                spin_steering = 80 if last_lane == "right" else -80
                drive_motor(0, spin_steering)
                spin_dir = "left" if spin_steering > 0 else "right"
                cv2.putText(frame, f"SEARCH ({spin_dir})", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)
                time.sleep(0.1)

            # Create and display edge view with colored lanes
            edge_view = create_edge_view_with_lanes(edges, left_x, y_left, right_x, y_right)

            # Create bird's eye view
            birds_eye_edges = warp_to_birds_eye(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), perspective_matrix)
            draw_birds_eye_lanes(birds_eye_edges, left_x, y_left, right_x, y_right, perspective_matrix)

            cv2.imshow("Lane following combined", frame)
            cv2.imshow("Edge Detection", edge_view)
            cv2.imshow("Bird's Eye View", birds_eye_edges)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                print("⏹️  Dừng xe...")
                running = False
                break

    except KeyboardInterrupt:
        print("⚠️  Interrupt từ keyboard")
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