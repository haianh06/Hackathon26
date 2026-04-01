import cv2
import os
import time
import sys
import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from manual_control.servo import _set_pwm, stop, LEFT_PIN, RIGHT_PIN
from manual_control.gpio_handle import gpio_close
import numpy as np

# ============ CÁC THÔNG SỐ ĐIỀU KHIỂN ============
# Bạn có thể tuning nhẹ lại Kp, Kd nếu xe đánh lái gắt quá.
Kp = 3.0   
Kd = 1.5   
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
    non_zeros = np.nonzero(left_line)[0]
    if len(non_zeros) == 0:
        return -1, y
    left_x = int(np.mean(non_zeros)) # Bắt trung bình cộng vạch đường thay vì điểm cạnh
    return left_x, y


def find_right_lane(edges):
    height, width = edges.shape
    y = int(height * SCAN_Y_RATIO_right)
    line = edges[y]
    mid = len(line) // 2
    right_line = line[mid:]
    non_zeros = np.nonzero(right_line)[0]
    if len(non_zeros) == 0:
        return -1, y
    right_x = mid + int(np.mean(non_zeros)) # Bắt chính giữa dải băng đường
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

    print("📹 Bật preview camera. Nhấn 'Space' để bắt đầu bám làn, 'q' để thoát.")
    preview_mode = True

    while preview_mode and running:
        frame = get_camera_frame(picam2, cap)
        if frame is None:
            continue
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        cv2.putText(frame, "Press 'Space' to start", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Camera Preview", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == 32: # Space
            preview_mode = False
        elif key == ord('q'):
            running = False
            preview_mode = False

    cv2.destroyWindow("Camera Preview")

    if not running:
        print("❌ Kết thúc do người dùng yêu cầu.")
        return

    print("🚗 Bắt đầu bám làn (RUNNING).")
    
    # --- Khởi tạo State Machine & Variables ---
    car_state = "RUNNING"
    last_lane = "right"  # Mặc định xoay phía này khi mất cả 2 làn ngay từ đầu
    prev_error = 0
    # ----------------------------------------
    
    try:
        while running:
            frame = get_camera_frame(picam2, cap)
            if frame is None:
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
            # Đọc phím bấm (waitKeyEx bắt được các phím mũi tên)
            key = cv2.waitKeyEx(1)
            
            if key in [ord('q'), ord('Q')]:
                running = False
                break
            elif key == 32: # Space
                if car_state in ["PAUSED", "MANUAL"]:
                    car_state = "RUNNING"
                    print("▶️ Đã TRỞ LẠI trạng thái TỰ ĐỘNG BÁM LÀN (RUNNING)")
                else:
                    car_state = "PAUSED"
                    drive_motor(0, 0)
                    print("⏸ Đã TẠM DỪNG (PAUSED)")
            elif key in [ord('s'), ord('S')]: # Screenshot
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(os.path.dirname(__file__), f"screenshot_{ts}.jpg")
                cv2.imwrite(filename, frame)
                print(f"📸 Đã lưu khung hình: {filename}")
            
            # --- Các lệnh điểu khiển Manual Override ---
            elif key in [65362, 82, ord('w'), ord('W')]: # Lên / W
                car_state = "MANUAL"
                drive_motor(BASE_SPEED, 0)
                time.sleep(0.05)
                drive_motor(0, 0)
            elif key in [65361, 81, ord('a'), ord('A')]: # Trái / A
                car_state = "MANUAL"
                drive_motor(0, 80) # Xoay trái
                time.sleep(0.05)
                drive_motor(0, 0)
            elif key in [65363, 83, ord('d'), ord('D')]: # Phải / D
                car_state = "MANUAL"
                drive_motor(0, -80) # Xoay phải
                time.sleep(0.05)
                drive_motor(0, 0)
                
            # Nếu đang ở chế độ MANUAL / PAUSED thì hiển thị trạng thái và cấm code auto bám làn chạy
            if car_state in ["PAUSED", "MANUAL"]:
                display_text = "PAUSED" if car_state == "PAUSED" else "MANUAL OVERRIDE"
                cv2.putText(frame, f"{display_text} (Space to Auto)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow("Lane following combined", frame)
                continue

            # ==========================================
            #   BƯỚC NÀY CHỈ CHẠY KHI car_state == "RUNNING"
            # ==========================================
            
            # Cắt ảnh làm đen nửa trên (ROI - Region of Interest) để loại bỏ nhiễu
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_top = FRAME_HEIGHT // 2  # Chỉ giữ lại 1/2 khung hình từ dưới lên
            gray[0:roi_top, :] = 0
            
            # Lọc nhiễu Gaussian để tránh hột trên thảm/sàn nhà
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Dò tìm góc cạnh (Canny)
            edges = cv2.Canny(blurred, 100, 200)

            # Tìm tọa độ đường trên dòng quét cố định
            left_x, y_left = find_left_lane(edges)
            right_x, y_right = find_right_lane(edges)
            y = y_left

            # Vẽ UI hỗ trợ debug
            cv2.line(frame, (0, y), (FRAME_WIDTH, y), (80, 80, 80), 1)
            center_x = FRAME_WIDTH // 2
            cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (255, 0, 0), 1)
            
            cv2.line(frame, (TARGET_LEFT, y-10), (TARGET_LEFT, y+10), (0, 255, 255), 2)
            cv2.putText(frame, "Target Left", (TARGET_LEFT-30, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            cv2.line(frame, (TARGET_RIGHT, y-10), (TARGET_RIGHT, y+10), (0, 255, 255), 2)
            cv2.putText(frame, "Target Right", (TARGET_RIGHT-40, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Khởi tạo thông số controller
            mode = "No lane"
            steering = 0
            
            # Áp dụng bộ điều khiển PD (Proportional - Derivative)
            if right_x != -1:
                mode = "right"
                last_lane = "right"
                error = right_x - TARGET_RIGHT
                steering = - (Kp * error + Kd * (error - prev_error))
                prev_error = error
                cv2.circle(frame, (right_x, y), 5, (0, 255, 0), -1)
                
            elif left_x != -1:
                mode = "left"
                last_lane = "left"
                error = left_x - TARGET_LEFT
                steering = - (Kp * error + Kd * (error - prev_error))
                prev_error = error
                cv2.circle(frame, (left_x, y), 5, (0, 127, 255), -1)
                
            else:
                mode = "search"
                prev_error = 0 # reset error chống tích lũy nhiễu đạo hàm 

            if mode in ["right", "left"]:
                drive_motor(BASE_SPEED, steering)
                cv2.putText(frame, f"Mode: {mode} lane", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Steer: {int(steering)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Xoay ngược hướng bám làn cũ khi mất dấu
                spin_steering = 80 if last_lane == "right" else -80
                drive_motor(0, spin_steering)
                cv2.putText(frame, f"Mode: SEARCH ({last_lane})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                spin_dir = "left" if spin_steering > 0 else "right"
                cv2.putText(frame, f"Spinning {spin_dir} to find lane", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Render ảnh
            cv2.imshow("Lane following combined", frame)
            cv2.imshow("Edge", edges)

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
