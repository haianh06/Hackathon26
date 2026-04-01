import cv2
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from manual_control.servo import _set_pwm, stop, LEFT_PIN, RIGHT_PIN
from manual_control.gpio_handle import gpio_close
import numpy as np

# ============ CÁC THÔNG SỐ ĐIỀU KHIỂN ============
# Tham số PID cho hệ thống lái
Kp = 0.8  # Tăng cái này nếu phản ứng rẽ quá chậm, giảm nếu xe lắc lư qua lại liên tục
Kd = 0.2  # Giúp xe làm dịu bớt (thắng lại) sự rung lắc
Ki = 0.0  # Thông thường không cần dùng I (Ki = 0) cho bám làn

BASE_SPEED = 120       # Tốc độ đi thẳng cơ bản (Giới hạn từ 0-500, khuyên dùng tầm 100-200)
TARGET_LEFT = 50       # TỌA ĐỘ X MỐC CỐ ĐỊNH (Mục tiêu của Vạch Trái)
SCAN_Y_RATIO = 0.55    # Vị trí dòng quét ngang - 0.75 nghĩa là quét ở tầm 3/4 chiều cao mâm ảnh (gần xe nhất)

FRAME_WIDTH  = 320     
FRAME_HEIGHT = 240
# =================================================

prev_error = 0
sum_error = 0

def drive_motor(speed, steering):
    """ Hàm truyền lệnh tới động cơ """
    # Giới hạn góc lái tối đa, tránh xe bẻ lái giật cụng
    steering = max(-200, min(200, steering))
    
    # Ở servo.py, thông số 1500 là đứng yên.
    # Chiều tiến (Forward): Bánh L < 1500, Bánh R > 1500
    # ĐỂ RẼ PHẢI (steering > 0): Bánh Trái (bánh ngoài) phải chạy nhanh hơn, Bánh Phải (bánh trong) chạy chậm lại
    # Tức là left_us cần lùi xa 1500 hơn nữa (- steering), right_us cần tiến về 1500 (- steering)
    left_us = 1500 - speed - steering
    right_us = 1500 + speed - steering
    
    _set_pwm(LEFT_PIN, left_us)
    _set_pwm(RIGHT_PIN, right_us)

def find_left_lane(edges):
    """ Tìm vạch trái bằng cách lấy TARGET_LEFT làm mốc và quét lan tỏa sang 2 bên gần nhất """
    height, width = edges.shape
    y = int(height * SCAN_Y_RATIO)
    line = edges[y]
    mid = len(line) // 2
    right_line = line[mid:]
    left_line = line[:mid]
    right_x = mid + np.argmax(right_line)
    left_x = mid - np.argmax(left_line[::-1])
    if np.argmax(left_line[::-1]) == 0:
        return -1, y
    return left_x, y

def get_camera_frame(picam2, cap):
    if picam2:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cap.read()
        if not ret: return None
    return frame

def main():
    global prev_error, sum_error
    if "QT_QPA_PLATFORM" in os.environ: del os.environ["QT_QPA_PLATFORM"]

    picam2 = None
    cap = None

    # Khởi tạo camera
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        sensor_res = picam2.sensor_resolution
        scale = 800 / sensor_res[0] if sensor_res[0] > 800 else 1.0
        config = picam2.create_preview_configuration(main={"size": (int(sensor_res[0]*scale), int(sensor_res[1]*scale)), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        print("📸 Camera Stream Started (Picamera2).")
    except Exception as e:
        print("⚠️ Picamera2 Error, trying fallback OpenCV...")
        pipeline = "libcamerasrc ! video/x-raw, width=800, height=600, framerate=30/1 ! videoconvert ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened(): cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not picam2 and (not cap or not cap.isOpened()):
        print("❌ Cannot open camera")
        return

    running = True
    stop() # Đảm bảo xe đứng yên ở điểm xuất phát

    print("🚗 Bắt đầu chạy test THUẬT TOÁN PID BÁM LÀN TRÁI...")
    print("📹 Preview camera. Press 's' to start lane following, 'q' to quit.")
    
    preview_mode = True
    while preview_mode:
        frame = get_camera_frame(picam2, cap)
        if frame is None: continue
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        cv2.putText(frame, "Press 's' to start", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Camera Preview", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            preview_mode = False
        elif key == ord('q'):
            running = False
            preview_mode = False
    
    if not running:
        return
    
    try:
        while running:
            frame = get_camera_frame(picam2, cap)
            if frame is None: continue
            
            # Đưa về độ phân giải chuẩn nhỏ gọn để test PID mượt nhất
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            # Quét dòng đơn duy nhất để tìm line trái
            left_x, y = find_left_lane(edges)
            center_x = FRAME_WIDTH // 2

            # ================= SHOW TRỰC QUAN =================
            # Vẽ đường quét ngang
            cv2.line(frame, (0, y), (FRAME_WIDTH, y), (50, 50, 50), 1) 
            # Vẽ trục tâm xe (Màu Xanh Biển)
            cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (255, 0, 0), 1)
            # Vẽ MỐC ĐÍCH TARGET cố định (Màu Vàng) làm "điểm mốc đặt để làm threshold"
            cv2.line(frame, (TARGET_LEFT, y-10), (TARGET_LEFT, y+10), (0, 255, 255), 2)
            cv2.putText(frame, "Target Left", (TARGET_LEFT-15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            if left_x != -1:
                # ------ BỘ ĐIỀU KHIỂN PID BÁM MỐC TUYỆT ĐỐI ------
                # left_x > TARGET_LEFT: Vạch trái nằm sau điểm mốc -> Xe đang lệch phải -> Rẽ trái (error âm)
                # left_x < TARGET_LEFT: Vạch trái nằm lấn sang phải điểm mốc -> Xe đang lệch về mép trái -> Rẽ phải (error dương)
                print(left_x, y)
                steering = - (left_x - TARGET_LEFT) * 3
                drive_motor(BASE_SPEED, steering)
                
                # Text Log & Draw Focus
                cv2.circle(frame, (left_x, y), 5, (0, 255, 0), -1) # Điểm phát hiện (Xanh Lá)
                cv2.putText(frame, f"Err: {steering}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(frame, f"Steer: {int(steering)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                # Trạng thái MẤT DẤU - Rẽ vòng sang trái để xoay xe dò tìm lại vạch trái
                drive_motor(80, -80) # Tốc độ tiến 80, bù lái -80 (để xoay trái)
                cv2.putText(frame, "Searching Left Line...", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.imshow("Lane Keeping Left - Camera", frame)
            cv2.imshow("Lane Keeping Left - Edge", edges)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                running = False
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        print("🛑 Dừng xe và giải phóng tài nguyên...")
        stop()
        gpio_close()
        if picam2: picam2.stop()
        if cap: cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()