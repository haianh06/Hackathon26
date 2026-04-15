import threading
import os
# Xóa bỏ gán cứng QT_QPA_PLATFORM để hệ điều hành (hiện tại là Bookworm dùng Wayland) tự động nhận diện
if "QT_QPA_PLATFORM" in os.environ:
    del os.environ["QT_QPA_PLATFORM"]
import time
import cv2

from servo import joystick_loop, stop
from sonar import get_distance
from rfid import RFIDReader
from gpio_handle import gpio_close

# Global control
running = True
rfid = RFIDReader()

# Shared data
latest_distance = None
distance_lock = threading.Lock()

# ================= SONAR THREAD =================
def sonar_task():
    global latest_distance
    while running:
        d = get_distance()
        if d is not None:
            with distance_lock:
                latest_distance = d

            if d < 20:
                print(f"⚠️ Obstacle: {d:.1f}cm – STOP")
                stop()
        time.sleep(0.2)

# ================= RFID THREAD =================
def rfid_task():
    while running:
        try:
            # Nếu đang tắt máy, dừng ngay lập tức
            if not running: break 
            
            uid = rfid.read_uid_hex(timeout=0.5)
            if uid and running:
                with distance_lock:
                    d = latest_distance
                dist_str = f"{d:.1f} cm" if d is not None else "N/A"
                print(f"🆔 UID: {uid} | 📏 Distance: {dist_str}")
                time.sleep(1) 
        except Exception:
            # Khi shutdown, file descriptor bị đóng là bình thường, không cần in lỗi
            if running:
                print("⚠️ RFID Error: Communication lost")
            time.sleep(1)

# ================= CAMERA & DISPLAY (MAIN THREAD) =================
def run_camera_display():
    global running
    
    # Ưu tiên sử dụng Picamera2 (thư viện libcamera chính thức cho Python trên Raspberry Pi)
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        # Align with HD standalone script: dynamic scaling to preserve wide FOV
        sensor_res = picam2.sensor_resolution
        scale = 800 / sensor_res[0] if sensor_res[0] > 800 else 1.0
        target_size = (int(sensor_res[0] * scale), int(sensor_res[1] * scale))
        
        picam2.configure(config)
        try:
            if "LensPosition" in picam2.controls:
                picam2.set_controls({"LensPosition": 0.5})
        except: pass
        picam2.start()
        print(f"📸 libcamera Stream Started ({target_size[0]}x{target_size[1]}).")

        while running:
            try:
                # Lấy frame trực tiếp từ thư viện libcamera
                frame = picam2.capture_array()
                
                # Chuyển đổi hệ màu từ RGB sang BGR để OpenCV hiển thị đúng
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Chỉ khởi chạy cửa sổ đồ hoạ nếu có môi trường graphic (Desktop) hoặc SSH X-forwarding
                if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
                    cv2.imshow("Robot Monitor", frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        running = False
                        break
                else:
                    # Headless mode: Không display để tránh crash, vẫn giữ vòng lặp mượt mà
                    time.sleep(0.01)
            except Exception as e:
                print(f"⚠️ Error reading frame: {e}")
                time.sleep(0.1)

        picam2.stop()
        cv2.destroyAllWindows()
        return
        
    except ImportError:
        print("⚠️ Không tìm thấy Picamera2, chuyển sang dùng OpenCV Capture dự phòng...")
    except Exception as e:
        print(f"⚠️ Lỗi khởi tạo Picamera2: {e}, thử dùng fallback...")

    # Fallback dự phòng: OpenCV Capture với GStreamer (libcamerasrc) hoặc V4L2
    try:
        # Pipeline GStreamer sử dụng libcamera
        pipeline = "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            # Sử dụng V4L2 dự phòng truyền thống
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("❌ Error: Could not open camera.")
            return

        print("📸 Camera Stream Started (OpenCV Fallback).")

        while running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
                cv2.imshow("Robot Monitor", frame)
            
                # Thoát khi nhấn 'q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    running = False
                    break
            else:
                time.sleep(0.01)

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"❌ Lỗi vòng lặp fallback camera: {e}")


# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    t_joy = None 
    try:
        t_sonar = threading.Thread(target=sonar_task, daemon=True)
        t_rfid = threading.Thread(target=rfid_task, daemon=True)
        t_joy = threading.Thread(target=joystick_loop, daemon=True)

        t_sonar.start()
        t_rfid.start()
        t_joy.start()
        
        run_camera_display()

    except KeyboardInterrupt:
        pass
    finally:
        running = False # Phát lệnh dừng toàn cục
        print("\n⏳ Shutting down safely...")
        
        # Đợi thread joystick thoát (Quan trọng: Không dùng timeout quá ngắn)
        if t_joy and t_joy.is_alive():
            t_joy.join(timeout=2.0) 
        
        # Chỉ gọi stop() nếu handle vẫn khả dụng
        try:
            stop() 
        except:
            pass

        rfid.cleanup()
        
        # Đảm bảo đây là dòng cuối cùng được thực thi
        print("🛑 Closing GPIO handles...")
        gpio_close() 
        print("🛑 System shutdown complete")