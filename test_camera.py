import cv2
import time
import os
import sys

# Thêm thư mục hiện tại vào path để import được các module trong dự án
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from hardware.camera import camera_manager
    USE_MANAGER = True
except ImportError:
    print("⚠️ Không tìm thấy hardware.camera. Sử dụng OpenCV thuần.")
    USE_MANAGER = False

def test_camera_with_manager():
    print("🚀 Đang khởi động CameraManager...")
    camera_manager.start()
    
    # Đợi camera khởi động
    time.sleep(2)
    
    print("📺 Đang hiển thị camera. Nhấn 'q' để thoát.")
    
    try:
        while True:
            frame = camera_manager.get_frame()
            
            if frame is not None:
                # Nếu là Picamera2 (RGB), cần chuyển sang BGR để OpenCV hiển thị đúng màu
                # Kiểm tra xem có Picamera2 hay không thông qua hardware.camera
                from hardware.camera import HAS_PICAMERA
                if HAS_PICAMERA and camera_manager.picam2 is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                cv2.imshow("Camera Test - Manager", frame)
            else:
                print("⌛ Đang đợi frame từ camera...")
                time.sleep(0.5)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera_manager.stop()
        cv2.destroyAllWindows()
        print("✅ Đã dừng CameraManager.")

def test_camera_raw():
    print("🚀 Đang khởi động OpenCV trực tiếp...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Không thể mở camera qua index 0.")
        return

    print("📺 Đang hiển thị camera. Nhấn 'q' để thoát.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Không thể đọc frame.")
                break
                
            cv2.imshow("Camera Test - Raw", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Đã đóng camera.")

if __name__ == "__main__":
    if USE_MANAGER:
        print("--- TEST CAMERA VỚI CAMERAMANAGER ---")
        try:
            test_camera_with_manager()
        except Exception as e:
            print(f"❌ Lỗi khi dùng Manager: {e}")
            print("\n--- THỬ LẠI VỚI OPENCV THUẦN ---")
            test_camera_raw()
    else:
        test_camera_raw()
