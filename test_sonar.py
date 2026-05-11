import lgpio
import time
import sys

# Cấu hình các chân GPIO
TRIGGER_PIN = 23
ECHO_PIN = 24

def measure_distance():
    # Mở kết nối với gpiochip
    h = lgpio.gpiochip_open(0)
    
    try:
        # Thiết lập TRIGGER là OUTPUT, ban đầu ở mức thấp
        lgpio.gpio_claim_output(h, TRIGGER_PIN)
        lgpio.gpio_write(h, TRIGGER_PIN, 0)
        
        # Thiết lập ECHO là INPUT
        lgpio.gpio_claim_input(h, ECHO_PIN)
        
        print(f"Đang bắt đầu đo khoảng cách (Trigger: GPIO {TRIGGER_PIN}, Echo: GPIO {ECHO_PIN})...")
        print("Nhấn Ctrl+C để dừng.")
        
        # Chờ cảm biến ổn định
        time.sleep(0.5)
        
        while True:
            # Gửi xung Trigger (ít nhất 10us)
            lgpio.gpio_write(h, TRIGGER_PIN, 1)
            time.sleep(0.00001) # 10 microseconds
            lgpio.gpio_write(h, TRIGGER_PIN, 0)
            
            # Khởi tạo thời gian
            pulse_start = time.time()
            pulse_end = time.time()
            
            # Chờ Echo lên mức 1
            timeout_start = time.time()
            while lgpio.gpio_read(h, ECHO_PIN) == 0:
                pulse_start = time.time()
                if pulse_start - timeout_start > 0.5: # Timeout 0.5s
                    print("Lỗi: Không nhận được tín hiệu Echo (timeout).")
                    break
            
            # Chờ Echo xuống mức 0
            while lgpio.gpio_read(h, ECHO_PIN) == 1:
                pulse_end = time.time()
                if pulse_end - pulse_start > 0.5: # Timeout
                    break
            
            # Tính toán khoảng cách
            pulse_duration = pulse_end - pulse_start
            
            # Tốc độ âm thanh x thời gian / 2 (đi và về)
            # Tốc độ âm thanh ~ 34300 cm/s
            distance = pulse_duration * 17150
            distance = round(distance, 2)
            
            if 2 < distance < 400: # Khoảng cách hoạt động hiệu quả của HC-SR04
                print(f"Khoảng cách: {distance} cm")
            else:
                print("Ngoài phạm vi đo (2cm - 400cm)")
            
            time.sleep(0.5) # Đợi 0.5s cho lần đo tiếp theo

    except KeyboardInterrupt:
        print("\nĐã dừng chương trình.")
    finally:
        # Đóng gpiochip
        lgpio.gpiochip_close(h)

if __name__ == "__main__":
    measure_distance()
