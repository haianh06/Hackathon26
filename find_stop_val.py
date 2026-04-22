import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import lgpio
from hardware.motor import init_motors, LEFT_PIN, RIGHT_PIN, PWM_FREQ

def set_pulse(h, pin, us):
    duty = (us / 20000) * 100
    lgpio.tx_pwm(h, pin, PWM_FREQ, duty)

def find_stop(h, name, pin):
    curr_val = 1500
    print(f"\n=== ĐANG TÌM ĐIỂM DỪNG (STOP_VAL) CHO MOTOR {name} (GPIO {pin}) ===")
    print("Sử dụng bàn phím:")
    print("  'w' : +1 us")
    print("  's' : -1 us")
    print("  'a' : -10 us")
    print("  'd' : +10 us")
    print("  'Enter' : Xong (Lưu giá trị này)")
    
    while True:
        set_pulse(h, pin, curr_val)
        print(f"  Giá trị hiện tại: {curr_val} us    ", end='\r')
        
        # Simple character input for Linux
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            
        if ch == 'w':
            curr_val += 1
        elif ch == 's':
            curr_val -= 1
        elif ch == 'd':
            curr_val += 10
        elif ch == 'a':
            curr_val -= 10
        elif ch == '\r' or ch == '\n':
            print(f"\n=> Đã chọn STOP_VAL cho motor {name}: {curr_val}")
            return curr_val
        elif ch == '\x03': # Ctrl+C
            raise KeyboardInterrupt

def main():
    h = init_motors()
    results = {}
    
    try:
        results['LEFT'] = find_stop(h, "TRÁI", LEFT_PIN)
        set_pulse(h, LEFT_PIN, 0) # Dừng hẳn để test motor kia
        
        results['RIGHT'] = find_stop(h, "PHẢI", RIGHT_PIN)
        set_pulse(h, RIGHT_PIN, 0)
        
        print("\n=== KẾT QUẢ CÂN CHỈNH ===")
        print(f"Motor TRÁI: {results['LEFT']} us")
        print(f"Motor PHẢI: {results['RIGHT']} us")
        print("\nBạn có thể cập nhật các giá trị này vào `hardware/motor.py` nếu chúng khác 1500.")
        
    except KeyboardInterrupt:
        print("\nĐã hủy.")
    finally:
        set_pulse(h, LEFT_PIN, 0)
        set_pulse(h, RIGHT_PIN, 0)

if __name__ == "__main__":
    main()
