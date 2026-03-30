# Hackathon26 - Xe Tự Hành Raspberry Pi (LoPi)

Dự án LoPi hoạt động như một hệ thống xe tự hành (Raspberry Pi + phần cứng) kèm giao diện web điều khiển/quan sát.

## 1. Yêu cầu phần cứng

- Raspberry Pi 4/5 (có GPIO + CSI cam)
- MFRC522 (RFID)
- HC-SR04 (sonar)
- Servo + motor + driver (L298N, TB6612, ...)
- Cáp, breadboard, nguồn nuôi ổn định

## 2. Cài đặt hệ thống cơ bản

1. Flash Raspberry Pi OS bằng Raspberry Pi Imager
2. SSH vào Pi: `ssh pi@pi5.local`
3. Nếu cần reset key: `ssh-keygen -R pi5.local`
4. Chạy `sudo raspi-config`:
   - Interfacing Options -> SPI enable
   - (tuỳ chọn) I2C enable nếu dùng cảm biến khác
5. Khởi động lại: `sudo reboot`

## 3. Cài đặt phần mềm

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-lgpio python3-spidev -y
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install numpy pandas streamlit picamera2 opencv-python
```

> Nếu gặp cảnh báo "externally-managed-environment", dùng `--break-system-packages` trong môi trường đã activate.

## 4. Cấu trúc dự án

```
Hackathon26/
├── main.py              # Đọc sonar/RFID và điều khiển servo
├── gpio_handle.py
├── MFRC522.py
├── rfid.py
├── servo.py
├── sonar.py
└── find_your_way/
    ├── app.py          # Streamlit UI
    ├── autonomous_main.py
    ├── streamlit_remote_control.py
    ├── config/graph_data.py
    └── core/navigation.py
```

## 5. Chạy dự án

Terminal 1 (phần cứng):
```bash
source venv/bin/activate
python3 main.py
```

Terminal 2 (web):
```bash
source venv/bin/activate
export PATH=$PATH:~/.local/bin
python3 -m streamlit run find_your_way/app.py
```

Nếu `streamlit: command not found`, dùng `python3 -m streamlit run find_your_way/app.py`.

## 6. Tính năng hiện tại

- Tab Navigating: tính lại đường đi, chạy tự động (AutonomousCar)
- Tab Mapping RFID: quét UID, gán node
- Tab System Config: xem graph + turn table
- Tab Camera: hiện camera đang kết nối (Picamera2 ưu tiên, OpenCV fallback), nút refresh + auto 1s
- Đã bỏ điều khiển WASD trên web (cơ tại phần cứng / script riêng)

## 7. Khắc phục sự cố

- `Camera in Configured state ...`: dùng camera từ nút refresh, tắt auto nếu bị lock
- `No module named 'hardware'`: đã fix import trong MFRC522 (`from gpio_handle import gpio_open`)
- `OSError: Bad file descriptor`: gọi cleanup đúng luồng (đã chỉnh)

## 8. Mở rộng

- thêm overlay thông tin sensor (RFID, sonar, speed)
- stream hình ảnh liên tục với `st.experimental_rerun` / `st.empty`
- remote control qua websocket hoặc MQTT

---

Hoàn tất README mới, giờ bạn có thể chạy ngay và kiểm tra UI/camera/states.

## Cài Đặt Phần Mềm

1. **Cài đặt thư viện GPIO:**
   ```bash
   sudo apt install python3-lgpio -y
   sudo apt install python3-spidev -y
   ```

2. **Tạo môi trường ảo:**
   ```bash
   python3 -m venv venv --system-site-packages
   source venv/bin/activate  # Kích hoạt môi trường
   ```

3. **Cài đặt thư viện Python:**
   ```bash
   pip install numpy pandas streamlit picamera2 opencv-python
   ```

4. **Clone dự án:**
   ```bash
   git clone <repository-url>
   cd Hackathon26
   ```

## Cấu Trúc Dự Án

```
Hackathon26/
├── main.py                 # Script chính điều khiển phần cứng
├── gpio_handle.py          # Xử lý GPIO
├── MFRC522.py              # Thư viện RFID MFRC522
├── rfid.py                 # Đọc RFID
├── servo.py                # Điều khiển servo
├── sonar.py                # Cảm biến siêu âm
└── find_your_way/
    ├── app.py              # Ứng dụng Streamlit chính
    ├── autonomous_main.py  # Logic xe tự hành
    ├── streamlit_remote_control.py  # Điều khiển từ xa
    ├── config/
    │   └── graph_data.py   # Dữ liệu bản đồ và cấu hình
    └── core/
        ├── navigation.py   # Engine điều hướng
        └── hardware/
            ├── gpio_handle.py
            ├── mfrc522_lib.py
            ├── motor.py
            └── rfid.py
```

## Chạy Dự Án

### 1. Chạy Script Điều Khiển Phần Cứng

Script `main.py` chạy liên tục để giám sát cảm biến và RFID:

```bash
python3 main.py
```

Script này sẽ:
- Đọc khoảng cách từ cảm biến siêu âm
- Phát hiện thẻ RFID
- Điều khiển servo dựa trên joystick (nếu có)
- Hiển thị thông tin trên console

### 2. Chạy Ứng Dụng Web Streamlit

Ứng dụng web để điều khiển xe tự hành:

```bash
export PATH=$PATH:~/.local/bin
streamlit run find_your_way/app.py
```

Nếu báo lỗi "streamlit: command not found", dùng:

```bash
python3 -m streamlit run find_your_way/app.py
```

Ứng dụng bao gồm:
- **Tab Navigating:** Tìm đường đi giữa các điểm
- **Tab Mapping RFID:** Ánh xạ vị trí RFID
- **Tab System Config:** Cấu hình hệ thống
- **Tab Manual Control & Camera:** Điều khiển WASD + hiển thị camera

### 3. Chạy Đồng Thời

Để chạy toàn bộ hệ thống, mở hai terminal riêng biệt:

**Terminal 1 (Điều khiển phần cứng):**
```bash
python3 main.py
```

**Terminal 2 (Giao diện web):**
```bash
streamlit run find_your_way/app.py
```

## Cấu Hình

- Chỉnh sửa `find_your_way/config/graph_data.py` để cấu hình bản đồ và các điểm RFID
- Sử dụng giao diện web để cập nhật cấu hình động

## Lưu Ý Quan Trọng

- Đảm bảo kết nối đúng chân GPIO trên Raspberry Pi
- Kiểm tra quyền truy cập GPIO (có thể cần `sudo` cho một số thao tác)
- Xe sẽ dừng tự động khi phát hiện vật cản gần (< 20cm)
- Thẻ RFID được đọc liên tục để xác định vị trí

## Khắc Phục Sự Cố

- Nếu gặp lỗi GPIO: Kiểm tra phiên bản lgpio và quyền sudo
- Nếu Streamlit không chạy: Kiểm tra port 8501 có bị chiếm không
- Nếu cảm biến không hoạt động: Kiểm tra kết nối vật lý và điện áp

## Phát Triển Thêm

- Thêm camera để thị giác máy tính
- Tích hợp GPS cho điều hướng ngoài trời
- Cải thiện thuật toán tránh vật cản
- Thêm giao thức truyền dữ liệu không dây (Bluetooth/WiFi)