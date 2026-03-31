# 🚗 LoPi — Autonomous Vehicle on Raspberry Pi

**Hackathon 2026 Project**

Hệ thống xe tự hành sử dụng Raspberry Pi, tích hợp cảm biến siêu âm, RFID, camera và bộ điều khiển PID bám làn — kèm giao diện web Streamlit để giám sát & điều khiển từ xa.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi-4%2F5-A22846?logo=raspberrypi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📑 Mục Lục

- [Tổng Quan](#-tổng-quan)
- [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
- [Yêu Cầu Phần Cứng](#-yêu-cầu-phần-cứng)
- [Sơ Đồ Kết Nối GPIO](#-sơ-đồ-kết-nối-gpio)
- [Cài Đặt](#-cài-đặt)
- [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
- [Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
- [Thuật Toán Bám Làn (Lane Following)](#-thuật-toán-bám-làn-lane-following)
- [Giao Diện Web (Streamlit)](#-giao-diện-web-streamlit)
- [Khắc Phục Sự Cố](#-khắc-phục-sự-cố)
- [Phát Triển Thêm](#-phát-triển-thêm)

---

## 🔍 Tổng Quan

**LoPi** là dự án xe tự hành mini chạy trên nền tảng Raspberry Pi, được phát triển cho cuộc thi Hackathon 2026. Hệ thống kết hợp nhiều module phần cứng hoạt động song song (multi-threaded) để thực hiện:

| Chức năng | Mô tả |
|---|---|
| 🛞 **Điều khiển động cơ** | Servo PWM 2 bánh, hỗ trợ joystick & tự hành |
| 📡 **Định vị RFID** | Đọc thẻ RFID MFRC522 qua SPI để xác định vị trí trên bản đồ |
| 📏 **Tránh vật cản** | Cảm biến siêu âm HC-SR04, tự dừng khi phát hiện chướng ngại < 20 cm |
| 📸 **Thị giác máy tính** | Camera CSI + Canny Edge Detection + PID Controller để bám làn đường |
| 🌐 **Giao diện web** | Streamlit UI để tìm đường, ánh xạ RFID, cấu hình hệ thống |

---

## 🏗 Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────┐
│                   Raspberry Pi                      │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Sonar   │  │   RFID   │  │  Camera (CSI)    │  │
│  │  Thread  │  │  Thread  │  │  Main Thread     │  │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│       │              │                 │             │
│       └──────────────┼─────────────────┘             │
│                      ▼                               │
│            ┌──────────────────┐                      │
│            │  Motor Control   │                      │
│            │  (Servo PWM)     │                      │
│            └──────────────────┘                      │
│                                                      │
│  ┌───────────────────────────────────────────────┐   │
│  │  Streamlit Web UI (find_your_way/)            │   │
│  │  ├── Navigating (Dijkstra pathfinding)        │   │
│  │  ├── RFID Mapping                             │   │
│  │  └── System Config                            │   │
│  └───────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 Yêu Cầu Phần Cứng

| Linh kiện | Vai trò | Ghi chú |
|---|---|---|
| Raspberry Pi 4/5 | Bo xử lý trung tâm | Có GPIO 40-pin + CSI camera port |
| Camera Module (CSI) | Thị giác máy tính | Hỗ trợ Picamera2 / libcamera |
| MFRC522 | Đọc thẻ RFID | Kết nối qua SPI (spidev0.0) |
| HC-SR04 | Cảm biến siêu âm | Đo khoảng cách tránh vật cản |
| Servo / Motor + Driver | Điều khiển 2 bánh | PWM 50 Hz, driver L298N / TB6612 |
| Joystick (tùy chọn) | Điều khiển thủ công | Qua thư viện Pygame |
| Breadboard, cáp, nguồn | Hạ tầng | Nguồn ổn định 5V/3A cho Pi |

---

## 📌 Sơ Đồ Kết Nối GPIO

| Module | Pin (BCM) | Chức năng |
|---|---|---|
| **Servo Trái** | GPIO 12 | PWM — Bánh trái |
| **Servo Phải** | GPIO 13 | PWM — Bánh phải |
| **HC-SR04 TRIG** | GPIO 23 | Trigger siêu âm |
| **HC-SR04 ECHO** | GPIO 24 | Echo siêu âm |
| **MFRC522 RST** | GPIO 22 | Reset RFID |
| **MFRC522 SPI** | SPI0 (CE0) | MOSI / MISO / SCLK / CS |

---

## ⚙ Cài Đặt

### 1. Cấu hình Raspberry Pi OS

```bash
# Flash Raspberry Pi OS bằng Raspberry Pi Imager, sau đó SSH vào Pi
ssh pi@<hostname>.local

# Bật SPI (bắt buộc cho RFID MFRC522)
sudo raspi-config
# → Interfacing Options → SPI → Enable

sudo reboot
```

### 2. Cài đặt phần mềm hệ thống

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-lgpio python3-spidev
```

### 3. Tạo môi trường ảo & cài thư viện Python

```bash
# Clone dự án
git clone <repository-url>
cd Hackathon26

# Tạo virtualenv (kế thừa system packages cho lgpio, spidev)
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Cài thư viện Python
pip install numpy pandas streamlit picamera2 opencv-python pygame
```

> **Lưu ý:** Nếu gặp lỗi `externally-managed-environment`, thêm flag `--break-system-packages` khi chạy pip.

---

## 📁 Cấu Trúc Dự Án

```
Hackathon26/
│
├── main.py                  # Chương trình chính — chạy đa luồng: sonar, RFID, camera, joystick
├── lane_follow.py           # Thuật toán bám làn đường bằng PID + Canny Edge Detection
├── servo.py                 # Điều khiển servo PWM 2 bánh + joystick loop
├── sonar.py                 # Đọc khoảng cách từ cảm biến siêu âm HC-SR04
├── rfid.py                  # Wrapper đọc UID thẻ RFID (hex string)
├── MFRC522.py               # Driver thư viện MFRC522 (SPI low-level)
├── gpio_handle.py           # Singleton quản lý GPIO handle (lgpio)
│
└── find_your_way/           # Module giao diện web & điều hướng tự động
    ├── app.py               # Ứng dụng Streamlit chính (3 tabs)
    ├── autonomous_main.py   # Logic xe tự hành — AutonomousCar class
    ├── config/
    │   └── graph_data.py    # Dữ liệu bản đồ: GRAPH, TURN_TABLE, RFID_MAP, TURN_CONFIG
    ├── core/
    │   └── navigation.py    # NavEngine — thuật toán Dijkstra tìm đường ngắn nhất
    └── hardware/
        ├── gpio_handle.py   # GPIO handle cho module web
        ├── mfrc522_lib.py   # Thư viện MFRC522 cho module web
        ├── motor.py         # Hàm điều khiển motor: move_straight, turn_left, turn_right, stop
        └── rfid.py          # RFIDReader cho module web
```

---

## 🚀 Hướng Dẫn Sử Dụng

### Chế độ 1: Chạy toàn bộ hệ thống (Phần cứng + Camera + Joystick)

```bash
source venv/bin/activate
python3 main.py
```

Chương trình `main.py` khởi chạy **4 tác vụ song song**:

| Thread | Chức năng |
|---|---|
| Sonar Thread | Đo khoảng cách liên tục, tự dừng xe khi < 20 cm |
| RFID Thread | Quét thẻ RFID và in UID ra console |
| Joystick Thread | Đọc tay cầm điều khiển (nút B để thoát) |
| Main Thread | Hiển thị video camera (nhấn `q` để thoát) |

### Chế độ 2: Bám làn đường (Lane Following)

```bash
source venv/bin/activate
python3 lane_follow.py
```

Xe sẽ tự động lái theo vạch kẻ bên phải sử dụng thuật toán PID + Canny Edge Detection. Nhấn `q` để dừng.

### Chế độ 3: Giao diện web (Điều hướng tự động)

```bash
source venv/bin/activate
streamlit run find_your_way/app.py
```

> Nếu gặp lỗi `streamlit: command not found`, dùng: `python3 -m streamlit run find_your_way/app.py`

### Chạy đồng thời (khuyến nghị)

Mở **2 terminal riêng biệt**:

```bash
# Terminal 1 — Phần cứng
source venv/bin/activate && python3 main.py

# Terminal 2 — Web UI
source venv/bin/activate && streamlit run find_your_way/app.py
```

---

## 🧠 Thuật Toán Bám Làn (Lane Following)

Hệ thống bám làn trong `lane_follow.py` hoạt động theo pipeline:

```
Camera Frame → Resize (320×240) → Grayscale → Canny Edge → Quét dòng ngang → PID → Motor
```

### Các thông số chính

| Thông số | Giá trị mặc định | Ý nghĩa |
|---|---|---|
| `Kp` | 0.8 | Hệ số tỷ lệ — Tăng nếu xe phản ứng chậm |
| `Kd` | 0.2 | Hệ số vi phân — Giảm rung lắc |
| `Ki` | 0.0 | Hệ số tích phân — Thường không cần cho bám làn |
| `BASE_SPEED` | 120 | Tốc độ đi thẳng (0–500, khuyên dùng 100–200) |
| `TARGET_X` | 300 | Tọa độ X mục tiêu của vạch phải trên ảnh |
| `SCAN_Y_RATIO` | 0.65 | Vị trí dòng quét ngang (0.65 = 65% chiều cao ảnh) |

### Nguyên lý

1. **Quét dòng đơn**: Lấy 1 dòng ngang tại `SCAN_Y_RATIO` trên ảnh Canny
2. **Tìm vạch phải**: Quét từ tâm ảnh ra 2 bên để phát hiện edge gần nhất bên phải
3. **Tính sai số**: `error = -(right_x - TARGET_X)` → Nếu vạch lệch khỏi mốc → sinh lái bù
4. **Điều khiển motor**: Tốc độ 2 bánh được điều chỉnh theo `steering = error × hệ số`
5. **Mất vạch**: Xe tự xoay phải để tìm lại vạch kẻ

---

## 🌐 Giao Diện Web (Streamlit)

Truy cập tại `http://<pi-ip>:8501` sau khi khởi chạy Streamlit.

| Tab | Chức năng |
|---|---|
| **Navigating** | Chọn điểm đầu/cuối → Tìm đường ngắn nhất (Dijkstra) → Bấm "Start Mission" để xe tự chạy |
| **Mapping RFID** | Quét thẻ RFID → Gán vào node trên bản đồ → Lưu cấu hình tự động |
| **System Config** | Xem cấu trúc đồ thị (GRAPH) và bảng rẽ (TURN_TABLE) dạng JSON |

---

## 🛠 Khắc Phục Sự Cố

| Lỗi | Nguyên nhân | Giải pháp |
|---|---|---|
| `No module named 'lgpio'` | Chưa cài hoặc venv không kế thừa system packages | `sudo apt install python3-lgpio` và tạo lại venv với `--system-site-packages` |
| `Camera in Configured state` | Camera bị lock bởi process khác | Tắt các process dùng camera rồi thử lại |
| `No module named 'hardware'` | Chạy từ sai thư mục | `cd` vào thư mục `find_your_way/` trước khi chạy Streamlit |
| `OSError: Bad file descriptor` | GPIO handle đã bị đóng trước khi thread kết thúc | Đã được xử lý trong code — đảm bảo dùng phiên bản mới nhất |
| `streamlit: command not found` | Streamlit chưa nằm trong PATH | Dùng `python3 -m streamlit run ...` hoặc `export PATH=$PATH:~/.local/bin` |
| Cảm biến siêu âm trả về `None` | Timeout / kết nối lỏng | Kiểm tra chân TRIG (GPIO 23) và ECHO (GPIO 24) |
| Xe không phản ứng joystick | Joystick không được nhận diện | Kiểm tra `pygame.joystick.get_count()` và kết nối USB/Bluetooth |

---

## 🔮 Phát Triển Thêm

- [ ] Overlay thông tin sensor (RFID, sonar, speed) lên video camera
- [ ] Streaming video realtime qua WebSocket hoặc MJPEG
- [ ] Điều khiển từ xa qua MQTT / WebSocket
- [ ] Tích hợp GPS cho điều hướng ngoài trời
- [ ] Cải thiện thuật toán tránh vật cản (multi-sensor fusion)
- [ ] Thêm giao thức Bluetooth/WiFi Direct cho truyền dữ liệu
- [ ] Ghi log hành trình và replay

---

<div align="center">

**Hackathon 2026** · Built with ❤️ on Raspberry Pi

</div>
