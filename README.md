# 🚗 Hệ Thống Xe Tự Hành - LoPi (Hackathon 2026)

Hệ thống xe tự hành thông minh chạy trên nền tảng Raspberry Pi, tích hợp Camera ML nhận diện biển báo giao thông, RFID định vị vị trí và thuật toán bám làn đường nâng cao (HD Lane Following). Đi kèm một Dashboard hoàn chỉnh viết bằng Streamlit giúp điều hướng, cấu hình và tinh chỉnh bằng giao diện trực quan 100%.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi-4%2F5-A22846?logo=raspberrypi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B?logo=streamlit&logoColor=white)

---

## 🔑 Tính năng nổi bật

| Chức năng | Mô tả |
| --- | --- |
| 🛞 **Bám Làn Siêu Nét HD** | Thuật toán nhận diện làn đường bằng Canny Edge, hiệu chỉnh Bird-eye view, xử lý tốc độ cao ổn định. |
| 🛑 **Nhận Hiện Biển Báo** | Tích hợp thư viện xử lý phân loại biển báo trên đường. |
| 📡 **Định vị & Lập Bản Đồ RFID** | Quét thẻ RFID MFRC522 để đồng bộ vị trí xe với "Bản đồ ảo" ngay lập tức. |
| 🚀 **Web App Tích Hợp** | Toàn bộ hệ thống gói gọn trong duy nhất 1 Web UI chạy bằng Streamlit. Không cần chạy các file đa luồng rời rạc phức tạp! |
| 🎮 **Lái Xe Bằng Gamepad (WASD)** | Lái xe thủ công từ điện thoại hoặc màn hình cảm ứng để phục vụ việc debug điểm ảnh. |

---

## 🏗 Kiến Trúc Hệ Thống Chuẩn Hóa
Toàn bộ mã nguồn đã được tái cấu trúc dồn về một vị trí duy nhất:

```text
Hackathon26/
├── app.py                  # Dựng toàn bộ API Web điều khiển (Khởi chạy CHÍNH từ file này)
├── autonomous_main.py      # Lõi Xe tự hành AutonomousCar xử lý NavEngine, HD Lane Following
├── core/                   # Các logic toán học: navigation, sign classifier, sign detector.
├── graph/                  # GraphManager và Dijkstra Pathfinding
├── hardware/               # Driver API điều khiển Motor(lgpio), Camera(libcamera), RFID(SPI)
├── ui/                     # Widget, Dashboard Graph dành cho nền tảng Streamlit app.
├── data/                   # JSON cấu hình tự động (graph.json, turn_config.json) & Recordings.
└── templates/              # Thư viện ảnh biển báo (Feature matching AI detection)
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Cài đặt Môi trường (Cần Thiết)
Flash Raspberry Pi OS có Desktop hoặc cài hệ thông chuẩn trên Pi 5.

```bash
sudo raspi-config
# Bật SPI trong Interfacing Options.

sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-lgpio python3-spidev
```

Clone và cài thư viện:
```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install numpy pandas streamlit picamera2 opencv-python networkx plotly
```

### 2. Khởi Chạy Hệ Thống Duy Nhất
Mọi tính năng từ Camera cho tới Điều Hướng nay đã tích hợp trong Website. Bật 1 lệnh duy nhất sau ở máy Raspberry:

```bash
source venv/bin/activate
streamlit run app.py
```

### 3. Thao tác trên Web

1. **Truy cập IP**: `http://<IP-Của-Pi>:8501`
2. **Tìm đường thông minh**: Qua Tab `Bản đồ cốt lõi` nhấn chọn điểm xuất phát và chọn ngã rẽ. 
3. **Cấu hình xe gốc**: Qua Tab `Calibration` có tính năng tính "Pixel per second" để bạn cho xe chạy thẳng đo thời gian hoặc tự động cài test góc cua (`90°` vuông rẽ trái hoặc `180°` quay lại) cho phù hợp với pin vật lý.
4. **Lái thủ công**: Vào Tab `Điều khiển WASD`, nhấp vào vùng kích hoạt lái, thử di chuyển xe nhẹ nhàng.

---

## 🔧 Xử Lý Sự Cố Khẩn 

1. **`ModuleNotFoundError: ...`**: Ensure dependencies in the Python Virtual Environment are correctly bound.
2. **Xe quá giật khi bám làn**: Tinh chỉnh lại ngưỡng PID trong hàm `follow_lane()` file `autonomous_main.py`, đặc biệt là `base_speed`.
3. **`Camera in Configured state`**: Có thể thư viện libcamera OS chưa giải phóng kịp session cũ. Hãy giết process `pkill -f streamlit` và bật lại.`
