# Hackathon 2026: Autonomous Navigation System (LoPi)

A high-performance autonomous robot control system built on Raspberry Pi 5. Features computer vision-based lane following, RFID-guided navigation, and a comprehensive Streamlit dashboard.

---

## 🏗 System Architecture

The project is organized into modular layers for perception, navigation, and hardware control:

- `app.py`: Integrated Web UI and system entry point.
- `autonomous_main.py`: Core mission logic and high-speed lane perception.
- `core/`: Vision processing (Sign detection/classification) and navigation engines.
- `graph/`: Dijkstra-based pathfinding and graph management.
- `hardware/`: Drivers for Motor control (LGPIO), Camera (libcamera), and RFID (SPI).
- `data/`: Real-time configuration (JSON maps, path data) and mission recordings.

---

## 🚀 Getting Started

### 1. System Requirements
Requires Raspberry Pi OS (Pi 4 or Pi 5). Enable SPI via `raspi-config`.

### 2. Installation

First, install **uv** (a fast Python package manager):
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Install system dependencies and setup the project:
```bash
# System dependencies (LGPIO, SPI, and Build headers)
sudo apt update && sudo apt install -y python3-lgpio python3-spidev libcap-dev

# Environment setup and dependencies
uv venv --system-site-packages
source .venv/bin/activate
source ~/.local/bin/env
uv pip install numpy streamlit picamera2 opencv-python networkx plotly
```

### 3. Launching the System
All modules are managed via the integrated Dashboard.
```bash
streamlit run app.py
```

---

## 🛠 Features

- **Advanced Lane Perception**: Dual-mode lane following (Basic Scan or Sliding Window with Polynomial Fitting).
- **RFID Virtual Positioning**: Real-time position syncing between the physical robot and the virtual map.
- **Mission 2026**: Fully autonomous round-trip mission (START → P1 → P2 → START) with automatic JSON logging of traffic signs and arrival timestamps.
- **Unified Control**: Real-time camera streaming, manual WASD overrides, and dynamic parameter tuning (Timings/Perspective) from a single Web UI.

---

## 🔧 Troubleshooting

- **Hardware Cleanup**: If the camera state persists after a crash, clear processes using `pkill -f streamlit`.
- **Latency**: Ensure the Pi 5 is powered by a stable 5V/5A supply for optimal libcamera performance.
