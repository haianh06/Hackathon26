import streamlit as st
import os
import base64
import json
import sys
import time
import math
import cv2
import numpy as np

@st.cache_resource
def get_camera_manager():
    """Persistent camera manager instance."""
    from hardware.camera import camera_manager
    camera_manager.start()
    return camera_manager

try:
    from hardware.rfid import RFIDReader
    from autonomous_main import AutonomousCar
    import hardware.motor as motor
    import importlib
    import sys
    import threading
    HAS_HW = True
    HAS_RFID = True
    # Initial trigger to ensure manager is ready
    _ = get_camera_manager()
except Exception as e:
    print(f"Hardware init warning: {e}")
    HAS_HW = False
    HAS_RFID = False

@st.cache_resource
def get_rfid_reader():
    if HAS_RFID:
        return RFIDReader()
    return None

from graph.graph_manager import GraphManager
from graph.pathfinding import compute_shortest_path, compute_multi_stop_path, compute_sequential_path
from utils.rfid_simulator import RFIDSimulator
from ui.canvas import create_parking_lot_map
from ui.controls import render_sidebar_controls

st.set_page_config(layout="wide", page_title="Self-Driving Navigation System", page_icon="🚗")

# ─── Session State Initialization ─────────────────────────────────────────────
if "graph_manager" not in st.session_state:
    gm = GraphManager()
    if os.path.exists("data/graph.json"):
        gm.load_from_json("data/graph.json")
    st.session_state.graph_manager = gm

if "simulator" not in st.session_state:
    st.session_state.simulator = RFIDSimulator()

if "current_path" not in st.session_state:
    st.session_state.current_path = []

if "path_cost" not in st.session_state:
    st.session_state.path_cost = 0.0

if "click_step" not in st.session_state:
    st.session_state.click_step = 0

if "s_node" not in st.session_state:
    st.session_state.s_node = None
if "e_node" not in st.session_state:
    st.session_state.e_node = None
if "initial_heading" not in st.session_state:
    st.session_state.initial_heading = 0

if "manual_last_cmd" not in st.session_state:
    st.session_state.manual_last_cmd = "STOP"

# RFID pin wizard state
if "rfid_pending" not in st.session_state:
    st.session_state.rfid_pending = None

# Node editing state
if "editing_node" not in st.session_state:
    st.session_state.editing_node = None

# ── Multi-stop state ──────────────────────────────────────────────────────────
if "multi_waypoints" not in st.session_state:
    st.session_state.multi_waypoints = []      # list of node IDs selected by user

if "multi_start" not in st.session_state:
    st.session_state.multi_start = None        # starting node for multi-stop tour

if "tour_visit_order" not in st.session_state:
    st.session_state.tour_visit_order = []     # optimized visit order (node IDs)

if "segment_breaks" not in st.session_state:
    st.session_state.segment_breaks = []       # index positions in current_path for each leg

if "multi_click_step" not in st.session_state:
    st.session_state.multi_click_step = 0      # 0=wait for start, 1=collecting waypoints

# ── Turn config persistence ───────────────────────────────────────────────────
# ... existing turn config init ...

# ── Route Persistence Helpers ─────────────────────────────────────────────────
ROUTES_FILE = "data/routes.json"

def load_saved_routes():
    if not os.path.exists(ROUTES_FILE):
        return {}
    try:
        with open(ROUTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_route(name, data):
    routes = load_saved_routes()
    routes[name] = data
    os.makedirs(os.path.dirname(ROUTES_FILE), exist_ok=True)
    with open(ROUTES_FILE, "w", encoding="utf-8") as f:
        json.dump(routes, f, indent=4, ensure_ascii=False)

def delete_route(name):
    routes = load_saved_routes()
    if name in routes:
        del routes[name]
        with open(ROUTES_FILE, "w", encoding="utf-8") as f:
            json.dump(routes, f, indent=4, ensure_ascii=False)
        return True
    return False
TURN_CONFIG_PATH = "data/turn_config.json"

def _load_turn_config() -> dict:
    """Load turn timing from JSON file, fallback to defaults."""
    defaults = {"90_DEG": 1.7, "180_DEG": 1.8, "STRAIGHT": 1.0}
    try:
        if os.path.exists(TURN_CONFIG_PATH):
            with open(TURN_CONFIG_PATH) as f:
                data = json.load(f)
                return {**defaults, **data}
        return defaults
    except Exception:
        return defaults

def _save_turn_config(cfg: dict):
    import json
    os.makedirs(os.path.dirname(TURN_CONFIG_PATH), exist_ok=True)
    with open(TURN_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)

if "turn_config" not in st.session_state:
    st.session_state.turn_config = _load_turn_config()

if "lane_mode" not in st.session_state:
    st.session_state.lane_mode = "basic"

if "perspective_src" not in st.session_state:
    # Defaults: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    st.session_state.perspective_src = [
        [120, 160], [200, 160], 
        [300, 230], [20, 230]
    ]

# ── Media & Recording Initialization ──────────────────────────────────────────
REC_DIR = "data/recordings"
SNAP_DIR = "data/snapshots"
os.makedirs(REC_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "video_writer" not in st.session_state:
    st.session_state.video_writer = None
if "recording_file" not in st.session_state:
    st.session_state.recording_file = ""

def stop_recording():
    if st.session_state.video_writer:
        st.session_state.video_writer.release()
        st.session_state.video_writer = None
    st.session_state.is_recording = False
    st.session_state.recording_file = ""

# ── Camera Feed & Status Fragments ──────────────────────────────────────────
@st.fragment(run_every=1.0)
def render_live_camera_and_status(show_media_controls=False, key_suffix="main"):
    if HAS_HW:
        with st.container(border=True):
            cam_col1, cam_col2 = st.columns([2, 1])
            with cam_col1:
                frame = None
                if st.session_state.get("car_instance"):
                    frame = getattr(st.session_state.car_instance, 'debug_frame', None)
                
                if frame is None:
                    frame = get_camera_manager().get_frame() if HAS_HW else None
                
                if frame is not None:
                    if st.session_state.is_recording and st.session_state.video_writer:
                        # OpenCV VideoWriter expects BGR, frame is BGR natively now
                        st.session_state.video_writer.write(frame)

                    # Convert to Base64 to bypass Streamlit's MediaFileStorageError
                    _, buffer = cv2.imencode('.jpg', frame)
                    img_base64 = base64.b64encode(buffer).decode()
                    
                    # Use a full-width div to ensure the image stretches to fill the column
                    st.markdown(
                        f"""
                        <div style="width: 100%; margin-bottom: 20px;">
                            <img src="data:image/jpeg;base64,{img_base64}" 
                                 style="width: 100%; display: block; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                            <div style="text-align: center; color: #888; font-size: 0.8em; margin-top: 5px;">Live Camera Feed</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.info("Chờ Camera khởi động...")
            with cam_col2:
                st.markdown("### 🚦 Status")
                is_auton = st.session_state.get("car_instance") and st.session_state.car_instance.is_active
                label_prefix = "Vehicle Mode" if key_suffix == "main" else "Vehicle Mode (Manual)"
                st.metric(label_prefix, "Autonomous" if is_auton else "Standby")
                
                if show_media_controls:
                    # ── Cấu hình thời gian (Timings) ──
                    with st.expander("⚙️ Cấu hình thời gian (Timings)", expanded=False):
                        cfg = st.session_state.turn_config
                        c1, c2, c3 = st.columns(3)
                        t_straight = c1.number_input("Đi thẳng (s):", 0.0, 5.0, float(cfg.get("STRAIGHT", 1.0)), 0.1)
                        t_90 = c2.number_input("Rẽ 90° (s):", 0.0, 5.0, float(cfg.get("90_DEG", 1.7)), 0.1)
                        t_180 = c3.number_input("Quay đầu (s):", 0.0, 5.0, float(cfg.get("180_DEG", 1.8)), 0.1)
                        
                        if st.button("💾 Lưu Cấu Hình Thời Gian", width='stretch'):
                            new_cfg = {"STRAIGHT": t_straight, "90_DEG": t_90, "180_DEG": t_180}
                            st.session_state.turn_config = new_cfg
                            _save_turn_config(new_cfg)
                            st.success("✅ Đã lưu cấu hình mới!")

                    # ── Cấu hình thuật toán bám làn & BEV ──
                    with st.expander("📐 Thuật toán bám làn & Hiệu chuẩn (BEV)", expanded=False):
                        l_mode = st.selectbox("Thuật toán bám làn:", ["basic", "sliding_window"], index=0 if st.session_state.lane_mode == "basic" else 1, key=f"mode_sel_{key_suffix}")
                        st.session_state.lane_mode = l_mode
                        
                        if l_mode == "sliding_window":
                            st.write("**Hiệu chuẩn Perspective (Trapezoid):**")
                            p = st.session_state.perspective_src
                            # Point 1 (Top-Left)
                            c1, c2 = st.columns(2)
                            p[0][0] = c1.slider("TL-X", 0, 160, int(p[0][0]), 1, key=f"tlx_{key_suffix}")
                            p[0][1] = c2.slider("TL-Y", 0, 240, int(p[0][1]), 1, key=f"tly_{key_suffix}")
                            # Point 2 (Top-Right)
                            c1, c2 = st.columns(2)
                            p[1][0] = c1.slider("TR-X", 160, 320, int(p[1][0]), 1, key=f"trx_{key_suffix}")
                            p[1][1] = c2.slider("TR-Y", 0, 240, int(p[1][1]), 1, key=f"try_{key_suffix}")
                            # Point 3 (Bottom-Right)
                            c1, c2 = st.columns(2)
                            p[2][0] = c1.slider("BR-X", 160, 320, int(p[2][0]), 1, key=f"brx_{key_suffix}")
                            p[2][1] = c2.slider("BR-Y", 0, 240, int(p[2][1]), 1, key=f"bry_{key_suffix}")
                            # Point 4 (Bottom-Left)
                            c1, c2 = st.columns(2)
                            p[3][0] = c1.slider("BL-X", 0, 160, int(p[3][0]), 1, key=f"blx_{key_suffix}")
                            p[3][1] = c2.slider("BL-Y", 0, 240, int(p[3][1]), 1, key=f"bly_{key_suffix}")
                            st.session_state.perspective_src = p
                            
                        if st.button("💾 Lưu Cấu Hình Bám Làn", width='stretch', key=f"save_lane_{key_suffix}"):
                            st.success("✅ Đã ghi nhớ cấu hình thuật toán!")

                    st.divider()
                    # Snapshot Button
                    if st.button("📸 Chụp Ảnh", width='stretch', key=f"snap_btn_{key_suffix}"):
                        if frame is not None:
                            fname = f"snap_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                            fpath = os.path.join(SNAP_DIR, fname)
                            cv2.imwrite(fpath, frame)
                            st.toast(f"✅ Đã lưu: {fname}")
                    
                    # Recording Toggle
                    if not st.session_state.is_recording:
                        if st.button("🔴 Ghi Hình", width='stretch', key=f"rec_btn_{key_suffix}"):
                            fname = f"rec_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                            fpath = os.path.join(REC_DIR, fname)
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            h, w = frame.shape[:2]
                            st.session_state.video_writer = cv2.VideoWriter(fpath, fourcc, 1.0, (w, h))
                            st.session_state.is_recording = True
                            st.session_state.recording_file = fname
                            st.rerun()
                    else:
                        st.error(f"⏺️ Đang ghi... ({st.session_state.recording_file})")
                        if st.button("⏹️ Dừng Ghi", width='stretch', key=f"stop_rec_btn_{key_suffix}"):
                            stop_recording()
                            st.rerun()

                if st.button("🔄 Refresh Stream", key=f"refresh_btn_{key_suffix}"):
                    st.toast("Đang tải lại Camera...")
                    try:
                        cm = get_camera_manager()
                        cm.stop()
                        from hardware.camera import CameraManager
                        CameraManager._instance = None
                        get_camera_manager.clear()
                    except Exception as e:
                        print(f"Lỗi refresh stream: {e}")
                    st.rerun()

@st.fragment(run_every=1.5)
def render_live_logs():
    st.markdown("### 📝 Live Console Logs")
    log_container = st.empty()
    
    # Sync simulator position with real car if running
    car_alive = 'car_thread' in st.session_state and st.session_state.car_thread.is_alive()
    if car_alive and 'car_instance' in st.session_state:
        curr = getattr(st.session_state.car_instance, 'current_node', None)
        if curr:
            st.session_state.simulator.force_scan(curr)

    if 'car_instance' in st.session_state:
        logs = st.session_state.car_instance.log_history
        log_container.code("\n".join(logs) if logs else "No logs yet...")
    else:
        log_container.info("Logs will appear here when route starts.")

@st.fragment
def render_wasd_manual_controls():
    # --- Consolidated Keyboard Focus Area & JS ---
    combined_html = """
    <div id="gamepad-focus-area" style="
        background-color: #1a1a1a;
        border: 2px solid #555;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        cursor: pointer;
        margin-bottom: 10px;
        transition: all 0.2s;
        user-select: none;
        font-family: sans-serif;
    " onclick="window.focus();">
        <h3 style="margin:0; color:#fff; font-size: 1.4em;">🎮 Vùng Kích Hoạt Bàn Phím</h3>
        <p style="margin:8px 0; color:#888; font-size: 1em;">Nhấn vào đây để lái bằng phím WASD</p>
        <div style="margin-top: 15px;">
            <div id="kb-status" style="
                display: inline-block;
                width: 14px;
                height: 14px;
                background-color: #ff4b4b;
                border-radius: 50%;
                margin-right: 10px;
                box-shadow: 0 0 10px #ff4b4b;
            "></div>
            <span id="kb-text" style="color:#eee; font-weight: bold; font-size: 1.1em;">Chưa sẵn sàng</span>
        </div>
    </div>

    <script>
    const pressedKeys = new Set();
    const statusDot = document.getElementById('kb-status');
    const statusText = document.getElementById('kb-text');
    const focusArea = document.getElementById('gamepad-focus-area');
    
    function updateStatus(active, key="") {
        if (!statusDot || !statusText) return;
        if (active) {
            statusDot.style.backgroundColor = '#00ff00';
            statusDot.style.boxShadow = '0 0 15px #00ff00';
            statusText.innerText = 'ĐANG NHẬN PHÍM: ' + key.toUpperCase();
            focusArea.style.borderColor = '#00ff00';
            focusArea.style.backgroundColor = '#1a2e1a';
        } else {
            statusDot.style.backgroundColor = '#ffbb00';
            statusDot.style.boxShadow = '0 0 15px #ffbb00';
            statusText.innerText = 'ĐÃ SẴN SÀNG (Thả phím)';
            focusArea.style.borderColor = '#ffbb00';
            focusArea.style.backgroundColor = '#1a1a1a';
        }
    }

    function triggerStreamlitButton(labelPart) {
        let found = false;
        const searchInDoc = (doc) => {
            if (!doc) return false;
            const buttons = doc.querySelectorAll('button');
            for (let btn of buttons) {
                if (btn.innerText && btn.innerText.includes(labelPart)) {
                    btn.click();
                    return true;
                }
            }
            return false;
        };
        if (searchInDoc(document)) found = true;
        if (!found && window.parent) found = searchInDoc(window.parent.document);
        if (!found && window.top) found = searchInDoc(window.top.document);
        return found;
    }

    function handleKey(e, isDown) {
        const key = e.key.toLowerCase();
        if (!['w','a','s','d'].includes(key)) return;
        e.preventDefault();
        if (isDown) {
            if (!pressedKeys.has(key)) {
                pressedKeys.add(key);
                updateStatus(true, key);
                let label = "";
                if (key === 'w') label = "W (Tiến)";
                else if (key === 'a') label = "A (Trái)";
                else if (key === 's') label = "S (Dừng)";
                else if (key === 'd') label = "D (Phải)";
                triggerStreamlitButton(label);
            }
        } else {
            if (pressedKeys.has(key)) {
                pressedKeys.delete(key);
                if (pressedKeys.size === 0) {
                    updateStatus(false);
                    triggerStreamlitButton("S (Dừng)");
                } else {
                    const lastKey = Array.from(pressedKeys).pop();
                    updateStatus(true, lastKey);
                }
            }
        }
    }

    const setReady = () => {
        window.focus();
        if (statusDot) {
            statusDot.style.backgroundColor = '#ffbb00';
            statusDot.style.boxShadow = '0 0 10px #ffbb00';
            statusText.innerText = 'ĐÃ SẴN SÀNG (Nhấn phím để lái)';
            focusArea.style.borderColor = '#ffbb00';
        }
    };

    if (focusArea) focusArea.addEventListener('mousedown', setReady);
    document.addEventListener('keydown', (e) => handleKey(e, true));
    document.addEventListener('keyup', (e) => handleKey(e, false));
    try {
        window.parent.document.addEventListener('keydown', (e) => {
            if (['w','a','s','d'].includes(e.key.toLowerCase())) setReady();
            handleKey(e, true);
        });
        window.parent.document.addEventListener('keyup', (e) => handleKey(e, false));
    } catch (err) {}
    </script>
    """
    import streamlit.components.v1 as components
    components.html(combined_html, height=200)

    st.info("Sử dụng **W, A, S, D** trên bàn phím (Nhấn để chạy, Thả để dừng).")
    
    # Simple WASD layout
    cw1, cw2, cw3 = st.columns([1, 1, 1])
    with cw2:
        if st.button("🔼 W (Tiến)", key="btn_w", width='stretch'):
            st.session_state.manual_last_cmd = "FORWARD"
            if HAS_HW: motor.move_straight()
    
    cl1, cl2, cl3 = st.columns([1, 1, 1])
    with cl1:
        if st.button("◀️ A (Trái)", key="btn_a", width='stretch'):
            st.session_state.manual_last_cmd = "LEFT"
            if HAS_HW: motor.turn_left()
    with cl2:
        if st.button("⏹️ S (Dừng)", key="btn_s", width='stretch'):
            st.session_state.manual_last_cmd = "STOP"
            if HAS_HW: motor.stop()
    with cl3:
        if st.button("▶️ D (Phải)", key="btn_d", width='stretch'):
            st.session_state.manual_last_cmd = "RIGHT"
            if HAS_HW: motor.turn_right()
    
    st.divider()
    st.write(f"Lệnh cuối cùng: **{st.session_state.manual_last_cmd}**")

# ── Main UI Layout ────────────────────────────────────────────────────────────
render_live_camera_and_status(key_suffix="main")

# ─────────────────────────────────────────────────────────────────────────────
gm: GraphManager = st.session_state.graph_manager
sim: RFIDSimulator = st.session_state.simulator

# Load UI Controls in Sidebar
mode = render_sidebar_controls(gm, sim, st.session_state.current_path, st.session_state.path_cost)

# Main App Panel
st.title("🚗 Self-Driving Vehicle Pathfinding Map")

col1, col2, col3 = st.columns(3)
col1.metric("Live Nodes", gm.graph.number_of_nodes())
col2.metric("Path Route Distance", f"{st.session_state.path_cost:.2f}")
col3.metric(
    "Vehicle Status",
    "In Motion" if sim.is_running else ("Idle" if not st.session_state.current_path else "Reached Destination")
)


tab_options = {
    "🗺️ Bản Đồ Cốt Lõi": "tab1",
    "⚙️ Config Dừng Xe (Calibration)": "tab2",
    "🎮 Điều Khiển WASD": "tab3",
    "🛑 Traffic Sign Debug": "tab4"
}

# Thay vì dùng st.tabs (chạy ngầm tất cả tab), dùng st.radio để chỉ chạy tab hiện tại
selected_tab_label = st.radio(
    "Chế độ hiển thị (Tabs):",
    list(tab_options.keys()),
    horizontal=True,
    label_visibility="collapsed"
)
active_tab = tab_options[selected_tab_label]

st.markdown("---")

if active_tab == "tab2":
    st.header("⚙️ Calibration — Tốc độ & Thời gian Rẽ")

    # ── Speed calibration ────────────────────────────────────────────────
    st.subheader("🚀 Cân chỉnh Tốc độ Thẳng")
    st.caption("Chạy thử xe 1 khoảng thời gian thực tế, đo cự ly vật lý, so đối chiếu 2 điểm trên web để lấy ra Cự ly Pixels. Lấy Pixel / Giây.")
    new_speed = st.number_input("Tốc độ xe thực tế (Pixel trên map / 1 giây):", value=float(gm.speed_px_per_sec), key="cal_speed")
    if st.button("💾 Lưu Tốc Độ"):
        gm.speed_px_per_sec = new_speed
        gm._auto_save()
        st.success("Lưu thành công!")

    st.divider()
    st.subheader("⏱️ Test Motor Thẳng")
    test_secs = st.number_input("Chạy test trong bao nhiêu giây:", value=2.0, key="cal_test_sec")
    if st.button("▶️ Chạy thử để đo"):
        if HAS_HW:
            import hardware.motor as motor
            motor.move_straight()
            time.sleep(test_secs)
            motor.stop()
            st.success(f"Đã chạy {test_secs} giây. Hãy đem thước ra đo trên sàn!")
        else:
            st.warning("Không có phần cứng thực tế.")

    st.divider()

    # ── Turn timing calibration ──────────────────────────────────────────
    st.subheader("🔄 Cân chỉnh Thời gian Rẽ (Turn Timing)")
    st.info(
        "Cơ chế: Nhấn **Test Rẽ** để xe rẽ trong khoảng thời gian bạn đặt. "
        "Chỉnh số giây đến khi xe rẽ đúng góc mong muốn, rồi nhấn **Lưu**."
    )

    turn_cfg = st.session_state.turn_config
    tc_col1, tc_col2 = st.columns(2)

    with tc_col1:
        with st.container(border=True):
            st.markdown("**↩️ Rẽ 90° (Góc vuông)**")
            new_90 = st.number_input(
                "Thời gian rẽ 90° (giây)",
                value=float(turn_cfg.get("90_DEG", 1.5)),
                min_value=0.1, max_value=10.0, step=0.05,
                key="cal_90deg",
                help="Thời gian motor rẽ để xe quay đúng 90 độ"
            )
            t_col1, t_col2 = st.columns(2)
            if t_col1.button("⬅️ Test Rẽ Trái 90°", width='stretch'):
                if HAS_HW:
                    import hardware.motor as motor
                    motor.turn_left()
                    time.sleep(new_90)
                    motor.stop()
                    st.success(f"Rẽ trái {new_90}s xong.")
                else:
                    st.warning("Không có HW.")
            if t_col2.button("➡️ Test Rẽ Phải 90°", width='stretch'):
                if HAS_HW:
                    import hardware.motor as motor
                    motor.turn_right()
                    time.sleep(new_90)
                    motor.stop()
                    st.success(f"Rẽ phải {new_90}s xong.")
                else:
                    st.warning("Không có HW.")

    with tc_col2:
        with st.container(border=True):
            st.markdown("**↪️ Quay đầu xe**")
            new_180 = st.number_input(
                "Thời gian quay đầu xe (giây)",
                value=float(turn_cfg.get("180_DEG", 0.5)),
                min_value=0.1, max_value=10.0, step=0.05,
                key="cal_180deg",
                help="Thời gian motor rẽ để xe quay đúng 180 độ"
            )
            t_col3, t_col4 = st.columns(2)
            if t_col3.button("⬅️ Test Quay đầu xe", width='stretch'):
                if HAS_HW:
                    import hardware.motor as motor
                    motor.turn_right()
                    time.sleep(new_180)
                    motor.turn_right()
                    time.sleep(new_180)
                    motor.stop()
                    st.success(f"Quay đầu xe {new_180}s xong.")
                else:
                    st.warning("Không có HW.")
    
    st.write("")
    if st.button("💾 Lưu Cấu Hình Rẽ", type="primary", width='stretch'):
        new_cfg = {"90_DEG": new_90, "180_DEG": new_180}
        _save_turn_config(new_cfg)
        st.session_state.turn_config = new_cfg
        st.success(f"✅ Đã lưu: 90° = {new_90}s | 180° = {new_180}s")

    st.caption(f"📌 Cấu hình hiện tại: 90° = {turn_cfg.get('90_DEG')}s | 180° = {turn_cfg.get('180_DEG')}s")

if active_tab == "tab1":
    # ──────────────────────────────────────────────
    # SECTION 1: RFID Node Registration Wizard
    # ──────────────────────────────────────────────
    st.subheader("📍 Đăng Ký Thẻ RFID Mới")

    with st.container(border=True):
        st.markdown("**Bước 1 — Khai báo thông tin thẻ**")
        form_col1, form_col2, form_col3 = st.columns([2, 3, 1])

        with form_col1:
            if st.session_state.get('last_scanned_uid'):
                st.session_state["main_scanner"] = st.session_state.last_scanned_uid
                st.session_state.last_scanned_uid = ""
            scanned_id = st.text_input(
                "🆔 ID thẻ RFID",
                placeholder="VD: AB-12-CD-EF",
                key="main_scanner",
                help="Quét hoặc nhập thủ công UID hex của thẻ RFID"
            )

        with form_col2:
            scanned_label = st.text_input(
                "🏷️ Tên / Nhãn vị trí",
                placeholder="VD: Cổng vào, Bãi A1, Ngã tư số 3...",
                key="rfid_label_input",
                help="Tên gợi nhớ để dễ nhận diện trên bản đồ"
            )

        with form_col3:
            st.write("")
            st.write("")
            if HAS_RFID:
                if st.button("💳 Quét (5s)", width='stretch'):
                    with st.spinner("Chờ thẻ..."):
                        reader = get_rfid_reader()
                        try:
                            uid = reader.read_uid_hex(timeout=5.0)
                            if uid:
                                st.session_state.last_scanned_uid = uid
                                st.rerun()
                            else:
                                st.error("Timeout!")
                        except Exception as e:
                            st.error(f"Lỗi: {e}")
            else:
                st.button("💳 Quét (Tắt)", disabled=True, width='stretch')

        id_ok = bool(scanned_id and scanned_id.strip())
        label_ok = bool(scanned_label and scanned_label.strip())
        pending = st.session_state.rfid_pending

        btn_col, status_col = st.columns([2, 3])
        with btn_col:
            if pending is None:
                lock_disabled = not (id_ok and label_ok)
                lock_help = "" if not lock_disabled else "Cần điền đầy đủ ID và Tên mới khóa được"
                if st.button(
                    "🔒 Khóa thông tin → Nhấp Map để ghim",
                    disabled=lock_disabled,
                    help=lock_help,
                    width='stretch',
                    type="primary"
                ):
                    uid_clean = scanned_id.strip()
                    if gm.is_uid_used(uid_clean):
                        st.error(f"❌ UID '{uid_clean}' đã được sử dụng!")
                    else:
                        st.session_state.rfid_pending = {
                            "id": uid_clean,
                            "label": scanned_label.strip()
                        }
                        st.rerun()
            else:
                if st.button("✖️ Hủy ghim", width='stretch'):
                    st.session_state.rfid_pending = None
                    st.rerun()

        with status_col:
            if pending:
                st.info(
                    f"📌 **Đang chờ ghim:** **{pending['label']}** `[{pending['id']}]`  "
                    f"\n👇 Nhấp vào bất kỳ điểm trống nào trên bản đồ bên dưới."
                )
            elif not id_ok or not label_ok:
                st.caption("⬅️ Điền đầy đủ ID và Tên thẻ rồi nhấn Khóa.")
            else:
                st.caption("✅ Thông tin đã điền. Nhấn **Khóa** để chuyển sang bước ghim.")

    # ── Cập nhật vị trí xe theo RFID ──
    with st.expander("🔄 Cập nhật vị trí xe theo thẻ RFID", expanded=False):
        upd_col1, upd_col2 = st.columns([3, 2])
        upd_id = upd_col1.text_input("UID thẻ tại vị trí xe hiện tại:", placeholder="Nhập UID hex...")
        upd_col2.write("")
        upd_col2.write("")
        if upd_col2.button("Cập nhật", width='stretch'):
            resolved_node = gm.get_node_by_uid(upd_id)
            if resolved_node and sim.force_scan(resolved_node):
                lbl = gm.graph.nodes[resolved_node].get('label', '')
                display = f"**{lbl}** `[{resolved_node}]`" if lbl else f"`[{resolved_node}]`"
                st.success(f"✅ Cập nhật vị trí xe thành công! (Node: {display})")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("❌ Thẻ không tồn tại trên Map hoặc không nằm trên lộ trình!")

    st.write("---")

    current_node = sim.get_current_node()
    start_node = st.session_state.get('s_node')
    end_node = st.session_state.get('e_node')

    # ──────────────────────────────────────────────
    # MAP MODE SELECTOR
    # ──────────────────────────────────────────────
    pending = st.session_state.rfid_pending
    if pending:
        click_action = "📍 Ghim thẻ RFID mới (Dùng mã đã khai báo ở trên)"
        st.info(f"🔒 Chế độ ghim đang chạy: `{pending['id']}` — _{pending['label']}_. Nhấp map để xác nhận vị trí.")
    else:
        click_action = st.radio(
            "Chế độ nhấp Map:",
            [
                "🚩 Lập Kế Hoạch Lộ Trình (3 Bước)",
                "⚪ Ghim điểm hướng dẫn (Waypoint - Không RFID)",
                "📍 Ghim thẻ RFID mới (Dùng mã đã khai báo ở trên)",
            ],
            horizontal=True,
            key="map_click_mode"
        )
    
    # ── ROUTE SAVING/LOADING UI ───────────────────
    with st.expander("📂 Quản Lý Lộ Trình Đã Lưu", expanded=False):
        saved_routes = load_saved_routes()
        
        if saved_routes:
            s_col1, s_col2 = st.columns([3, 1])
            selected_route_name = s_col1.selectbox("Chọn lộ trình:", list(saved_routes.keys()), label_visibility="collapsed")
            if s_col2.button("📂 Tải", width='stretch', type="primary"):
                r_data = saved_routes[selected_route_name]
                st.session_state.multi_start = r_data.get("multi_start")
                st.session_state.multi_waypoints = r_data.get("multi_waypoints", [])
                st.session_state.initial_heading = r_data.get("initial_heading", 0)
                st.session_state.multi_click_step = 2 # Jump to Step 3
                st.session_state.s_node = st.session_state.multi_start
                # Reset previous calculation results
                st.session_state.current_path = []
                st.session_state.tour_visit_order = []
                st.session_state.segment_breaks = []
                st.session_state.path_cost = 0.0
                st.success(f"✅ Đã tải lộ trình: **{selected_route_name}**")
                st.rerun()
            
            if st.button(f"🗑️ Xóa lộ trình '{selected_route_name}'", type="secondary", width='stretch'):
                if delete_route(selected_route_name):
                    st.toast(f"Đã xóa {selected_route_name}")
                    st.rerun()

        st.divider()
        st.markdown("**💾 Lưu lộ trình hiện tại:**")
        r_name = st.text_input("Tên lộ trình:", placeholder="Ví dụ: Cổng -> Kho A")
        save_disabled = not (st.session_state.multi_start and st.session_state.multi_waypoints)
        if st.button("💾 Lưu Lại", disabled=save_disabled, width='stretch'):
            if not r_name.strip():
                st.error("Vui lòng nhập tên lộ trình!")
            else:
                r_data = {
                    "multi_start": st.session_state.multi_start,
                    "multi_waypoints": st.session_state.multi_waypoints,
                    "initial_heading": st.session_state.initial_heading
                }
                save_route(r_name.strip(), r_data)
                st.success(f"✅ Đã lưu lộ trình: **{r_name}**")
                st.rerun()

    # ──────────────────────────────────────────────
    # ROUTE PLANNING PANEL (3 STEPS)
    # ──────────────────────────────────────────────
    if click_action == "🚩 Lập Kế Hoạch Lộ Trình (3 Bước)":
        with st.container(border=True):
            st.markdown("### 🚩 Lộ Trình Tuần Tự (3 Bước)")

            ms_col1, ms_col2 = st.columns([3, 2])

            with ms_col1:
                step = st.session_state.multi_click_step
                multi_start = st.session_state.multi_start
                multi_wps = st.session_state.multi_waypoints

                if step == 0:
                    st.info("📍 **Bước 1:** Nhấp vào một node trên bản đồ để chọn **điểm xuất phát**.")
                elif step == 1:
                    if multi_start and multi_start in gm.graph:
                        lbl = gm.graph.nodes[multi_start].get('label', '')
                        start_disp = f"**{lbl}** `[{multi_start}]`" if lbl else f"`[{multi_start}]`"
                        st.success(f"🏁 **Xuất phát:** {start_disp}")
                        st.info("🏎️ **Bước 2:** Nhấp vào hướng bất kỳ xung quanh điểm xuất phát để **xoay xe**.")
                    else:
                        st.warning("⚠️ Điểm xuất phát không hợp lệ hoặc đã bị xóa. Vui lòng chọn lại (Bước 1).")
                        st.session_state.multi_click_step = 0
                        st.session_state.multi_start = None
                        st.rerun()
                else:
                    if multi_start and multi_start in gm.graph:
                        lbl = gm.graph.nodes[multi_start].get('label', '')
                        start_disp = f"**{lbl}** `[{multi_start}]`" if lbl else f"`[{multi_start}]`"
                        st.success(f"🏁 **Xuất phát:** {start_disp}")
                        st.success(f"🏎️ **Hướng:** `{st.session_state.initial_heading}°`")
                        st.info("🎯 **Bước 3:** Nhấp chọn các **điểm đến** tiếp theo theo thứ tự.")
                    else:
                        st.warning("⚠️ Điểm xuất phát không hợp lệ hoặc đã bị xóa. Vui lòng chọn lại.")
                        st.session_state.multi_click_step = 0
                        st.session_state.multi_start = None
                        st.rerun()

                    if not multi_wps:
                        st.info("👇 **Bước 3:** Nhấp chọn các node trên bản đồ để thêm vào lộ trình. Rồi nhấn 'Tính Lộ Trình'.")
                    else:
                        st.markdown(f"**Điểm ghé thăm đã chọn ({len(multi_wps)} điểm):**")
                        for i, wp in enumerate(multi_wps):
                            wp_label = gm.graph.nodes[wp].get('label', '') or wp if wp in gm.graph else wp
                            r1, r2 = st.columns([5, 1])
                            r1.markdown(f"&nbsp;&nbsp;`{i+1}.` **{wp_label}** `[{wp}]`")
                            if r2.button("✖", key=f"rm_wp_{i}", help="Xóa điểm này"):
                                st.session_state.multi_waypoints.pop(i)
                                # Clear results if any
                                st.session_state.current_path = []
                                st.session_state.tour_visit_order = []
                                st.session_state.segment_breaks = []
                                st.session_state.path_cost = 0.0
                                st.rerun()

            with ms_col2:
                st.markdown("&nbsp;")
                algo = st.session_state.get("algo_select", "Dijkstra")
                n_wps = len(st.session_state.multi_waypoints)

                if n_wps > 12:
                    st.warning(f"⚠️ {n_wps} điểm: dùng thuật toán **Greedy + 2-opt** (gần tối ưu).")
                elif n_wps > 0:
                    st.success(f"✅ {n_wps} điểm: dùng **Held-Karp DP** (chính xác 100%).")

                btn_disabled = (st.session_state.multi_start is None or n_wps == 0)
                if st.button(
                    "🔀 Tính Lộ Trình Tối Ưu",
                    disabled=btn_disabled,
                    width='stretch',
                    type="primary",
                    help="Cần chọn điểm xuất phát và ít nhất 1 điểm ghé thăm"
                ):
                    with st.spinner("Đang tính toán lộ trình theo thứ tự..."):
                        full_path, total_cost, visit_order = compute_sequential_path(
                            gm,
                            st.session_state.multi_start,
                            st.session_state.multi_waypoints,
                            algo,
                            initial_heading=st.session_state.get("initial_heading")
                        )
                    if not full_path:
                        st.error("❌ Không thể tính lộ trình! Kiểm tra xem các node có kết nối không.")
                    else:
                        # Compute segment_breaks
                        full_seq = [st.session_state.multi_start] + visit_order
                        breaks = [0]
                        pos = 0
                        for seg_idx in range(1, len(full_seq)):
                            target = full_seq[seg_idx]
                            for k in range(pos, len(full_path)):
                                if full_path[k] == target:
                                    breaks.append(k)
                                    pos = k
                                    break

                        st.session_state.current_path = full_path
                        st.session_state.path_cost = total_cost
                        st.session_state.tour_visit_order = visit_order
                        st.session_state.segment_breaks = breaks
                        st.session_state.s_node = st.session_state.multi_start
                        st.session_state.e_node = None
                        sim.start_route(full_path)
                        st.rerun()

                if st.button("⏪ Quay lại bước trước", width='stretch', disabled=(step == 0)):
                    st.session_state.multi_click_step -= 1
                    if st.session_state.multi_click_step == 0:
                        st.session_state.multi_start = None
                        st.session_state.s_node = None
                    st.rerun()

                if st.button("🔄 Reset Tất Cả", width='stretch'):
                    st.session_state.multi_start = None
                    st.session_state.multi_waypoints = []
                    st.session_state.tour_visit_order = []
                    st.session_state.segment_breaks = []
                    st.session_state.current_path = []
                    st.session_state.path_cost = 0.0
                    st.session_state.multi_click_step = 0
                    st.session_state.s_node = None
                    st.session_state.e_node = None
                    sim.reset()
                    st.rerun()

    # ──────────────────────────────────────────────
    # RENDER MAP
    # ──────────────────────────────────────────────
    # In multi-stop mode, pass waypoints and tour info to canvas
    is_multi_mode = (click_action == "🚩 Lập Kế Hoạch Lộ Trình (3 Bước)")
    canvas_waypoints = st.session_state.multi_waypoints if is_multi_mode else []
    canvas_visit_order = st.session_state.tour_visit_order if is_multi_mode else []
    canvas_breaks = st.session_state.segment_breaks if is_multi_mode else []

    fig = create_parking_lot_map(
        gm,
        st.session_state.current_path,
        st.session_state.get('s_node') if is_multi_mode else st.session_state.s_node,
        None if is_multi_mode else st.session_state.e_node,
        current_node,
        waypoints=canvas_waypoints,
        visit_order=canvas_visit_order,
        segment_breaks=canvas_breaks,
        initial_heading=st.session_state.get("initial_heading")
    )

    try:
        selection = st.plotly_chart(fig, on_select="rerun", selection_mode="points", width='stretch')
        if selection and hasattr(selection, "selection") and "points" in selection.selection:
            pts = selection.selection["points"]
            if pts and click_action != "Không làm gì":
                node_id = pts[0].get('customdata')
                clk_x, clk_y = pts[0]['x'], pts[0]['y']

                # ── Mode: Waypoint ────────────────────────────────────────
                if click_action == "⚪ Ghim điểm hướng dẫn (Waypoint - Không RFID)":
                    if node_id is not None and str(node_id) in gm.graph:
                        st.error("⚠️ Vị trí này đã có node rồi!")
                    else:
                        # Suggest a split if clicked on edge
                        edge, proj = gm.get_closest_edge(clk_x, clk_y)
                        if edge:
                            w_id = gm._new_waypoint_id()
                            # Manually split edge but with is_rfid=False
                            original_weight = gm.graph[edge[0]][edge[1]].get("weight", 1.0)
                            x0, y0 = gm.graph.nodes[edge[0]]["x"], gm.graph.nodes[edge[0]]["y"]
                            x1, y1 = gm.graph.nodes[edge[1]]["x"], gm.graph.nodes[edge[1]]["y"]
                            total_len = math.hypot(x1 - x0, y1 - y0) or 1.0
                            w1 = original_weight * math.hypot(proj[0] - x0, proj[1] - y0) / total_len
                            w2 = original_weight * math.hypot(x1 - proj[0], y1 - proj[1]) / total_len

                            gm.graph.remove_edge(edge[0], edge[1])
                            gm.add_node(proj[0], proj[1], w_id, label="", is_rfid=False)
                            gm.add_edge(edge[0], w_id, weight=w1)
                            gm.add_edge(w_id, edge[1], weight=w2)
                            st.success(f"✅ Đã chia Edge và chèn Waypoint **{w_id}**")
                        else:
                            # Create standalone waypoint
                            w_id = gm._new_waypoint_id()
                            gm.add_node(clk_x, clk_y, w_id, label="", is_rfid=False)
                            st.success(f"✅ Đã thêm Waypoint **{w_id}** tại ({clk_x:.1f}, {clk_y:.1f})")
                        st.rerun()

                # ── Mode: Lập Kế Hoạch 3 Bước ──────────────────────────────
                if click_action == "🚩 Lập Kế Hoạch Lộ Trình (3 Bước)":
                    step = st.session_state.multi_click_step
                    
                    if step == 0:
                        # Bước 1: Chọn điểm xuất phát
                        if not node_id or str(node_id) not in gm.graph:
                            edge, proj = gm.get_closest_edge(clk_x, clk_y)
                            if edge:
                                node_id = gm.create_virtual_node_on_edge(proj[0], proj[1], edge[0], edge[1])
                        
                        if node_id and str(node_id) in gm.graph:
                            st.session_state.multi_start = node_id
                            st.session_state.s_node = node_id
                            st.session_state.multi_click_step = 1
                            st.rerun()
                    
                    elif step == 1:
                        # Bước 2: Chọn hướng xe
                        s_coords = gm.graph.nodes[st.session_state.s_node]
                        sx, sy = s_coords['x'], s_coords['y']
                        dx = clk_x - sx
                        dy = clk_y - sy
                        
                        if abs(dx) > abs(dy):
                            heading = 0 if dx > 0 else 180
                        else:
                            heading = 90 if dy > 0 else 270
                        
                        st.session_state.initial_heading = heading
                        st.session_state.multi_click_step = 2
                        
                        # Clear any existing path as start/heading has changed
                        st.session_state.current_path = []
                        st.session_state.tour_visit_order = []
                        st.session_state.segment_breaks = []
                        st.rerun()
                    
                    elif step == 2:
                        # Bước 3: Chọn các điểm đến
                        if not node_id or str(node_id) not in gm.graph:
                            edge, proj = gm.get_closest_edge(clk_x, clk_y)
                            if edge:
                                node_id = gm.create_virtual_node_on_edge(proj[0], proj[1], edge[0], edge[1])

                        if node_id and str(node_id) in gm.graph:
                            node_id = str(node_id)
                            if node_id == st.session_state.multi_start:
                                st.toast("⚠️ Điểm xuất phát đã được chọn.", icon="⚠️")
                            elif node_id in st.session_state.multi_waypoints:
                                st.toast(f"ℹ️ Node `{node_id}` đã có trong danh sách.", icon="ℹ️")
                            else:
                                st.session_state.multi_waypoints.append(node_id)
                                st.session_state.current_path = [] 
                                st.rerun()

                # ── Mode: RFID Pin ────────────────────────────────────────
                elif click_action == "📍 Ghim thẻ RFID mới (Dùng mã đã khai báo ở trên)":
                    pending = st.session_state.rfid_pending
                    if node_id is not None and str(node_id) in gm.graph:
                        st.error("⚠️ Trùng vị trí! Hãy nhấp ra chỗ trống trên bản đồ.")
                    elif pending is None:
                        st.warning("💡 Chưa khóa thông tin thẻ. Hãy điền ID + Tên rồi nhấn **Khóa** trước.")
                    else:
                        id_str = pending["id"]
                        lbl_str = pending["label"]
                        if gm.is_uid_used(id_str):
                            st.error(f"❌ UID '{id_str}' đã được sử dụng!")
                        else:
                            gm.add_node(clk_x, clk_y, id_str, lbl_str)
                            st.success(f"✅ Đã ghim thẻ **{id_str}** ({lbl_str}) tại ({clk_x:.1f}, {clk_y:.1f})!")
                            st.session_state.rfid_pending = None
                            st.session_state.last_scanned_uid = ""
                            st.rerun()

    except TypeError:
        st.plotly_chart(fig, width='stretch')

    # ──────────────────────────────────────────────
    # SECTION 2: Node Management Table
    # ──────────────────────────────────────────────
    st.write("---")
    all_nodes = [(n, d) for n, d in gm.graph.nodes(data=True) if not str(n).startswith("V_")]
    with st.expander(f"🗂️ Quản lý Node RFID ({len(all_nodes)} thẻ)", expanded=False):
        if not all_nodes:
            st.info("Chưa có thẻ RFID nào được đăng ký. Hãy ghim thẻ đầu tiên lên bản đồ.")
        else:
            editing = st.session_state.editing_node

            for node_id, ndata in all_nodes:
                is_rfid = ndata.get('is_rfid', True)
                
                if editing == node_id:
                    with st.container(border=True):
                        st.markdown(f"**✏️ Đang sửa node:** `{node_id}`")
                        ec1, ec2, ec3, ec4 = st.columns([3, 1.5, 1.5, 1])
                        new_label = ec1.text_input("Tên mới", value=ndata.get('label', ''), key=f"edit_lbl_{node_id}")
                        new_x = ec2.number_input("X", value=float(ndata['x']), key=f"edit_x_{node_id}", step=1.0)
                        new_y = ec3.number_input("Y", value=float(ndata['y']), key=f"edit_y_{node_id}", step=1.0)
                        ec4.write("")
                        ec4.write("")
                        sv, cn = st.columns(2)
                        if sv.button("💾 Lưu", key=f"save_{node_id}", width='stretch', type="primary"):
                            gm.edit_node(node_id, new_x, new_y, new_label)
                            st.session_state.editing_node = None
                            st.rerun()
                        if cn.button("✖️ Hủy", key=f"cancel_{node_id}", width='stretch'):
                            st.session_state.editing_node = None
                            st.rerun()
                else:
                    with st.container(border=True):
                        c_info, c_uids, c_actions = st.columns([3.5, 4, 1.5])
                        
                        with c_info:
                            st.markdown(f"**🆔 Node ID:** `{node_id}`")
                            st.markdown(f"**🏷️ Tên vị trí:** {ndata.get('label', '—')}")
                            lbl_type = "🎟️ Thẻ RFID" if is_rfid else "⚪ Waypoint (Chỉ dẫn)"
                            st.markdown(f"**🏢 Loại:** {lbl_type}")
                            st.markdown(f"**📍 Tọa độ:** X: `{ndata['x']:.1f}` | Y: `{ndata['y']:.1f}`")
                            
                        with c_uids:
                            if is_rfid:
                                uids = ndata.get('uids', [node_id])
                                st.markdown("**🔗 Danh sách UIDs (Thẻ) liên kết:**")
                                
                                uid_cols = st.columns(3)
                                for i, u in enumerate(uids):
                                    with uid_cols[i % 3]:
                                        if u == node_id:
                                            st.caption(f"Gốc: `{u}`")
                                        else:
                                            if st.button(f"✖ `{u}`", key=f"del_uid_{node_id}_{u}", help=f"Xóa UID khỏi node", type="secondary"):
                                                if gm.remove_secondary_uid(node_id, u):
                                                    st.rerun()
                                
                                with st.expander("➕ Quét hoặc Thêm UIDs thủ công", expanded=False):
                                    add_uid_key = f"add_uid_input_{node_id}"
                                    if add_uid_key not in st.session_state:
                                        st.session_state[add_uid_key] = ""
                                        
                                    col_input, col_scan = st.columns([2, 1])
                                    
                                    def make_handler(n_id, n_key):
                                        def _handler():
                                            val = st.session_state[n_key].strip()
                                            if val:
                                                if gm.add_secondary_uid(n_id, val):
                                                    st.toast(f"✅ Đã thêm {val} cho node {n_id}")
                                                    st.session_state[n_key] = ""
                                                else:
                                                    st.error(f"❌ UID {val} đã tồn tại!")
                                        return _handler
                                        
                                    col_input.text_input(
                                        "Nhập mã HEX UID:", 
                                        key=add_uid_key, 
                                        placeholder="VD: A0-B1-...",
                                        on_change=make_handler(node_id, add_uid_key),
                                        label_visibility="collapsed"
                                    )
                                    
                                    if HAS_RFID:
                                        if col_scan.button("🔍 Quét Bằng Máy", key=f"scan_btn_{node_id}", width='stretch'):
                                            with st.spinner(f"Chờ thẻ {node_id}..."):
                                                reader = get_rfid_reader()
                                                try:
                                                    scanned = reader.read_uid_hex(timeout=3.0)
                                                    if scanned:
                                                        if gm.add_secondary_uid(node_id, scanned):
                                                            st.toast(f"✅ Đã gán {scanned} cho {node_id}")
                                                            st.rerun()
                                                        else:
                                                            st.error(f"❌ UID {scanned} đã tồn tại!")
                                                    else:
                                                        st.warning("⚠️ Timeout!")
                                                except Exception as e:
                                                    st.error(f"Lỗi: {e}")
                                    else:
                                        col_scan.button("🔍 Quét Bằng Máy", disabled=True, key=f"scan_btn_off_{node_id}", width='stretch')
                            else:
                                st.info("ℹ️ Node dạng Waypoint không có Thẻ RFID vật lý liên kết.")

                        with c_actions:
                            if st.button("✏️ Sửa", key=f"edit_btn_{node_id}", width='stretch'):
                                st.session_state.editing_node = node_id
                                st.rerun()
                            if st.button("🗑️ Xóa", key=f"del_btn_{node_id}", width='stretch', type="primary"):
                                gm.delete_node(node_id)
                                if st.session_state.get('s_node') == node_id:
                                    st.session_state.s_node = None
                                if st.session_state.get('e_node') == node_id:
                                    st.session_state.e_node = None
                                if node_id in st.session_state.get("multi_waypoints", []):
                                    st.session_state.multi_waypoints.remove(node_id)
                                if node_id == st.session_state.get("multi_start"):
                                    st.session_state.multi_start = None
                                st.session_state.current_path = []
                                st.rerun()

    # ──────────────────────────────────────────────
    # SECTION 3: Turn Instructions & Path Display
    # ──────────────────────────────────────────────
    def compute_turn_instructions(path, gm, initial_heading=None):
        if len(path) < 2: return "Đã ở đích"
        instructions = []
        
        # New: Initial Turn from starting orientation
        if initial_heading is not None:
            curr = path[0]
            nxt = path[1]
            if curr in gm.graph.nodes and nxt in gm.graph.nodes:
                dx = gm.graph.nodes[nxt]['x'] - gm.graph.nodes[curr]['x']
                dy = gm.graph.nodes[nxt]['y'] - gm.graph.nodes[curr]['y']
                target_ang = math.degrees(math.atan2(dy, dx))
                
                # Normalize target_ang to [0, 360) to match initial_heading logic
                target_ang_norm = target_ang % 360
                
                diff = (target_ang_norm - initial_heading) % 360
                if diff > 180: diff -= 360
                
                # Debug logging (visible in terminal)
                print(f"[DEBUG] Orientation: curr={curr}, nxt={nxt}, dx={dx}, dy={dy}")
                print(f"[DEBUG] target_ang={target_ang:.2f}, initial={initial_heading}, diff={diff:.2f}")

                if -25 <= diff <= 25: turn = "Hướng ban đầu: Đã đúng hướng ⬆️"
                elif 25 < diff <= 135: turn = "Hướng ban đầu: Cần rẽ TRÁI ⬅️"
                elif -135 <= diff < -25: turn = "Hướng ban đầu: Cần rẽ PHẢI ➡️"
                else: turn = "Hướng ban đầu: Ngược hướng, cần QUAY ĐẦU ⬇️"
                instructions.append(f"• **Bắt đầu**: {turn}")

        for i in range(1, len(path)-1):
            prev = path[i-1]
            curr = path[i]
            nxt = path[i+1]

            if curr not in gm.graph.nodes or prev not in gm.graph.nodes or nxt not in gm.graph.nodes:
                continue

            dx1 = gm.graph.nodes[curr]['x'] - gm.graph.nodes[prev]['x']
            dy1 = gm.graph.nodes[curr]['y'] - gm.graph.nodes[prev]['y']
            a1 = math.degrees(math.atan2(dy1, dx1))

            dx2 = gm.graph.nodes[nxt]['x'] - gm.graph.nodes[curr]['x']
            dy2 = gm.graph.nodes[nxt]['y'] - gm.graph.nodes[curr]['y']
            a2 = math.degrees(math.atan2(dy2, dx2))

            diff = (a2 - a1) % 360
            if diff > 180: diff -= 360

            if -25 <= diff <= 25: turn = "Tiến thẳng ⬆️"
            elif 25 < diff <= 135: turn = "Rẽ TRÁI ⬅️"
            elif -135 <= diff < -25: turn = "Rẽ PHẢI ➡️"
            else: turn = "Quay đầu ⬇️"

            if i == len(path) - 2 and str(nxt).startswith("V_"):
                dist_px = math.hypot(dx2, dy2)
                time_s = dist_px / gm.speed_px_per_sec
                turn += f" (rồi chạy Mù thêm {time_s:.2f}s để tới Đích 🎯)"

            instructions.append(f"• Qua chốt **{curr}**: {turn}")

        if not instructions: return "Cứ đi thẳng không có khúc cua."
        return "\n".join(instructions)

    if st.session_state.current_path:
        is_multi = bool(st.session_state.tour_visit_order)

        if is_multi:
            st.success("🔀 Lộ Trình Tối Ưu Multi-Stop:")
            visit_order = st.session_state.tour_visit_order
            multi_start = st.session_state.multi_start

            # Show tour sequence with labels
            def node_display(nid):
                if nid not in gm.graph:
                    return nid
                lbl = gm.graph.nodes[nid].get('label', '') or nid
                return f"{lbl} [{nid}]"

            tour_seq = [multi_start] + visit_order
            tour_str = " → ".join([f"**{node_display(n)}**" for n in tour_seq])
            st.markdown(f"**Thứ tự:** {tour_str}")
            st.metric("Tổng chi phí (round trip)", f"{st.session_state.path_cost:.2f}")

            # Per-leg breakdown in expander
            with st.expander("📋 Chi tiết từng chặng"):
                for i in range(len(tour_seq) - 1):
                    u, v = tour_seq[i], tour_seq[i + 1]
                    breaks = st.session_state.segment_breaks
                    if i < len(breaks) - 1:
                        seg = st.session_state.current_path[breaks[i]:breaks[i+1]+1]
                        seg_cost = 0.0
                        for k in range(len(seg)-1):
                            a, b = seg[k], seg[k+1]
                            if gm.graph.has_edge(a, b):
                                seg_cost += gm.graph[a][b].get('weight', 1.0)
                        st.markdown(f"**Chặng {i+1}:** `{node_display(u)}` → `{node_display(v)}` — Cost: `{seg_cost:.2f}` — Nodes: `{' → '.join(seg)}`")
        else:
            st.success("Lộ trình dạng Chuỗi Điểm (Sequence):")
            st.code(" → ".join(st.session_state.current_path))

        st.info("Hệ lệnh Rẽ cho Robot (Turns):\n" + compute_turn_instructions(st.session_state.current_path, gm, st.session_state.get("initial_heading")))

        if HAS_HW:
            st.write("---")
            st.subheader("🤖 Điều Khiển Robot Thực Tế")
            run_col, stop_col = st.columns(2)

            if run_col.button("🚀 BẮT ĐẦU CHẠY XE (RUN MISSION)"):
                target = st.session_state.current_path[-1] if st.session_state.get('current_path') else None
                if not target:
                    st.error("⚠️ XE KHÔNG BIẾT ĐI ĐÂU! Vui lòng lên lộ trình trước khi chạy xe!")
                    st.stop()

                st.session_state.car_logs = []
                graph_adj = {
                    'nodes': {n: {'x': gm.graph.nodes[n]['x'], 'y': gm.graph.nodes[n]['y']} for n in gm.graph.nodes()},
                    'edges': {}
                }
                for u in gm.graph.nodes():
                    graph_adj['edges'][u] = {}
                    for v in gm.graph.neighbors(u):
                        graph_adj['edges'][u][v] = gm.graph[u][v].get('weight', 1)

                rfid_map = {}
                for n, d in gm.graph.nodes(data=True):
                    node_uids = d.get('uids', [n])
                    for u in node_uids:
                        rfid_map[u] = n

                # Safety: Stop any existing car instance before starting a new one
                if 'car_instance' in st.session_state:
                    try:
                        st.session_state.car_instance.stop_system()
                        time.sleep(0.2) # Give it a moment to stop
                    except:
                        pass

                from autonomous_main import AutonomousCar
                car = AutonomousCar(
                    target_node=target,
                    graph=graph_adj,
                    turn_table={},
                    rfid_map=rfid_map,
                    turn_config=st.session_state.turn_config,
                    predefined_path=st.session_state.current_path,
                    initial_heading=st.session_state.get("initial_heading", 0),
                    speed_px_per_sec=float(gm.speed_px_per_sec),
                    lane_mode=st.session_state.lane_mode,
                    perspective_src=st.session_state.perspective_src
                )
                st.session_state.car_instance = car
                st.session_state.car_thread = threading.Thread(target=car.execute, daemon=True)
                st.session_state.car_thread.start()
                st.success("Đã gửi API điều hướng xuống vi điều khiển xe!")

            if stop_col.button("🛑 DỪNG XE KHẨN CẤP (STOP)"):
                if 'car_instance' in st.session_state:
                    st.session_state.car_instance.stop_system()
                    st.success("Đã kích hoạt phanh khẩn cấp!")
            
            render_live_logs()

if active_tab == "tab3":
    st.header("🎮 Điều Khiển Thủ Công (WASD)")
    # Live Camera with Media Controls inside this tab
    render_live_camera_and_status(show_media_controls=True, key_suffix="wasd")
    # Optimized Manual Controls in a separate fragment to prevent camera flickering
    render_wasd_manual_controls()


if active_tab == "tab4":
    st.header("🛑 Traffic Sign Realtime Debugger")
    st.info("Tab này kéo trực tiếp frame từ luồng camera để phân tích độc lập giúp bạn dễ dàng hiệu chỉnh thông số màu sắc của các biển báo.")
    if HAS_HW:
        from core.detector import SignDetector
        from core.classifier import SignClassifier
        
        if "debug_sign_detector" not in st.session_state:
            st.session_state.debug_sign_detector = SignDetector()
            st.session_state.debug_sign_classifier = SignClassifier(templates_dir='templates')
            
        sign_det = st.session_state.debug_sign_detector
        sign_cls = st.session_state.debug_sign_classifier
        
        @st.fragment(run_every=0.5)
        def render_traffic_sign_debug():
            frame = get_camera_manager().get_frame() if HAS_HW else None
            if frame is not None:
                # frame is BGR natively from manager
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detected_signs, mask = sign_det.get_detection_with_mask(rgb_frame)
                
                annotated = frame.copy()
                detection_logs = []
                
                if detected_signs:
                    for idx, (roi, bbox) in enumerate(detected_signs):
                        sign_type, conf = sign_cls.classify(roi)
                        if conf >= 0.1:
                            x, y, w, h = bbox
                            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(annotated, f"{sign_type.upper()}: {conf:.2f}", (x, max(15, y - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            log_msg = f"✅ Phát hiện biển **{sign_type.upper()}** (Độ tin cậy: {conf*100:.1f}%) tại toạ độ {bbox}"
                            detection_logs.append(log_msg)
                            
                            # Thuận theo request "Luôn ghi ra log"
                            print(f"[TRAFFIC-SIGN-DEBUG] Detected {sign_type.upper()} ({conf*100:.1f}%)")
                            
                st.markdown("### Luồng phân tích (Realtime)")
                col1, col2 = st.columns(2)
                
                # Pre-encode frames to bytes so Streamlit does not lose them in garbage collection
                _, buffer_annotated = cv2.imencode('.jpg', annotated)
                _, buffer_mask = cv2.imencode('.jpg', mask)
                
                with col1:
                    st.image(buffer_annotated.tobytes(), caption="Camera + Detections", width='stretch')
                with col2:
                    st.image(buffer_mask.tobytes(), caption="HSV Blue Mask", width='stretch', clamp=True)
                    
                st.markdown("### Nhật ký Mới nhất:")
                if detection_logs:
                    for log in detection_logs:
                        st.success(log)
                else:
                    st.caption("Chưa phát hiện biển báo nào. Vui lòng đưa biển báo vào luồng camera...")
            else:
                st.warning("Camera chưa sẵn sàng.")
                
        render_traffic_sign_debug()
    else:
        st.error("Tính năng này yêu cầu Hardware Camera hoạt động.")

# ── End of App ────────────────────────────────────────────────────────────────
