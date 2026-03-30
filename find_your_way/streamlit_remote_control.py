import os
import time
import threading
from pathlib import Path

import streamlit as st

from hardware.motor import move_straight, turn_left, turn_right, stop
from hardware.rfid import RFIDReader

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
    from torchvision import transforms
except ImportError:
    torch = None


MODEL_PATHS = [
    Path("models/detect/best_unet_v1.pth"),
    Path("models/detect/best_unet_v2.pth"),
    Path("models/best_yolo.pt"),
    Path("models/best_trans.pt"),
]


def load_model():
    if torch is None:
        return None

    for p in MODEL_PATHS:
        if p.exists():
            try:
                model = torch.load(str(p), map_location="cpu")
                model.eval()
                return model
            except Exception:
                continue
    return None


def model_infer(model, frame):
    """Fake inference step for display; replace with specific model pipeline."""
    if model is None or frame is None:
        return "No model loaded"

    try:
        # Basic placeholder: convert to grayscale mean and show as "confidence"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m = float(gray.mean())
        return f"Frame brightness mean={m:.2f} (model inference placeholder)"
    except Exception as e:
        return f"Model infer error: {e}"


def _ensure_state():
    if "remote_state" not in st.session_state:
        st.session_state.remote_state = {
            "camera_running": False,
            "rfid_running": False,
            "camera_frame": None,
            "camera_log": [],
            "rfid_log": [],
            "model_log": [],
            "reader": None,
            "cam_cap": None,
            "model": None,
            "lock": threading.Lock(),
        }


def _append_deque(log_list, value, max_len=80):
    log_list.insert(0, value)
    if len(log_list) > max_len:
        log_list.pop()


def rfid_loop():
    state = st.session_state.remote_state
    reader = RFIDReader()
    state["reader"] = reader

    while state["rfid_running"]:
        uid = reader.read_uid_hex(timeout=0.5)
        if uid:
            entry = f"{time.strftime('%H:%M:%S')} - {uid}"
            with state["lock"]:
                _append_deque(state["rfid_log"], entry, max_len=100)
        time.sleep(0.1)


def camera_loop():
    state = st.session_state.remote_state
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with state["lock"]:
            _append_deque(state["camera_log"], f"[{time.strftime('%H:%M:%S')}] Cannot open camera device")
        state["camera_running"] = False
        return

    state["cam_cap"] = cap
    model = state["model"]

    while state["camera_running"]:
        ret, frame = cap.read()
        if not ret:
            with state["lock"]:
                _append_deque(state["camera_log"], f"[{time.strftime('%H:%M:%S')}] Camera frame read failed")
            time.sleep(0.2)
            continue

        with state["lock"]:
            state["camera_frame"] = frame

        if model is not None:
            info = model_infer(model, frame)
            with state["lock"]:
                _append_deque(state["model_log"], f"[{time.strftime('%H:%M:%S')}] {info}", max_len=50)

        time.sleep(0.03)

    if cap.isOpened():
        cap.release()


def motor_action(action):
    try:
        if action == "forward":
            move_straight()
        elif action == "left":
            turn_left()
        elif action == "right":
            turn_right()
        elif action == "stop":
            stop()
        elif action == "backward":
            # Thực tế motor không có backward; thực hiện turn 180? Tạm stop.
            stop()
        else:
            stop()

        _append_deque(st.session_state.remote_state["camera_log"], f"[{time.strftime('%H:%M:%S')}] Motor command: {action}")
    except Exception as e:
        _append_deque(st.session_state.remote_state["camera_log"], f"[{time.strftime('%H:%M:%S')}] Motor error: {e}")


def main():
    st.set_page_config(page_title="Vehicle Remote Control", layout="wide")
    _ensure_state()

    state = st.session_state.remote_state

    if state["model"] is None:
        state["model"] = load_model()

    st.title("Vehicle Remote + Camera + RFID Control")

    status_col, camera_col, logs_col = st.columns([1, 2, 1])

    with status_col:
        st.subheader("Keystones")

        if c := st.button("Forward", key="btn_forward"):
            motor_action("forward")
        if c := st.button("Left", key="btn_left"):
            motor_action("left")
        if c := st.button("Right", key="btn_right"):
            motor_action("right")
        if c := st.button("Stop", key="btn_stop"):
            motor_action("stop")
        if c := st.button("Backward (stop proxy)", key="btn_back"):
            motor_action("backward")

        st.markdown("---")

        # Camera controls
        if cv2 is None:
            st.error("OpenCV not installed (pip install opencv-python). Camera disabled")
        else:
            if not state["camera_running"]:
                if st.button("Start Camera", key="start_cam"):
                    state["camera_running"] = True
                    threading.Thread(target=camera_loop, daemon=True).start()
            else:
                if st.button("Stop Camera", key="stop_cam"):
                    state["camera_running"] = False

        # RFID controls
        if not state["rfid_running"]:
            if st.button("Start RFID scanning", key="start_rfid"):
                state["rfid_running"] = True
                threading.Thread(target=rfid_loop, daemon=True).start()
        else:
            if st.button("Stop RFID scanning", key="stop_rfid"):
                state["rfid_running"] = False

        st.markdown("---")
        st.write("Model status:")
        if state["model"] is not None:
            st.success("Model loaded")
        else:
            st.info("No model loaded, inference disabled")

    with camera_col:
        st.subheader("Video stream")
        camera_placeholder = st.empty()

        if state["camera_frame"] is not None:
            with state["lock"]:
                frame = state["camera_frame"].copy()
            camera_placeholder.image(frame, channels="BGR", caption="Live camera", use_column_width=True)
        else:
            camera_placeholder.info("No camera frame yet. Start camera to preview.")

        st.markdown("---")
        st.subheader("Camera logs")
        with st.expander("Camera/Action logs", expanded=True):
            for r in state["camera_log"][:80]:
                st.write(r)

    with logs_col:
        st.subheader("RFID log")
        with st.expander("RFID read history", expanded=True):
            for r in state["rfid_log"][:80]:
                st.write(r)

        st.markdown("---")
        st.subheader("Model log")
        with st.expander("Model inference history", expanded=True):
            for r in state["model_log"][:80]:
                st.write(r)

    # Refresh to keep stream updated. 1s interval.
    try:
        from streamlit import st_autorefresh
        st_autorefresh(interval=1000, key="remote_control_auto_refresh")
    except Exception:
        st.info("Auto-refresh not available in this streamlit version")


if __name__ == "__main__":
    main()
