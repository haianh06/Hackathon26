import streamlit as st
import os
import base64
import json
import sys
import time
import math
import cv2
import numpy as np
import threading

@st.cache_resource
def get_camera_manager():
    """Persistent camera manager instance."""
    from hardware.camera import camera_manager
    camera_manager.start()
    return camera_manager

def _reinit_camera():
    """Logic to stop, reset, and restart the camera manager."""
    try:
        cm = get_camera_manager()
        cm.stop()
        from hardware.camera import CameraManager
        CameraManager._instance = None
        get_camera_manager.clear()
        _ = get_camera_manager()
        print("✅ Camera re-initialized successfully.")
        return True
    except Exception as e:
        print(f"❌ Error re-initializing camera: {e}")
        return False

try:
    from hardware.rfid import RFIDReader
    from autonomous_main import AutonomousCar
    import hardware.motor as motor
    HAS_HW = True
    HAS_RFID = True
    _ = get_camera_manager()
except Exception as e:
    print(f"Hardware init warning: {e}")
    HAS_HW = False
    HAS_RFID = False

@st.cache_resource
def get_rfid_reader():
    if HAS_RFID: return RFIDReader()
    return None

from graph.graph_manager import GraphManager
from graph.pathfinding import compute_shortest_path, compute_multi_stop_path, compute_sequential_path
from utils.rfid_simulator import RFIDSimulator
from ui.canvas import create_parking_lot_map
from ui.controls import render_sidebar_controls

st.set_page_config(layout="wide", page_title="Self-Driving Navigation System", page_icon="🚗")

# --- Session State ---
if "graph_manager" not in st.session_state:
    gm = GraphManager()
    if os.path.exists("data/graph.json"): gm.load_from_json("data/graph.json")
    st.session_state.graph_manager = gm

if "simulator" not in st.session_state: st.session_state.simulator = RFIDSimulator()
if "current_path" not in st.session_state: st.session_state.current_path = []
if "path_cost" not in st.session_state: st.session_state.path_cost = 0.0
if "s_node" not in st.session_state: st.session_state.s_node = None
if "e_node" not in st.session_state: st.session_state.e_node = None
if "initial_heading" not in st.session_state: st.session_state.initial_heading = 0
if "rfid_pending" not in st.session_state: st.session_state.rfid_pending = None
if "multi_waypoints" not in st.session_state: st.session_state.multi_waypoints = []
if "tour_visit_order" not in st.session_state: st.session_state.tour_visit_order = []
if "map_key" not in st.session_state: st.session_state.map_key = 0
if "mission_was_running" not in st.session_state: st.session_state.mission_was_running = False
if "settings_click" not in st.session_state: st.session_state.settings_click = None
if "heading_set" not in st.session_state: st.session_state.heading_set = False
if "prev_start_node" not in st.session_state: st.session_state.prev_start_node = None

TURN_CONFIG_PATH = "data/turn_config.json"
def _load_turn_config() -> dict:
    defaults = {"90_DEG_RIGHT": 1.3, "90_DEG_LEFT": 2.0, "180_DEG": 3.6, "STRAIGHT": 2.0}
    try:
        if os.path.exists(TURN_CONFIG_PATH):
            with open(TURN_CONFIG_PATH) as f: return {**defaults, **json.load(f)}
        return defaults
    except: return defaults

def _save_turn_config(cfg: dict):
    os.makedirs(os.path.dirname(TURN_CONFIG_PATH), exist_ok=True)
    with open(TURN_CONFIG_PATH, "w") as f: json.dump(cfg, f, indent=4)

if "turn_config" not in st.session_state: st.session_state.turn_config = _load_turn_config()

# --- Fragments ---
@st.fragment(run_every=1.0)
def render_live_camera_and_status(show_settings=False, key_suffix="main"):
    if HAS_HW:
        with st.container(border=True):
            c1, c2 = st.columns([2, 1])
            with c1:
                frame = None
                if st.session_state.get("car_instance"): frame = getattr(st.session_state.car_instance, 'debug_frame', None)
                if frame is None: frame = get_camera_manager().get_frame_resized()
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    st.image(buffer.tobytes(), use_container_width=True, caption="Live Camera")
                else: st.info("Waiting for camera...")
            with c2:
                st.markdown("### 🚦 Status")
                is_auton = st.session_state.get("car_instance") and st.session_state.car_instance.is_active
                st.metric("Mode", "Autonomous" if is_auton else "Standby")
                if st.button("🔄 Refresh Stream", key=f"ref_{key_suffix}"): _reinit_camera(); st.rerun()
                
                # --- Waiting for User Confirmation ---
                if st.session_state.get("car_instance") and getattr(st.session_state.car_instance, 'state', None) and st.session_state.car_instance.state.name == "WAITING":
                    st.divider()
                    # Custom CSS for a big red button
                    st.markdown("""
                        <style>
                        .stButton>button[kind="primary"] {
                            background-color: #ff1744 !important;
                            color: white !important;
                            border: none !important;
                            height: 4em !important;
                            font-size: 24px !important;
                            font-weight: bold !important;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    if st.button("🚀 TIẾP TỤC HÀNH TRÌNH", type="primary", use_container_width=True):
                        st.session_state.car_instance.resume_mission()
                        st.rerun()
                    st.info("💡 Nhấn nút đỏ ở trên để xe tiếp tục di chuyển.")

@st.fragment(run_every=1.5)
def render_live_logs():
    st.markdown("### 📝 Live Logs")
    if 'car_instance' in st.session_state:
        logs = st.session_state.car_instance.log_history
        st.code("\n".join(logs[-10:]) if logs else "No logs yet...")
    else: st.info("Logs appear here during mission.")

@st.fragment(run_every=2.0)
def render_sign_gallery():
    if 'car_instance' in st.session_state:
        snapshots = getattr(st.session_state.car_instance, 'sign_snapshots', [])
        if snapshots:
            st.markdown("### 📸 Signs")
            cols = st.columns(len(snapshots[-5:]))
            for i, img in enumerate(reversed(snapshots[-5:])):
                cols[i].image(img, width='stretch')

@st.fragment
def render_wasd_manual_controls():
    st.info("Use WASD buttons below (HW required)")
    c1, c2, c3 = st.columns(3)
    if c2.button("🔼 W", width='stretch'): motor.move_straight()
    c1, c2, c3 = st.columns(3)
    if c1.button("◀️ A", width='stretch'): motor.turn_left()
    if c2.button("⏹️ S", width='stretch'): motor.stop()
    if c3.button("▶️ D", width='stretch'): motor.turn_right()

# --- Main App ---
gm = st.session_state.graph_manager
sim = st.session_state.simulator

st.title("🚗 Autonomous Parking Mission")
tab_options = {"🌐 Mission Dashboard": "tab1", "🔧 System Settings": "tab2", "🎮 Manual Control": "tab3", "📝 Mission History": "tab4"}
active_tab = tab_options[st.radio("Navigation", list(tab_options.keys()), horizontal=True, label_visibility="collapsed")]
st.markdown("---")

if active_tab == "tab1":
    render_live_camera_and_status()
    st.subheader("🚀 Mission Planning (START → P1 → P2 → START)")
    with st.container(border=True):
        real_nodes = [n for n in gm.graph.nodes() if not str(n).startswith("V_")]
        def node_fmt(nid):
            lbl = gm.graph.nodes[nid].get('label')
            return lbl if lbl else nid
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        start_node = m_col1.selectbox("🏠 START", real_nodes, format_func=node_fmt)
        p1_node = m_col2.selectbox("📍 Point 1", real_nodes, index=min(1, len(real_nodes)-1), format_func=node_fmt)
        p2_node = m_col3.selectbox("📍 Point 2", real_nodes, index=min(2, len(real_nodes)-1), format_func=node_fmt)
        
        # New Direction Selector (NWES)
        dir_opts = {"Bắc (North)": 90, "Đông (East)": 0, "Nam (South)": 270, "Tây (West)": 180}
        curr_dir_label = next((k for k, v in dir_opts.items() if v == st.session_state.initial_heading), "Đông (East)")
        selected_dir = m_col4.selectbox("🧭 Hướng xe", list(dir_opts.keys()), index=list(dir_opts.keys()).index(curr_dir_label))
        st.session_state.initial_heading = dir_opts[selected_dir]
        st.session_state.heading_set = True

        # Reset heading if start node changes (optional, but keep it if you want to force re-selection, 
        # though with a selectbox it's always set)
        if st.session_state.prev_start_node != start_node:
            st.session_state.prev_start_node = start_node
        
        # Buttons Row
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🗺️ Generate Route", type="primary", width='stretch', key="btn_gen_route"):
                path, cost, _ = compute_sequential_path(gm, start_node, [p1_node, p2_node, start_node], initial_heading=st.session_state.initial_heading)
                if path:
                    st.session_state.current_path = path
                    st.session_state.multi_waypoints = [p1_node, p2_node, start_node]
                    st.session_state.tour_visit_order = [p1_node, p2_node, start_node]
                    st.session_state.s_node = start_node
                    st.toast(f"✅ Route generated (Heading: {selected_dir})")
        
        with c2:
            if st.button("🚀 START MISSION", width='stretch', key="btn_start_mission"):
                if not st.session_state.current_path: st.error("Generate route first!")
                else:
                    graph_adj = {'nodes': {n: {
                        'x': gm.graph.nodes[n]['x'], 
                        'y': gm.graph.nodes[n]['y'],
                        'label': gm.graph.nodes[n].get('label', ''),
                        'is_rfid': gm.graph.nodes[n].get('is_rfid', True)
                    } for n in gm.graph.nodes()}, 'edges': {}}
                    for u in gm.graph.nodes(): graph_adj['edges'][u] = {v: gm.graph[u][v].get('weight', 1) for v in gm.graph.neighbors(u)}
                    rfid_map = {u: n for n, d in gm.graph.nodes(data=True) for u in d.get('uids', [n])}
                    if 'car_instance' in st.session_state: st.session_state.car_instance.stop_system()
                    car = AutonomousCar(target_node=st.session_state.current_path[-1], graph=graph_adj, turn_table={}, rfid_map=rfid_map,
                                        turn_config=st.session_state.turn_config, predefined_path=st.session_state.current_path,
                                        initial_heading=st.session_state.initial_heading,
                                        speed_px_per_sec=float(gm.speed_px_per_sec),
                                        pause_nodes=[p1_node, p2_node],
                                        sonar_threshold=st.session_state.get("sonar_threshold", 25))
                    st.session_state.car_instance = car
                    st.session_state.car_thread = threading.Thread(target=car.execute, daemon=True)
                    st.session_state.car_thread.start()
                    st.toast("🚀 Vehicle Dispatched!")
                    
        with c3:
            if st.button("🛑 EMERGENCY STOP", width='stretch', key="btn_emerg_stop"):
                if 'car_instance' in st.session_state: st.session_state.car_instance.stop_system(); st.error("🛑 STOP COMMAND SENT")

    # --- Sync Path from Car Instance ---
    if st.session_state.get("car_instance") and st.session_state.car_instance.is_active:
        if hasattr(st.session_state.car_instance, 'predefined_path') and st.session_state.car_instance.predefined_path:
            st.session_state.current_path = st.session_state.car_instance.predefined_path

    map_fig = create_parking_lot_map(gm, st.session_state.current_path, st.session_state.s_node, None, sim.get_current_node(), 
                                    waypoints=st.session_state.multi_waypoints, visit_order=st.session_state.tour_visit_order,
                                    initial_heading=st.session_state.initial_heading)
    main_event = st.plotly_chart(map_fig, width='stretch', on_select="rerun", key="main_map")
    
    if main_event and "selection" in main_event and main_event["selection"]["points"]:
        # Clicks on the main map no longer set heading. 
        # Future use cases can be added here (e.g. quick select waypoint)
        pass
    
    col_l, col_r = st.columns(2)
    with col_l: render_live_logs()
    with col_r: render_sign_gallery()

elif active_tab == "tab2":
    st.header("🔧 System Settings & Calibration")
    
    # --- CALIBRATION ---
    with st.expander("🚀 Motor & Timing Calibration", expanded=True):
        st.subheader("⏱️ Speed & Turn Timings")
        cfg = st.session_state.turn_config
        c1, c2, c3, c4 = st.columns(4)
        t_s = c1.number_input("Straight (s)", 0.0, 10.0, float(cfg.get("STRAIGHT", 1.0)), step=0.1)
        t_l = c2.number_input("Left 90 (s)", 0.0, 10.0, float(cfg.get("90_DEG_LEFT", 1.5)), step=0.05)
        t_r = c3.number_input("Right 90 (s)", 0.0, 10.0, float(cfg.get("90_DEG_RIGHT", 1.5)), step=0.05)
        t_180 = c4.number_input("U-Turn (s)", 0.0, 10.0, float(cfg.get("180_DEG", 2.0)), step=0.1)
        
        if st.button("💾 Save All Timings", width='stretch', type="primary"):
            st.session_state.turn_config = {"STRAIGHT": t_s, "90_DEG_LEFT": t_l, "90_DEG_RIGHT": t_r, "180_DEG": t_180}
            _save_turn_config(st.session_state.turn_config); st.success("Timings saved!")

    with st.expander("📡 Sonar Obstacle Avoidance", expanded=False):
        st.subheader("Sonar Configuration")
        if "sonar_threshold" not in st.session_state:
            from autonomous_main import SONAR_THRESHOLD
            st.session_state.sonar_threshold = SONAR_THRESHOLD
            
        new_thres = st.slider("Safety Threshold (cm)", 5, 100, int(st.session_state.sonar_threshold))
        if new_thres != st.session_state.sonar_threshold:
            st.session_state.sonar_threshold = new_thres
            st.toast(f"Threshold updated to {new_thres} cm")
            
        st.info("💡 Hệ thống sẽ tự động dừng 3 giây khi gặp vật cản và phân tích 3 kịch bản:\n"
                "1. Đối đầu: Quay đầu & Tìm đường mới.\n"
                "2. Đi ngang: Tiếp tục lộ trình cũ.\n"
                "3. Nối tiếp: Tiếp tục lộ trình cũ.")

        st.divider()
        st.subheader("🧪 Hardware Component Tests")
        st.info("Test each movement before saving timings.")
        tc1, tc2, tc3, tc4 = st.columns(4)
        if tc1.button("▶️ Test Straight", width='stretch'):
            if HAS_HW: motor.move_straight(); time.sleep(t_s); motor.stop(); st.toast("Straight test done!")
        if tc2.button("↩️ Test Left 90", width='stretch'):
            if HAS_HW: motor.turn_left(); time.sleep(t_l); motor.stop(); st.toast("Left turn test done!")
        if tc3.button("↪️ Test Right 90", width='stretch'):
            if HAS_HW: motor.turn_right(); time.sleep(t_r); motor.stop(); st.toast("Right turn test done!")
        if tc4.button("🔄 Test U-Turn", width='stretch'):
            if HAS_HW: motor.turn_right(); time.sleep(t_180); motor.stop(); st.toast("U-Turn test done!")

    # --- GRAPH EDITOR ---
    st.divider()
    st.subheader("🗺️ Graph & Node Management")
    
    # Preview Map for Settings
    map_fig_settings = create_parking_lot_map(gm, [], None, None, None, preview_point=st.session_state.settings_click)
    # Using on_select='rerun' to capture clicks. 
    # Since we have a dense background grid, clicking anywhere returns a point from the grid.
    event = st.plotly_chart(map_fig_settings, width='stretch', key="settings_map", on_select="rerun")
    
    if event and "selection" in event and event["selection"]["points"]:
        pt = event["selection"]["points"][0]
        raw_x, raw_y = pt["x"], pt["y"]
        
        # Snapping Logic
        if st.session_state.get("alignment_assist", True):
            # 1. Grid Snap (nearest 5)
            snap_x = round(raw_x / 5) * 5
            snap_y = round(raw_y / 5) * 5
            
            # 2. Node Snap (if very close to an existing node's axis)
            for nid, ndata in gm.graph.nodes(data=True):
                if abs(raw_x - ndata['x']) < 3.0: snap_x = ndata['x']
                if abs(raw_y - ndata['y']) < 3.0: snap_y = ndata['y']
            
            st.session_state.settings_click = (float(snap_x), float(snap_y))
        else:
            st.session_state.settings_click = (raw_x, raw_y)
        st.rerun()

    c1_opt, c2_opt = st.columns(2)
    with c1_opt:
        st.checkbox("🎯 Enable Alignment Assist (Snap to Grid/Nodes)", key="alignment_assist", value=True)
    
    if st.session_state.settings_click:
        c_clear1, c_clear2 = st.columns([4, 1])
        c_clear1.info(f"📍 Selected Position: X={st.session_state.settings_click[0]}, Y={st.session_state.settings_click[1]}")
        if c_clear2.button("✖️ Clear", use_container_width=True):
            st.session_state.settings_click = None
            st.rerun()
    else:
        st.info("💡 Click on the map above to set the position for a new node.")

    tab_edit1, tab_edit2, tab_edit3 = st.tabs(["📍 Manage Nodes", "🔗 Manage Edges", "🆔 Manage UIDs"])
    
    with tab_edit1:
        st.write("### Create New Node")
        c1, c2 = st.columns([2, 1])
        with c1:
            n_type = st.radio("Node Type", ["RFID Node (with Tag)", "Waypoint (Guidance Only)"], horizontal=True)
            
            if n_type.startswith("RFID"):
                if st.button("🔍 Scan RFID for ID", use_container_width=True):
                    if HAS_RFID:
                        with st.spinner("Scanning..."):
                            uid = get_rfid_reader().read_uid_hex(timeout=5.0)
                            if uid: st.session_state.new_node_id = uid; st.rerun()
                    else: st.warning("RFID Hardware not found.")

            n_id = st.text_input("Node ID / Primary UID", key="new_node_id")
            n_lbl = st.text_input("Label (Optional)", key="new_node_label")
        with c2:
            st.write("**Coordinates**")
            if st.session_state.settings_click:
                st.write(f"X: {st.session_state.settings_click[0]}")
                st.write(f"Y: {st.session_state.settings_click[1]}")
            else:
                st.warning("Please click on map first")

        if st.button("➕ Create Node", type="primary", width='stretch'):
            if not st.session_state.settings_click:
                st.error("Please click on map to set position!")
                st.stop()
            
            n_x, n_y = st.session_state.settings_click
            node_id_to_use = n_id
            if not node_id_to_use:
                if n_type.startswith("Waypoint"):
                    node_id_to_use = gm._new_waypoint_id()
                else:
                    st.error("RFID Node requires an ID (Scan tag or type manually)!")
                    st.stop()
            
            gm.add_node(n_x, n_y, node_id_to_use, n_lbl, is_rfid=n_type.startswith("RFID"))
            st.success(f"Node {node_id_to_use} created!"); st.rerun()

        st.divider()
        st.write("### Existing Nodes")
        for nid, ndata in list(gm.graph.nodes(data=True)):
            if not str(nid).startswith("V_"):
                nc1, nc2 = st.columns([4, 1])
                icon = "🆔" if ndata.get('is_rfid', True) else "📍"
                lbl = ndata.get('label') or nid
                nc1.write(f"{icon} **{lbl or nid}** (X:{ndata['x']}, Y:{ndata['y']})")
                if nc2.button("🗑️", key=f"del_node_{nid}"):
                    gm.delete_node(nid); st.rerun()

    with tab_edit2:
        st.write("### Connect Nodes")
        nodes_list = sorted(list(gm.graph.nodes()))
        c1, c2, c3 = st.columns(3)
        u = c1.selectbox("Node A", nodes_list, key="edge_u", format_func=lambda n: gm.graph.nodes[n].get('label') or n)
        v = c2.selectbox("Node B", nodes_list, key="edge_v", format_func=lambda n: gm.graph.nodes[n].get('label') or n)
        w = c3.number_input("Weight (Distance)", 0.1, 1000.0, 10.0)
        
        if st.button("🔗 Add/Update Edge", width='stretch', type="primary"):
            if u == v: st.error("Cannot connect a node to itself!")
            else:
                ulbl = gm.graph.nodes[u].get('label') or u
                vlbl = gm.graph.nodes[v].get('label') or v
                gm.add_edge(u, v, weight=w)
                st.success(f"Edge {ulbl} <-> {vlbl} added!"); st.rerun()
        
        st.write("### Existing Edges")
        for uu, vv, dd in list(gm.graph.edges(data=True)):
            ec1, ec2 = st.columns([4, 1])
            ulbl = gm.graph.nodes[uu].get('label') or uu
            vlbl = gm.graph.nodes[vv].get('label') or vv
            ec1.write(f"**{ulbl}** <-> **{vlbl}** (Weight: {dd['weight']})")
            if ec2.button("🗑️", key=f"del_edge_{uu}_{vv}"):
                gm.delete_edge(uu, vv); st.rerun()

    with tab_edit3:
        st.write("### Node UID Management")
        target_node = st.selectbox("Select Node to Manage", sorted(list(gm.graph.nodes())), key="manage_uid_node", format_func=lambda n: gm.graph.nodes[n].get('label') or n)
        if target_node:
            ndata = gm.graph.nodes[target_node]
            uids = ndata.get("uids", [target_node])
            st.write(f"**Current UIDs for {ndata.get('label') or target_node}:**")
            for uid in uids:
                uc1, uc2 = st.columns([4, 1])
                uc1.code(uid)
                if uid != target_node: # Cannot delete primary ID
                    if uc2.button("🗑️", key=f"del_uid_{target_node}_{uid}"):
                        gm.remove_secondary_uid(target_node, uid); st.rerun()
            
            st.divider()
            st.write("**Add New UID to this Node**")
            
            sc1, sc2 = st.columns(2)
            if sc1.button("🔍 Scan New UID", use_container_width=True):
                if HAS_RFID:
                    with st.spinner("Scanning..."):
                        uid = get_rfid_reader().read_uid_hex(timeout=5.0)
                        if uid: st.session_state.new_uid_input = uid; st.rerun()
            
            new_uid = st.text_input("New UID Hex", key="new_uid_input")
            
            if sc2.button("➕ Add UID", use_container_width=True, type="primary"):
                if new_uid:
                    if gm.add_secondary_uid(target_node, new_uid):
                        st.success("UID added!"); st.rerun()
                    else: st.error("UID already in use or invalid node.")

elif active_tab == "tab3":
    render_wasd_manual_controls()

elif active_tab == "tab4":
    st.header("📝 History")
    logs = [f for f in os.listdir("data") if f.startswith("mission_") and f.endswith(".json")]
    if logs:
        sel = st.selectbox("Select Log", sorted(logs, reverse=True))
        with open(os.path.join("data", sel)) as f: d = json.load(f)
        st.write(f"**Start:** {d.get('mission_start_time')}")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Events:**")
            for e in d.get('events', []): st.caption(f"[{e['time']}] {e['type']}: {e['details']}")
        with c2:
            st.write("**Signs:**")
            for s in d.get('signs_detected', []):
                with st.expander(f"{s['type']}"): st.image(s['image_path'])
    else: st.info("No logs.")
