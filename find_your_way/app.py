import streamlit as st
import json
import time
import threading
import importlib
from core.navigation import NavEngine
import config.graph_data as config
from hardware.rfid import RFIDReader
from autonomous_main import AutonomousCar

def save_config(new_graph, new_turn_table, new_rfid_map, new_turn_config):
    path = "config/graph_data.py"
    content = f"GRAPH = {json.dumps(new_graph, indent=4)}\n\n"
    content += "TURN_TABLE = {\n"
    for (start, end), action in new_turn_table.items():
        content += f"    ('{start}', '{end}'): '{action}',\n"
    content += "}\n\n"
    
    content += f"TURN_CONFIG = {json.dumps(new_turn_config, indent=4)}\n\n"
    content += f"RFID_MAP = {json.dumps(new_rfid_map, indent=4)}\n"
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def reload_config():
    importlib.reload(config)
    return config

@st.cache_resource
def get_rfid_reader():
    return RFIDReader()

st.set_page_config(page_title="Autonomous Car Control", layout="wide")
st.title("LoPi")

# --- TAB 1: Navigating
tabs = st.tabs(["Navigating", "Mapping RFID", "System Config"])

with tabs[0]:
    current_cfg = reload_config()
    nav = NavEngine(current_cfg.GRAPH, current_cfg.TURN_TABLE)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Let's travel")
        nodes = list(current_cfg.GRAPH.keys())
        start = st.selectbox("Start Pos:", nodes)
        target = st.selectbox("Target Pos:", nodes)
        
        if st.button("Find Path"):
            path = nav.get_shortest_path(start, target)
            if path:
                st.success(f"Best Path: {'=>'.join(path)}")
            else:
                st.error("Path not found")
        
        st.divider()
        if st.button("Start Mission"):
            # Truy?n config tuoi m?i nh?t v�o instance xe
            car = AutonomousCar(
                target_node=target,
                graph=current_cfg.GRAPH,
                turn_table=current_cfg.TURN_TABLE,
                rfid_map=current_cfg.RFID_MAP,
                turn_config=current_cfg.TURN_CONFIG
            )
            st.session_state.car_thread = threading.Thread(target=car.execute, daemon=True)
            st.session_state.car_thread.start()
            st.success("Mission started with updated turn times")

        if st.button("Emergency Stop"):
            from hardware.motor import stop
            stop()
            st.warning("Halted")

# --- TAB 2: Mapping RFID
with tabs[1]:
    st.subheader("RFID and Node Mapping")
    reader = get_rfid_reader()
    current_cfg = reload_config()
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Scan your card"):
            with st.spinner("Waiting for card..."):
                uid = reader.read_uid_hex(timeout=5.0)
                if uid:
                    st.session_state.last_uid = uid
                    st.success(f"Read Card: {uid}")
    with c2:
        if "last_uid" in st.session_state:
            uid = st.session_state.last_uid
            st.info(f"Current UID: {uid}")
            node_input = st.text_input("Assign to Node (e.g., A, B, C):", "").upper()
            if st.button("Confirm & Save"):
                if node_input:
                    new_map = current_cfg.RFID_MAP.copy()
                    new_map[uid] = node_input
                    save_config(current_cfg.GRAPH, current_cfg.TURN_TABLE, new_map)
                    st.success(f"Attached {uid} to Node {node_input}")
                    time.sleep(1)
                    st.rerun()

    st.divider()
    st.write("Current Mappings")
    st.table([{"UID": k, "Node": v} for k, v in current_cfg.RFID_MAP.items()])

# --- TAB 3: System Config
with tabs[2]:
    st.subheader("Graph Config")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Nodes & Weights")
        st.json(config.GRAPH)
    with col_b:
        st.write("Turn Table")
        st.json({str(k): v for k, v in config.TURN_TABLE.items()})