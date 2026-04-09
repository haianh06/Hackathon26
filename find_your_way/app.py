import streamlit as st
import os
import sys
import time
import math

try:
    from hardware.rfid import RFIDReader
    from config.graph_data import TURN_CONFIG
    from autonomous_main import AutonomousCar
    import importlib
    import sys
    import threading
    HAS_HW = True
    HAS_RFID = True
except Exception:
    HAS_HW = False
    HAS_RFID = False

@st.cache_resource
def get_rfid_reader():
    if HAS_RFID:
        return RFIDReader()
    return None

from graph.graph_manager import GraphManager
from graph.pathfinding import compute_shortest_path, compute_multi_stop_path
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
TURN_CONFIG_PATH = "data/turn_config.json"

def _load_turn_config() -> dict:
    """Load turn timing from JSON file, fallback to defaults."""
    defaults = {"90_DEG": 1.7}
    try:
        import json
        with open(TURN_CONFIG_PATH) as f:
            data = json.load(f)
            return {**defaults, **data}
    except Exception:
        return defaults

def _save_turn_config(cfg: dict):
    import json
    os.makedirs(os.path.dirname(TURN_CONFIG_PATH), exist_ok=True)
    with open(TURN_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)

if "turn_config" not in st.session_state:
    st.session_state.turn_config = _load_turn_config()

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


tab1, tab2 = st.tabs(["🗺️ Bản Đồ Cốt Lõi", "⚙️ Config Dừng Xe (Calibration)"])

with tab2:
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
            if t_col1.button("⬅️ Test Rẽ Trái 90°", use_container_width=True):
                if HAS_HW:
                    import hardware.motor as motor
                    motor.turn_left()
                    time.sleep(new_90)
                    motor.stop()
                    st.success(f"Rẽ trái {new_90}s xong.")
                else:
                    st.warning("Không có HW.")
            if t_col2.button("➡️ Test Rẽ Phải 90°", use_container_width=True):
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
            st.markdown("**↪️ Rẽ 45° (Nửa góc)**")
            new_45 = st.number_input(
                "Thời gian rẽ 45° (giây)",
                value=float(turn_cfg.get("45_DEG", 0.5)),
                min_value=0.1, max_value=10.0, step=0.05,
                key="cal_45deg",
                help="Thời gian motor rẽ để xe quay đúng 45 độ"
            )
            t_col3, t_col4 = st.columns(2)
            if t_col3.button("⬅️ Test Rẽ Trái 45°", use_container_width=True):
                if HAS_HW:
                    import hardware.motor as motor
                    motor.turn_left()
                    time.sleep(new_45)
                    motor.stop()
                    st.success(f"Rẽ trái {new_45}s xong.")
                else:
                    st.warning("Không có HW.")
            if t_col4.button("➡️ Test Rẽ Phải 45°", use_container_width=True):
                if HAS_HW:
                    import hardware.motor as motor
                    motor.turn_right()
                    time.sleep(new_45)
                    motor.stop()
                    st.success(f"Rẽ phải {new_45}s xong.")
                else:
                    st.warning("Không có HW.")

    st.write("")
    if st.button("💾 Lưu Cấu Hình Rẽ", type="primary", use_container_width=True):
        new_cfg = {"90_DEG": new_90, "45_DEG": new_45}
        _save_turn_config(new_cfg)
        st.session_state.turn_config = new_cfg
        st.success(f"✅ Đã lưu: 90° = {new_90}s | 45° = {new_45}s")

    st.caption(f"📌 Cấu hình hiện tại: 90° = {turn_cfg.get('90_DEG')}s | 45° = {turn_cfg.get('45_DEG')}s")

with tab1:
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
                if st.button("💳 Quét (5s)", use_container_width=True):
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
                st.button("💳 Quét (Tắt)", disabled=True, use_container_width=True)

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
                    use_container_width=True,
                    type="primary"
                ):
                    uid_clean = scanned_id.strip()
                    if uid_clean in gm.graph:
                        st.error(f"❌ ID '{uid_clean}' đã tồn tại trên bản đồ!")
                    else:
                        st.session_state.rfid_pending = {
                            "id": uid_clean,
                            "label": scanned_label.strip()
                        }
                        st.rerun()
            else:
                if st.button("✖️ Hủy ghim", use_container_width=True):
                    st.session_state.rfid_pending = None
                    st.rerun()

        with status_col:
            if pending:
                st.info(
                    f"📌 **Đang chờ ghim:** `{pending['id']}` — _{pending['label']}_  "
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
        if upd_col2.button("Cập nhật", use_container_width=True):
            if sim.force_scan(upd_id):
                st.success("✅ Cập nhật vị trí xe thành công!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("❌ Thẻ không nằm trên lộ trình hiện tại!")

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
                "🚘 Chọn Lộ Trình (Nhấp 2 điểm)",
                "🔀 Đi Nhiều Điểm + Về Lại (Multi-Stop Tour)",
                "📍 Ghim thẻ RFID mới (Dùng mã đã khai báo ở trên)",
            ],
            horizontal=True,
            key="map_click_mode"
        )

    # ──────────────────────────────────────────────
    # MULTI-STOP CONTROL PANEL
    # ──────────────────────────────────────────────
    if click_action == "🔀 Đi Nhiều Điểm + Về Lại (Multi-Stop Tour)":
        with st.container(border=True):
            st.markdown("### 🔀 Multi-Stop Tour Planner")

            ms_col1, ms_col2 = st.columns([3, 2])

            with ms_col1:
                # Status
                multi_start = st.session_state.multi_start
                multi_wps = st.session_state.multi_waypoints

                if multi_start is None:
                    st.info("👇 **Bước 1:** Nhấp vào một node trên bản đồ để chọn **điểm xuất phát**.")
                else:
                    start_label = gm.graph.nodes[multi_start].get('label', '') or multi_start
                    st.success(f"🏁 **Xuất phát:** `{multi_start}` — {start_label}")

                    if not multi_wps:
                        st.info("👇 **Bước 2:** Nhấp các node để **thêm điểm ghé thăm** (tùy số lượng). Rồi nhấn 'Tính Lộ Trình'.")
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
                    use_container_width=True,
                    type="primary",
                    help="Cần chọn điểm xuất phát và ít nhất 1 điểm ghé thăm"
                ):
                    with st.spinner("Đang tính toán lộ trình tối ưu..."):
                        full_path, total_cost, visit_order, is_exact = compute_multi_stop_path(
                            gm,
                            st.session_state.multi_start,
                            st.session_state.multi_waypoints,
                            algo
                        )
                    if not full_path:
                        st.error("❌ Không thể tính lộ trình! Kiểm tra xem các node có kết nối không.")
                    else:
                        # Compute segment_breaks (index in full_path where each leg starts)
                        full_seq = [st.session_state.multi_start] + visit_order
                        breaks = [0]
                        pos = 0
                        for seg_idx in range(1, len(full_seq)):
                            target = full_seq[seg_idx]
                            # Find where target appears in full_path after `pos`
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

                if st.button("🔄 Reset Multi-Stop", use_container_width=True):
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
    is_multi_mode = (click_action == "🔀 Đi Nhiều Điểm + Về Lại (Multi-Stop Tour)")
    canvas_waypoints = st.session_state.multi_waypoints if is_multi_mode else []
    canvas_visit_order = st.session_state.tour_visit_order if is_multi_mode else []
    canvas_breaks = st.session_state.segment_breaks if is_multi_mode else []

    fig = create_parking_lot_map(
        gm,
        st.session_state.current_path,
        st.session_state.get('s_node') if is_multi_mode else start_node,
        None if is_multi_mode else end_node,
        current_node,
        waypoints=canvas_waypoints,
        visit_order=canvas_visit_order,
        segment_breaks=canvas_breaks,
    )

    try:
        selection = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
        if selection and hasattr(selection, "selection") and "points" in selection.selection:
            pts = selection.selection["points"]
            if pts and click_action != "Không làm gì":
                node_id = pts[0].get('customdata')
                clk_x, clk_y = pts[0]['x'], pts[0]['y']

                # ── Mode: RFID Pin ────────────────────────────────────────
                if click_action == "📍 Ghim thẻ RFID mới (Dùng mã đã khai báo ở trên)":
                    pending = st.session_state.rfid_pending
                    if node_id is not None and str(node_id) in gm.graph:
                        st.error("⚠️ Trùng vị trí! Hãy nhấp ra chỗ trống trên bản đồ.")
                    elif pending is None:
                        st.warning("💡 Chưa khóa thông tin thẻ. Hãy điền ID + Tên rồi nhấn **Khóa** trước.")
                    else:
                        id_str = pending["id"]
                        lbl_str = pending["label"]
                        if id_str in gm.graph:
                            st.error(f"❌ Mã '{id_str}' đã có mặt trên Map rồi!")
                        else:
                            gm.add_node(clk_x, clk_y, id_str, lbl_str)
                            st.success(f"✅ Đã ghim thẻ **{id_str}** ({lbl_str}) tại ({clk_x:.1f}, {clk_y:.1f})!")
                            st.session_state.rfid_pending = None
                            st.session_state.last_scanned_uid = ""
                            st.rerun()

                # ── Mode: Multi-Stop ──────────────────────────────────────
                elif click_action == "🔀 Đi Nhiều Điểm + Về Lại (Multi-Stop Tour)":
                    # Resolve node (create virtual if blank area clicked)
                    if not node_id or str(node_id) not in gm.graph:
                        edge, proj = gm.get_closest_edge(clk_x, clk_y)
                        if edge:
                            node_id = gm.create_virtual_node_on_edge(proj[0], proj[1], edge[0], edge[1])

                    if node_id and str(node_id) in gm.graph:
                        node_id = str(node_id)
                        if st.session_state.multi_start is None:
                            # First click → set start
                            st.session_state.multi_start = node_id
                            st.session_state.multi_waypoints = []
                            st.session_state.tour_visit_order = []
                            st.session_state.segment_breaks = []
                            st.session_state.current_path = []
                            st.session_state.path_cost = 0.0
                            st.session_state.s_node = node_id
                            st.session_state.e_node = None
                            sim.reset()
                            st.rerun()
                        else:
                            # Subsequent clicks → add waypoints
                            if node_id == st.session_state.multi_start:
                                st.toast("⚠️ Điểm xuất phát đã được chọn. Hãy chọn điểm khác.", icon="⚠️")
                            elif node_id in st.session_state.multi_waypoints:
                                st.toast(f"ℹ️ Node `{node_id}` đã trong danh sách rồi.", icon="ℹ️")
                            else:
                                st.session_state.multi_waypoints.append(node_id)
                                # Clear previous result when waypoints change
                                st.session_state.current_path = []
                                st.session_state.tour_visit_order = []
                                st.session_state.segment_breaks = []
                                st.session_state.path_cost = 0.0
                                st.rerun()

                # ── Mode: 2-Point Route ───────────────────────────────────
                else:
                    if not node_id or str(node_id) not in gm.graph:
                        edge, proj = gm.get_closest_edge(clk_x, clk_y)
                        if edge:
                            node_id = gm.create_virtual_node_on_edge(proj[0], proj[1], edge[0], edge[1])

                    if node_id and str(node_id) in gm.graph:
                        if st.session_state.click_step == 0:
                            st.session_state.s_node = str(node_id)
                            st.session_state.e_node = None
                            st.session_state.current_path = []
                            sim.reset()
                            st.session_state.click_step = 1
                        else:
                            st.session_state.e_node = str(node_id)
                            path, cost = compute_shortest_path(gm, st.session_state.s_node, st.session_state.e_node, algo if 'algo' in dir() else "Dijkstra")
                            st.session_state.current_path = path
                            st.session_state.path_cost = cost
                            st.session_state.tour_visit_order = []
                            st.session_state.segment_breaks = []
                            sim.start_route(path)
                            st.session_state.click_step = 0
                        st.rerun()

    except TypeError:
        st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────
    # SECTION 2: Node Management Table
    # ──────────────────────────────────────────────
    st.write("---")
    all_nodes = [(n, d) for n, d in gm.graph.nodes(data=True) if not str(n).startswith("V_")]
    with st.expander(f"🗂️ Quản lý Node RFID ({len(all_nodes)} thẻ)", expanded=False):
        if not all_nodes:
            st.info("Chưa có thẻ RFID nào được đăng ký. Hãy ghim thẻ đầu tiên lên bản đồ.")
        else:
            hdr = st.columns([2, 3, 1.2, 1.2, 1.2, 1.2])
            hdr[0].markdown("**🆔 ID Thẻ**")
            hdr[1].markdown("**🏷️ Tên vị trí**")
            hdr[2].markdown("**X**")
            hdr[3].markdown("**Y**")
            hdr[4].markdown("**Sửa**")
            hdr[5].markdown("**Xóa**")
            st.divider()

            editing = st.session_state.editing_node

            for node_id, ndata in all_nodes:
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
                        if sv.button("💾 Lưu", key=f"save_{node_id}", use_container_width=True, type="primary"):
                            gm.edit_node(node_id, new_x, new_y, new_label)
                            st.session_state.editing_node = None
                            st.rerun()
                        if cn.button("✖️ Hủy", key=f"cancel_{node_id}", use_container_width=True):
                            st.session_state.editing_node = None
                            st.rerun()
                else:
                    row = st.columns([2, 3, 1.2, 1.2, 1.2, 1.2])
                    row[0].code(str(node_id))
                    row[1].write(ndata.get('label', '—') or '—')
                    row[2].write(f"{ndata['x']:.1f}")
                    row[3].write(f"{ndata['y']:.1f}")
                    if row[4].button("✏️", key=f"edit_btn_{node_id}", help="Sửa node này"):
                        st.session_state.editing_node = node_id
                        st.rerun()
                    if row[5].button("🗑️", key=f"del_btn_{node_id}", help="Xóa node này"):
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
    def compute_turn_instructions(path, gm):
        if len(path) < 2: return "Đã ở đích"
        instructions = []
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

        st.info("Hệ lệnh Rẽ cho Robot (Turns):\n" + compute_turn_instructions(st.session_state.current_path, gm))

        if HAS_HW:
            st.write("---")
            st.subheader("🤖 Điều Khiển Robot Thực Tế")
            run_col, stop_col = st.columns(2)

            if run_col.button("🚀 BẮT ĐẦU CHẠY XE (RUN MISSION)"):
                target = st.session_state.get('e_node') or (
                    st.session_state.multi_start if st.session_state.tour_visit_order else None
                )
                if not target:
                    st.error("⚠️ XE KHÔNG BIẾT ĐI ĐÂU! Vui lòng lên lộ trình trước khi chạy xe!")
                    st.stop()

                st.session_state.car_logs = []
                gm_edges = {}
                for u in gm.graph.nodes():
                    gm_edges[u] = {}
                    for v in gm.graph.neighbors(u):
                        gm_edges[u][v] = gm.graph[u][v].get('weight', 1)

                if 'autonomous_main' in sys.modules:
                    importlib.reload(sys.modules['autonomous_main'])
                from autonomous_main import AutonomousCar

                rfid_map = {n: n for n in gm.graph.nodes()}

                # Build adjacency dict expected by NavEngine and node coordinates
                graph_adj = {
                    'nodes': {n: {'x': gm.graph.nodes[n]['x'], 'y': gm.graph.nodes[n]['y']} for n in gm.graph.nodes()},
                    'edges': {}
                }
                for u in gm.graph.nodes():
                    graph_adj['edges'][u] = {}
                    for v in gm.graph.neighbors(u):
                        graph_adj['edges'][u][v] = gm.graph[u][v].get('weight', 1)

                # turn_table maps (from_node, to_node) -> action string (from graph_data.py)
                turn_table = TURN_CONFIG if isinstance(TURN_CONFIG, dict) else {}

                # Use calibrated turn timing from session state (saved in data/turn_config.json)
                live_turn_config = st.session_state.turn_config

                car = AutonomousCar(
                    target_node=target,
                    graph=graph_adj,
                    turn_table=turn_table,
                    rfid_map=rfid_map,
                    turn_config=live_turn_config,
                    predefined_path=st.session_state.current_path,
                )
                st.session_state.car_instance = car
                st.session_state.car_thread = threading.Thread(target=car.execute, daemon=True)
                st.session_state.car_thread.start()
                st.success("Đã gửi API điều hướng xuống vi điều khiển xe! Theo dõi hành trình ngoài thực tế.")

            if stop_col.button("🛑 DỪNG XE KHẨN CẤP (STOP)"):
                if 'car_instance' in st.session_state:
                    st.session_state.car_instance.stop_system()
                    st.success("Đã kích hoạt phanh khẩn cấp! Động cơ đã ngắt.")
                else:
                    st.warning("Xe chưa khởi động hoặc đã mất kết nối.")

            st.markdown("### 📷 Camera Trực Tiếp")
            enable_cam = st.checkbox("Bật Streaming Camera (Theo dõi Real-time)", value=False)
            
            if enable_cam:
                try:
                    import cv2
                    if "camera_cap" not in st.session_state:
                        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                        st.session_state.camera_cap = cap
                    
                    cap = st.session_state.camera_cap
                    if cap and cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            cam_col1, cam_col2 = st.columns(2)
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            cam_col1.image(rgb, caption="Cam Thường", use_container_width=True)
                            cam_col2.image(gray, caption="Cam Trắng Đen", use_container_width=True)
                        else:
                            st.error("Lỗi đọc frame từ Camera")
                    else:
                        st.warning("Không tìm thấy Camera vật lý (Index 0)")
                except Exception as e:
                    st.error(f"Lỗi Camera: {e}")
            else:
                if "camera_cap" in st.session_state:
                    st.session_state.camera_cap.release()
                    del st.session_state.camera_cap

            st.markdown("### 📝 Live Console Logs")
            log_box = st.empty()
            log_history = []
            if 'car_instance' in st.session_state:
                log_history = getattr(st.session_state.car_instance, 'log_history', [])

            log_box.code("\n".join(log_history) if log_history else "Chưa có log hệ thống.")

            car_alive = 'car_thread' in st.session_state and st.session_state.car_thread.is_alive()
            is_streaming = locals().get('enable_cam', False)
            
            if car_alive or is_streaming:
                if car_alive and 'car_instance' in st.session_state and getattr(st.session_state.car_instance, 'current_node', None):
                    sim.force_scan(st.session_state.car_instance.current_node)
                
                time.sleep(0.2 if is_streaming else 0.5)
                st.rerun()
