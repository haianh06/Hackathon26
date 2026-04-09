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
from graph.pathfinding import compute_shortest_path
from utils.rfid_simulator import RFIDSimulator
from ui.canvas import create_parking_lot_map
from ui.controls import render_sidebar_controls

st.set_page_config(layout="wide", page_title="Self-Driving Navigation System", page_icon="🚗")

# Session State Initialization
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
    st.session_state.rfid_pending = None  # dict {id, label} khi đang chờ click map

# Node editing state
if "editing_node" not in st.session_state:
    st.session_state.editing_node = None

gm: GraphManager = st.session_state.graph_manager
sim: RFIDSimulator = st.session_state.simulator

# Load UI Controls in Sidebar
mode = render_sidebar_controls(gm, sim, st.session_state.current_path, st.session_state.path_cost)

# Main App Panel
st.title("🚗 Self-Driving Vehicle Pathfinding Map")

col1, col2, col3 = st.columns(3)
col1.metric("Live Nodes", gm.graph.number_of_nodes())
col2.metric("Path Route Distance", f"{st.session_state.path_cost:.2f}")
col3.metric("Vehicle Status", "In Motion" if sim.is_running else ("Idle" if not st.session_state.current_path else "Reached Destination"))


tab1, tab2 = st.tabs(["🗺️ Bản Đồ Cốt Lõi", "⚙️ Config Dừng Xe (Calibration)"])

with tab2:
    st.header("Cân chỉnh Tốc độ (Speed Calibration)")
    st.write("Cơ chế: Chạy thử xe 1 khoảng thời gian thực tế, đo cự ly vật lý, so đối chiếu 2 điểm trên web để lấy ra Cự ly Pixels. Lấy Pixel / Giây.")
    new_speed = st.number_input("Tốc độ xe thực tế (Pixel trên map / 1 giây):", value=float(gm.speed_px_per_sec))
    if st.button("Lưu Tốc Độ"):
        gm.speed_px_per_sec = new_speed
        gm._auto_save()
        st.success("Lưu thành công!")
        
    st.divider()
    st.subheader("Test Motor Thẳng")
    test_secs = st.number_input("Chạy test trong bao nhiêu giây:", value=2.0)
    if st.button("Chạy thử để đo"):
        import hardware.motor as motor
        motor.move_straight()
        import time
        time.sleep(test_secs)
        motor.stop()
        st.success(f"Đã chạy {test_secs} giây. Hãy đem thước ra đo trên sàn!")

with tab1:
    # ──────────────────────────────────────────────
    # SECTION 1: RFID Node Registration Wizard
    # ──────────────────────────────────────────────
    st.subheader("📍 Đăng Ký Thẻ RFID Mới")

    with st.container(border=True):
        # ── BƯỚC 1: Điền thông tin thẻ ──
        st.markdown("**Bước 1 — Khai báo thông tin thẻ**")
        form_col1, form_col2, form_col3 = st.columns([2, 3, 1])

        with form_col1:
            # Nếu vừa quét thẻ xong (rerun), điền UID vào ô trước khi widget được tạo
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

        # Validate & Lock button
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

    # Map mode radio — chỉ hiện chọn lộ trình khi không đang ghim RFID
    pending = st.session_state.rfid_pending
    if pending:
        click_action = "📍 Ghim thẻ RFID mới (Dùng mã đã khai báo ở trên)"
        st.info(f"🔒 Chế độ ghim đang chạy: `{pending['id']}` — _{pending['label']}_. Nhấp map để xác nhận vị trí.")
    else:
        click_action = st.radio(
            "Chế độ nhấp Map:",
            ["🚘 Chọn Lộ Trình (Nhấp 2 điểm)", "📍 Ghim thẻ RFID mới (Dùng mã đã khai báo ở trên)"],
            horizontal=True
        )

    # Render map
    fig = create_parking_lot_map(gm, st.session_state.current_path, start_node, end_node, current_node)

    try:
        selection = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
        if selection and hasattr(selection, "selection") and "points" in selection.selection:
            pts = selection.selection["points"]
            if pts and click_action != "Không làm gì":
                node_id = pts[0].get('customdata')
                clk_x, clk_y = pts[0]['x'], pts[0]['y']

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

                else:
                    # Chế độ: Nhấp 2 Điểm
                    # Tạo Node ảo nếu bấm vào khoảng trắng
                    if not node_id or str(node_id) not in gm.graph:
                         edge, proj = gm.get_closest_edge(clk_x, clk_y)
                         if edge:
                             node_id = gm.create_virtual_node_on_edge(proj[0], proj[1], edge[0], edge[1])

                    if node_id and str(node_id) in gm.graph:
                         if st.session_state.click_step == 0:
                             # Click lần 1 -> Đặt là Nguồn
                             st.session_state.s_node = str(node_id)
                             st.session_state.e_node = None
                             st.session_state.current_path = []
                             sim.reset()
                             st.session_state.click_step = 1
                         else:
                             # Click lần 2 -> Đặt Đích
                             st.session_state.e_node = str(node_id)
                             # Gọi Pathfinding ngay lập tức
                             path, cost = compute_shortest_path(gm, st.session_state.s_node, st.session_state.e_node, "Dijkstra")
                             st.session_state.current_path = path
                             st.session_state.path_cost = cost
                             sim.start_route(path)
                             st.session_state.click_step = 0
                         st.rerun()

    except TypeError:
        # Streamlit older versions support
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
            # Header
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
                    # Inline edit form
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
                        st.session_state.current_path = []
                        st.rerun()


    def compute_turn_instructions(path, gm):
        if len(path) < 2: return "Đã ở đích"
        instructions = []
        for i in range(1, len(path)-1):
            prev = path[i-1]
            curr = path[i]
            nxt = path[i+1]

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

            # Bổ sung Text tính toán Timer nếu đích là Điểm Ảo
            if i == len(path) - 2 and str(nxt).startswith("V_"):
                dist_px = math.hypot(dx2, dy2)
                time_s = dist_px / gm.speed_px_per_sec
                turn += f" (rồi chạy Mù thêm {time_s:.2f}s để tới Đích 🎯)"

            instructions.append(f"• Qua chốt **{curr}**: {turn}")

        if not instructions: return "Cứ đi thẳng không có khúc cua."
        return "\n".join(instructions)

    if st.session_state.current_path:
        st.success("Lộ trình dạng Chuỗi Điểm (Sequence):")
        st.code(" → ".join(st.session_state.current_path))

        st.info("Hệ lệnh Rẽ cho Robot (Turns):\n" + compute_turn_instructions(st.session_state.current_path, gm))

        if HAS_HW:
            st.write("---")
            st.subheader("🤖 Điều Khiển Robot Thực Tế")
            run_col, stop_col = st.columns(2)

            if run_col.button("🚀 BẮT ĐẦU CHẠY XE (RUN MISSION)"):
                if not st.session_state.get('e_node'):
                    st.error("⚠️ XE KHÔNG BIẾT ĐI ĐÂU! Vui lòng Click 2 điểm trên bản đồ để lên lộ trình trước khi chạy xe!")
                    st.stop()

                st.session_state.car_logs = []
                # Truyền nguyên xi tọa độ map để NavEngine của xe tự tính góc Vector
                gm_edges = {}
                for u in gm.graph.nodes():
                     gm_edges[u] = {}
                     for v in gm.graph.neighbors(u):
                          gm_edges[u][v] = gm.graph[u][v].get('weight', 1)

                graph_doc = {
                     "nodes": {n: {"x": gm.graph.nodes[n]['x'], "y": gm.graph.nodes[n]['y']} for n in gm.graph.nodes()},
                     "edges": gm_edges
                }

                # Hot reload the module to fetch our newly added modifications (because it resides outside Streamlit's scope)
                if 'autonomous_main' in sys.modules:
                    importlib.reload(sys.modules['autonomous_main'])
                from autonomous_main import AutonomousCar

                rfid_map = {n: n for n in gm.graph.nodes()}
                car = AutonomousCar(
                    target_node=st.session_state.e_node,
                    graph_doc=graph_doc,
                    rfid_map=rfid_map,
                    turn_config=TURN_CONFIG,
                    pre_planned_path=st.session_state.current_path,
                    speed_px_per_sec=gm.speed_px_per_sec
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

            st.markdown("### 📝 Live Console Logs")
            log_box = st.empty()
            log_history = []
            if 'car_instance' in st.session_state:
                 log_history = st.session_state.car_instance.log_history

            log_box.code("\n".join(log_history) if log_history else "Chưa có log hệ thống.")

            if 'car_thread' in st.session_state and st.session_state.car_thread.is_alive():
                if 'car_instance' in st.session_state and getattr(st.session_state.car_instance, 'current_node', None):
                     # Nháy đồng bộ vị trí con xe thực lên bản đồ Đồ hoạ
                     sim.force_scan(st.session_state.car_instance.current_node)

                time.sleep(0.5)
                st.rerun()
