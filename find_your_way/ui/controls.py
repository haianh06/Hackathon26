"""
controls.py
Renders the Streamlit sidebar controls panel.
Returns the current map interaction mode string chosen by the user.
"""

import streamlit as st


def render_sidebar_controls(gm, sim, current_path: list, path_cost: float) -> str:
    """
    Render sidebar UI widgets (node management, edge management, simulator controls).

    Parameters
    ----------
    gm           : GraphManager  – the live graph manager instance
    sim          : RFIDSimulator – the live simulator instance
    current_path : list          – current computed path (list of node IDs)
    path_cost    : float         – total cost of current_path

    Returns
    -------
    str – currently selected map interaction mode (unused in sidebar context,
          kept for API compatibility with app.py)
    """
    with st.sidebar:
        st.header("🗺️ Map Controls")

        # ── Edge Management ──────────────────────────────────────────────
        st.subheader("🔗 Kết Nối Nodes (Edges)")

        all_node_ids = [n for n in gm.graph.nodes() if not str(n).startswith("V_")]

        if len(all_node_ids) >= 2:
            col_u, col_v = st.columns(2)
            edge_u = col_u.selectbox("Node A", all_node_ids, key="edge_u")
            edge_v = col_v.selectbox("Node B", all_node_ids, key="edge_v", index=min(1, len(all_node_ids) - 1))
            edge_w = st.number_input("Trọng số (Weight)", value=1.0, min_value=0.1, step=0.5, key="edge_w")

            e_col1, e_col2 = st.columns(2)
            if e_col1.button("➕ Thêm Edge", width='stretch'):
                if edge_u == edge_v:
                    st.error("Không thể nối một node với chính nó!")
                elif gm.graph.has_edge(edge_u, edge_v):
                    st.warning("Edge đã tồn tại rồi.")
                else:
                    gm.add_edge(edge_u, edge_v, weight=edge_w)
                    st.success(f"✅ Đã thêm edge {edge_u} ↔ {edge_v}")
                    st.rerun()

            if e_col2.button("🗑️ Xóa Edge", width='stretch'):
                if gm.graph.has_edge(edge_u, edge_v):
                    gm.delete_edge(edge_u, edge_v)
                    st.success(f"✅ Đã xóa edge {edge_u} ↔ {edge_v}")
                    st.rerun()
                else:
                    st.warning("Edge này không tồn tại.")
        else:
            st.info("Cần ít nhất 2 node để tạo edge.")

        st.divider()

        # ── Node Deletion ─────────────────────────────────────────────────
        st.subheader("🗑️ Xóa Node")
        if all_node_ids:
            # Build display labels: "Label (ID)" or just "ID" if no label
            def _node_label(nid):
                lbl = gm.graph.nodes[nid].get("label", "")
                return f"{lbl}  [{nid}]" if lbl else f"[{nid}]"

            node_options = {_node_label(n): n for n in all_node_ids}
            del_display = st.selectbox(
                "Chọn node cần xóa:",
                list(node_options.keys()),
                key="del_node_select"
            )
            del_node_id = node_options[del_display]

            # Show node info
            ndata = gm.graph.nodes[del_node_id]
            st.caption(
                f"📍 Vị trí: ({ndata.get('x', 0):.1f}, {ndata.get('y', 0):.1f})  |  "
                f"Kết nối: {gm.graph.degree(del_node_id)} edge(s)"
            )

            if st.button("🗑️ Xóa Node này", width='stretch', type="primary"):
                gm.delete_node(del_node_id)
                # Clear from session state if it was start/end/current node
                if st.session_state.get("s_node") == del_node_id:
                    st.session_state.s_node = None
                if st.session_state.get("e_node") == del_node_id:
                    st.session_state.e_node = None
                if del_node_id in st.session_state.get("current_path", []):
                    st.session_state.current_path = []
                    st.session_state.path_cost = 0.0
                    sim.reset()
                st.success(f"✅ Đã xóa node **{del_display}** cùng tất cả các edge liên quan.")
                st.rerun()
        else:
            st.info("Chưa có node nào trên bản đồ.")

        st.divider()

        # ── Route Info ───────────────────────────────────────────────────
        st.subheader("📍 Lộ Trình Hiện Tại")
        if current_path:
            st.success(f"Số node: {len(current_path)}  |  Cost: {path_cost:.2f}")
            idx, total = sim.get_progress()
            if total > 0:
                st.progress(idx / max(total - 1, 1), text=f"Sim: {idx}/{total - 1}")
        else:
            st.info("Chưa có lộ trình. Nhấp 2 điểm trên bản đồ.")

        if st.button("🔄 Reset Lộ Trình", width='stretch'):
            st.session_state.current_path = []
            st.session_state.path_cost = 0.0
            st.session_state.s_node = None
            st.session_state.e_node = None
            st.session_state.click_step = 0
            st.session_state.multi_start = None
            st.session_state.multi_waypoints = []
            st.session_state.tour_visit_order = []
            st.session_state.segment_breaks = []
            sim.reset()
            st.rerun()

        st.divider()

        # ── Algorithm selector (informational — used by app.py directly) ─
        st.subheader("⚙️ Thuật Toán")
        algo = st.selectbox("Chọn thuật toán:", ["Dijkstra", "A*"], key="algo_select")

        st.divider()

        # ── Virtual Node Management ──────────────────────────────────────
        st.subheader("🔮 Node Ảo (Tạm Thời)")
        virtual_nodes = [(n, d) for n, d in gm.graph.nodes(data=True) if str(n).startswith("V_")]
        n_virtual = len(virtual_nodes)

        if n_virtual == 0:
            st.caption("Không có node ảo nào trên bản đồ.")
        else:
            st.warning(f"⚠️ Có **{n_virtual}** node ảo đang tồn tại trên bản đồ.")

            if st.button("🗑️ Xóa Tất Cả Node Ảo", width='stretch', type="primary"):
                removed = gm.clear_virtual_nodes()
                # Clear route state because virtual nodes may be part of current path
                st.session_state.current_path = []
                st.session_state.path_cost = 0.0
                st.session_state.s_node = None
                st.session_state.e_node = None
                st.session_state.multi_start = None
                st.session_state.multi_waypoints = []
                st.session_state.tour_visit_order = []
                st.session_state.segment_breaks = []
                st.session_state.click_step = 0
                sim.reset()
                st.success(f"✅ Đã xóa {removed} node ảo và phục hồi các edge gốc.")
                st.rerun()

            with st.expander(f"📋 Danh sách ({n_virtual} node)", expanded=False):
                for vid, vdata in virtual_nodes:
                    vcol1, vcol2 = st.columns([3, 1])
                    vcol1.caption(f"`{vid}` ({vdata.get('x', 0):.1f}, {vdata.get('y', 0):.1f})")
                    if vcol2.button("✖", key=f"del_virt_{vid}", help=f"Xóa {vid}"):
                        gm.delete_virtual_node(vid)
                        # If this virtual node is in the current path, reset
                        if vid in st.session_state.get("current_path", []):
                            st.session_state.current_path = []
                            st.session_state.path_cost = 0.0
                            st.session_state.s_node = None
                            st.session_state.e_node = None
                            st.session_state.multi_start = None
                            st.session_state.multi_waypoints = []
                            st.session_state.tour_visit_order = []
                            st.session_state.segment_breaks = []
                            sim.reset()
                        st.rerun()

        st.divider()

        # ── Save / Load ──────────────────────────────────────────────────
        st.subheader("💾 Lưu / Tải Bản Đồ")
        if st.button("💾 Lưu Map ngay", width='stretch'):
            gm._auto_save()
            st.success("Đã lưu!")

        real_nodes = gm.graph.number_of_nodes() - n_virtual
        st.caption(f"Nodes thực: {real_nodes}  |  Node ảo: {n_virtual}  |  Edges: {gm.graph.number_of_edges()}")

    return algo
