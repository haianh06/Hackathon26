import plotly.graph_objects as go
import numpy as np

# Segment colors for multi-stop path visualization
_SEGMENT_COLORS = [
    "#00e676",  # green
    "#2979ff",  # blue
    "#ff6d00",  # orange
    "#d500f9",  # purple
    "#ffea00",  # yellow
    "#00e5ff",  # cyan
    "#ff1744",  # red
    "#76ff03",  # lime
    "#ff4081",  # pink
    "#1de9b6",  # teal
]

def create_parking_lot_map(
    graph_manager,
    current_path,
    start_node,
    end_node,
    sim_node,
    waypoints=None,
    visit_order=None,
    segment_breaks=None,
):
    """
    Render the interactive parking lot map.

    Parameters
    ----------
    graph_manager  : GraphManager
    current_path   : list of node IDs forming the full path to highlight
    start_node     : str – starting node (highlighted green)
    end_node       : str – single-destination node (highlighted red). None in multi-stop mode.
    sim_node       : str – current simulator position (highlighted yellow)
    waypoints      : list[str] – waypoints chosen by user (multi-stop mode)
    visit_order    : list[str] – optimized order of waypoints (multi-stop mode)
    segment_breaks : list[int] – indices in current_path where each new leg starts (multi-stop mode)
    """
    G = graph_manager.graph
    waypoints = waypoints or []
    visit_order = visit_order or []

    fig = go.Figure()

    # Add hidden scatter grid across plotting area [-5, 105] to capture ANY click
    grid_x, grid_y = np.meshgrid(np.arange(-5, 105, 5), np.arange(-5, 105, 5))
    fig.add_trace(go.Scatter(
        x=grid_x.flatten(),
        y=grid_y.flatten(),
        mode='markers',
        marker=dict(size=14, color='rgba(0,0,0,0)'),
        hoverinfo='none',
        showlegend=False,
        name='background_grid'
    ))

    # Add shapes for white obstacle blocks
    shapes = [
        dict(type="rect", x0=10, y0=70, x1=90, y1=95, fillcolor="white", line=dict(color="white"), layer="below"),
        dict(type="rect", x0=10, y0=35, x1=40, y1=60, fillcolor="white", line=dict(color="white"), layer="below"),
        dict(type="rect", x0=60, y0=35, x1=90, y1=60, fillcolor="white", line=dict(color="white"), layer="below"),
        dict(type="rect", x0=10, y0=5,  x1=40, y1=25, fillcolor="white", line=dict(color="white"), layer="below"),
        dict(type="rect", x0=60, y0=5,  x1=90, y1=25, fillcolor="white", line=dict(color="white"), layer="below"),
    ]

    # ── Regular (non-path) edges ──────────────────────────────────────────
    path_edge_set = set()
    if len(current_path) > 1:
        for i in range(len(current_path) - 1):
            path_edge_set.add((current_path[i], current_path[i + 1]))
            path_edge_set.add((current_path[i + 1], current_path[i]))

    edge_x, edge_y = [], []
    for u, v in G.edges():
        if (u, v) not in path_edge_set:
            x0, y0 = G.nodes[u]['x'], G.nodes[u]['y']
            x1, y1 = G.nodes[v]['x'], G.nodes[v]['y']
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=4, color='white', dash='dash'),
            hoverinfo='none', mode='lines', name='Road Network'
        ))

    # ── Path edges ────────────────────────────────────────────────────────
    if len(current_path) > 1:
        if segment_breaks and len(segment_breaks) > 1:
            # Multi-stop: draw each leg with a different color
            for seg_idx in range(len(segment_breaks) - 1):
                seg_start = segment_breaks[seg_idx]
                seg_end   = segment_breaks[seg_idx + 1] + 1
                seg_nodes = current_path[seg_start:seg_end]
                seg_x, seg_y = [], []
                for i in range(len(seg_nodes) - 1):
                    u, v = seg_nodes[i], seg_nodes[i + 1]
                    if u in G.nodes and v in G.nodes:
                        seg_x.extend([G.nodes[u]['x'], G.nodes[v]['x'], None])
                        seg_y.extend([G.nodes[u]['y'], G.nodes[v]['y'], None])
                color = _SEGMENT_COLORS[seg_idx % len(_SEGMENT_COLORS)]
                fig.add_trace(go.Scatter(
                    x=seg_x, y=seg_y,
                    line=dict(width=7, color=color),
                    hoverinfo='none', mode='lines',
                    name=f'Leg {seg_idx + 1}'
                ))
        else:
            # Single-destination mode: uniform green
            path_x, path_y = [], []
            for i in range(len(current_path) - 1):
                u, v = current_path[i], current_path[i + 1]
                if u in G.nodes and v in G.nodes:
                    path_x.extend([G.nodes[u]['x'], G.nodes[v]['x'], None])
                    path_y.extend([G.nodes[u]['y'], G.nodes[v]['y'], None])
            if path_x:
                fig.add_trace(go.Scatter(
                    x=path_x, y=path_y,
                    line=dict(width=7, color='green'),
                    hoverinfo='none', mode='lines', name='Shortest Path'
                ))

    # ── Nodes ─────────────────────────────────────────────────────────────
    # Build visit_order_map: node -> display number (1-indexed)
    visit_rank = {}
    if visit_order:
        for i, n in enumerate(visit_order):
            visit_rank[n] = i + 1

    node_x, node_y, custom_data, node_colors, node_text, sizes = [], [], [], [], [], []
    for node, data in G.nodes(data=True):
        node_x.append(data['x'])
        node_y.append(data['y'])
        custom_data.append(node)

        c = '#1f78b4'
        s = 10

        if node == sim_node:
            c = 'yellow'; s = 18
        elif node == start_node:
            c = '#00e676'; s = 18   # bright green for start
        elif node == end_node:
            c = 'red'; s = 15
        elif node in waypoints and node in visit_rank:
            # Color waypoints by segment
            seg_color = _SEGMENT_COLORS[(visit_rank[node] - 1) % len(_SEGMENT_COLORS)]
            c = seg_color; s = 15
        elif node in current_path:
            c = 'orange'; s = 12

        node_colors.append(c)
        sizes.append(s)

        # Hover text
        label = data.get('label', '') or node
        rank_txt = f"<br>🔢 Ghé thăm #: {visit_rank[node]}" if node in visit_rank else ""
        role_txt = ""
        if node == start_node:
            role_txt = "<br>🟢 Xuất phát / Về đích"
        elif node == end_node and not waypoints:
            role_txt = "<br>🔴 Điểm đến"
        node_text.append(f"<b>{label}</b><br>ID: {node}{rank_txt}{role_txt}")

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers',
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(showscale=False, color=node_colors, size=sizes, line_width=2),
        name="RFID Checkpoints",
        customdata=custom_data,
        textfont=dict(color="black")
    ))

    # ── Waypoint order annotations ────────────────────────────────────────
    if visit_order:
        # Annotate start
        if start_node and start_node in G.nodes:
            fig.add_annotation(
                x=G.nodes[start_node]['x'],
                y=G.nodes[start_node]['y'],
                text="🏁",
                showarrow=False,
                font=dict(size=16),
                yshift=18,
            )
        for rank, node in enumerate(visit_order, start=1):
            if node in G.nodes:
                fig.add_annotation(
                    x=G.nodes[node]['x'],
                    y=G.nodes[node]['y'],
                    text=f"<b>{rank}</b>",
                    showarrow=False,
                    font=dict(size=13, color="white"),
                    bgcolor=_SEGMENT_COLORS[(rank - 1) % len(_SEGMENT_COLORS)],
                    borderpad=3,
                    yshift=18,
                )

    fig.update_layout(
        title='<br>Interactive Parking Layout Map',
        title_font_size=16,
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='#999999',
        paper_bgcolor='#E0E0E0',
        shapes=shapes,
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 105]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 105]),
        clickmode='event+select',
        dragmode='pan',
        title_font=dict(color="blue", size=20)
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
