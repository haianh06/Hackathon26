import plotly.graph_objects as go
import numpy as np

def create_parking_lot_map(graph_manager, current_path, start_node, end_node, sim_node):
    G = graph_manager.graph
    
    fig = go.Figure()
    
    # Add hidden scatter grid across plotting area [-5, 105] to capture ANY click
    # The grid density ensures clicking anywhere finds a closest scatter node
    # to emit the coordinate back to streamlit
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
        # Top large block
        dict(type="rect", x0=10, y0=70, x1=90, y1=95, fillcolor="white", line=dict(color="white"), layer="below"),
        # Mid left
        dict(type="rect", x0=10, y0=35, x1=40, y1=60, fillcolor="white", line=dict(color="white"), layer="below"),
        # Mid right
        dict(type="rect", x0=60, y0=35, x1=90, y1=60, fillcolor="white", line=dict(color="white"), layer="below"),
        # Bot left
        dict(type="rect", x0=10, y0=5, x1=40, y1=25, fillcolor="white", line=dict(color="white"), layer="below"),
        # Bot right
        dict(type="rect", x0=60, y0=5, x1=90, y1=25, fillcolor="white", line=dict(color="white"), layer="below"),
    ]

    # Plot Edges
    edge_x = []
    edge_y = []
    
    # Path Edges
    path_edges = []
    if len(current_path) > 1:
        path_edges = list(zip(current_path[:-1], current_path[1:]))
    path_x, path_y = [], []
    
    for u, v, data in G.edges(data=True):
        x0, y0 = G.nodes[u]['x'], G.nodes[u]['y']
        x1, y1 = G.nodes[v]['x'], G.nodes[v]['y']
        
        is_path = (u, v) in path_edges or (v, u) in path_edges
        if is_path:
            path_x.extend([x0, x1, None])
            path_y.extend([y0, y1, None])
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    # Regular roads (thick white dashed)
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=4, color='white', dash='dash'),
            hoverinfo='none', mode='lines', name='Road Network'
        ))
    
    if path_x:
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            line=dict(width=7, color='green'),
            hoverinfo='none', mode='lines', name='Shortest Path'
        ))

    # Plot Nodes
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
            c = 'green'; s = 15
        elif node == end_node:
            c = 'red'; s = 15
        elif node in current_path:
            c = 'orange'; s = 12
            
        node_colors.append(c)
        sizes.append(s)
        node_text.append(f"Node: {data.get('label', '')}<br>ID: {node}")
        
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers',
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(showscale=False, color=node_colors, size=sizes, line_width=2),
        name="RFID Checkpoints",
        customdata=custom_data,
        textfont=dict(color="black") 
    ))

    # Setting up layout
    fig.update_layout(
         title='<br>Interactive Parking Layout Map',
         title_font_size=16,
         showlegend=True,
         hovermode='closest',
         plot_bgcolor='#999999',  # Grey background matching reference
         paper_bgcolor='#E0E0E0', 
         shapes=shapes,
         margin=dict(b=20,l=5,r=5,t=40),
         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 105]),
         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 105]),
         clickmode='event+select',
         dragmode='pan',
         title_font=dict(
            color="blue",
            size=20
        )
    )
    
    # Scale fix
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig
