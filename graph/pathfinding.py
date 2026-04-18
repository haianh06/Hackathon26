"""
pathfinding.py
Wraps NetworkX shortest-path algorithms for use with GraphManager.
Supports Dijkstra and A* algorithms.
Also implements multi-stop TSP tour planning (Held-Karp DP + Greedy 2-opt fallback).
"""

import math
import itertools
import networkx as nx


def _euclidean_heuristic(G: nx.Graph, u, v):
    """Euclidean distance heuristic for A*."""
    x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
    x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
    return math.hypot(x2 - x1, y2 - y1)


def compute_shortest_path(graph_manager, source: str, target: str, algorithm: str = "Dijkstra", initial_heading: float = None):
    """
    Compute the shortest path between source and target nodes, optionally preferring 
    the direction of 'initial_heading' at the start.
    """
    G = graph_manager.graph
    if source not in G or target not in G:
        return [], 0.0

    # If heading is set, we use a temporary directed graph to apply turn penalties
    # only on the edges leaving the source node.
    if initial_heading is not None:
        DG = nx.DiGraph(G)
        x1, y1 = G.nodes[source]["x"], G.nodes[source]["y"]
        
        for neighbor in G.neighbors(source):
            x2, y2 = G.nodes[neighbor]["x"], G.nodes[neighbor]["y"]
            edge_angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360
            
            diff = (edge_angle - initial_heading) % 360
            if diff > 180: diff -= 360
            
            penalty = 0.0
            if abs(diff) > 135: 
                penalty = 100.0  # Big penalty for turn-around
            elif abs(diff) > 45:
                penalty = 10.0   # Small penalty for side turns
            
            # Apply penalty to the outgoing edge from source
            if DG.has_edge(source, neighbor):
                DG[source][neighbor]["weight"] = G[source][neighbor].get("weight", 1.0) + penalty
        
        working_graph = DG
    else:
        working_graph = G

    try:
        if algorithm == "A*":
            path = nx.astar_path(
                working_graph, source, target,
                heuristic=lambda u, v: _euclidean_heuristic(G, u, v),
                weight="weight"
            )
        else:
            path = nx.dijkstra_path(working_graph, source, target, weight="weight")

        # Always calculate the TRUE cost using the original graph (w/o penalties)
        cost = nx.path_weight(G, path, weight="weight")
        return path, cost
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], 0.0


# ═══════════════════════════════════════════════════════════════
#  MULTI-STOP TSP PLANNING
# ═══════════════════════════════════════════════════════════════

def compute_distance_matrix(graph_manager, nodes: list, algorithm: str = "Dijkstra") -> dict:
    """
    Compute pairwise shortest-path costs and sub-paths between all nodes.

    Returns
    -------
    dict[(u, v)] = {"cost": float, "path": list[str]}
    Returns cost=inf and path=[] if no path exists between a pair.
    """
    matrix = {}
    for u in nodes:
        for v in nodes:
            if u == v:
                matrix[(u, v)] = {"cost": 0.0, "path": [u]}
                continue
            path, cost = compute_shortest_path(graph_manager, u, v, algorithm)
            if not path:
                cost = float("inf")
            matrix[(u, v)] = {"cost": cost, "path": path}
    return matrix


def _solve_tsp_held_karp(nodes: list, start: str, dist_matrix: dict):
    """
    Exact TSP via Held-Karp DP.
    Works well for n <= 15 waypoints + start.

    Parameters
    ----------
    nodes       : list of node IDs to visit (NOT including start, or including — handled below)
    start       : starting node ID
    dist_matrix : dict from compute_distance_matrix covering all nodes + start

    Returns
    -------
    (visit_order: list[str], total_cost: float)
    visit_order is the order of nodes (excluding start at begin/end).
    """
    # All nodes to visit (excluding start node itself)
    waypoints = [n for n in nodes if n != start]
    n = len(waypoints)

    if n == 0:
        return [], 0.0

    if n == 1:
        cost = dist_matrix[(start, waypoints[0])]["cost"] + dist_matrix[(waypoints[0], start)]["cost"]
        return waypoints, cost

    # Index waypoints 0..n-1
    idx = {wp: i for i, wp in enumerate(waypoints)}

    # dp[mask][i] = min cost to visit subset 'mask' ending at waypoint i
    INF = float("inf")
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]

    # Initialize: start -> each waypoint
    for i, wp in enumerate(waypoints):
        c = dist_matrix[(start, wp)]["cost"]
        dp[1 << i][i] = c

    # Fill DP
    for mask in range(1, 1 << n):
        for u_idx in range(n):
            if not (mask & (1 << u_idx)):
                continue
            if dp[mask][u_idx] == INF:
                continue
            for v_idx in range(n):
                if mask & (1 << v_idx):
                    continue
                new_mask = mask | (1 << v_idx)
                u_node = waypoints[u_idx]
                v_node = waypoints[v_idx]
                new_cost = dp[mask][u_idx] + dist_matrix[(u_node, v_node)]["cost"]
                if new_cost < dp[new_mask][v_idx]:
                    dp[new_mask][v_idx] = new_cost
                    parent[new_mask][v_idx] = u_idx

    # Find best last node to NOT return to start
    full_mask = (1 << n) - 1
    best_cost = INF
    last_idx = -1
    for i, wp in enumerate(waypoints):
        total = dp[full_mask][i]
        if total < best_cost:
            best_cost = total
            last_idx = i

    if last_idx == -1:
        return waypoints, INF  # no valid tour found

    # Backtrack to find order
    order_reversed = []
    mask = full_mask
    cur = last_idx
    while cur != -1:
        order_reversed.append(waypoints[cur])
        prev = parent[mask][cur]
        mask ^= (1 << cur)
        cur = prev

    visit_order = list(reversed(order_reversed))
    return visit_order, best_cost


def _two_opt_improve(visit_order: list, start: str, dist_matrix: dict):
    """
    Apply 2-opt local search to improve a TSP tour.
    Works on visit_order (list of waypoints, NOT including start at begin/end).
    """
    def tour_cost(order):
        nodes_seq = [start] + order
        return sum(
            dist_matrix.get((nodes_seq[i], nodes_seq[i + 1]), {"cost": float("inf")})["cost"]
            for i in range(len(nodes_seq) - 1)
        )

    best = visit_order[:]
    best_cost = tour_cost(best)
    improved = True

    while improved:
        improved = False
        for i in range(len(best)):
            for j in range(i + 2, len(best)):
                new_order = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                new_cost = tour_cost(new_order)
                if new_cost < best_cost - 1e-9:
                    best = new_order
                    best_cost = new_cost
                    improved = True

    return best, best_cost


def _solve_tsp_greedy_2opt(nodes: list, start: str, dist_matrix: dict):
    """
    Greedy nearest-neighbor TSP heuristic followed by 2-opt improvement.
    Used as fallback for n > 15 waypoints.
    """
    waypoints = [n for n in nodes if n != start]
    if not waypoints:
        return [], 0.0

    # Greedy nearest neighbor
    unvisited = set(waypoints)
    order = []
    current = start

    while unvisited:
        best_next = min(unvisited, key=lambda v: dist_matrix.get((current, v), {"cost": float("inf")})["cost"])
        order.append(best_next)
        unvisited.remove(best_next)
        current = best_next

    # 2-opt improvement
    order, cost = _two_opt_improve(order, start, dist_matrix)
    return order, cost


def solve_tsp_tour(dist_matrix: dict, nodes: list, start: str):
    """
    Solve TSP tour: start → visit all nodes → return to start.
    Chooses algorithm based on number of waypoints.

    Parameters
    ----------
    dist_matrix : from compute_distance_matrix()
    nodes       : ALL nodes in the tour (including start)
    start       : starting node ID

    Returns
    -------
    (visit_order: list[str], total_cost: float)
    visit_order does NOT include start; the full tour is [start]+visit_order+[start]
    """
    waypoints = [n for n in nodes if n != start]
    n = len(waypoints)

    if n == 0:
        return [], 0.0

    if n <= 12:
        return _solve_tsp_held_karp(nodes, start, dist_matrix)
    else:
        return _solve_tsp_greedy_2opt(nodes, start, dist_matrix)


def compute_multi_stop_path(graph_manager, start: str, waypoints: list, algorithm: str = "Dijkstra", initial_heading: float = None):
    """
    Compute an optimal multi-stop round trip: start → [waypoints in best order] → start.
    Respects initial_heading for the first segment.
    """
    all_nodes = list(dict.fromkeys([start] + waypoints))  # deduplicate, preserve order

    # Validate all nodes exist
    G = graph_manager.graph
    missing = [n for n in all_nodes if n not in G]
    if missing:
        return [], 0.0, [], True

    if len(waypoints) == 0:
        return [start], 0.0, [], True

    # Step 1: compute pairwise distances
    # For the matrix, only the distances starting from 'start' should respect initial_heading
    dist_matrix = {}
    for u in all_nodes:
        for v in all_nodes:
            if u == v:
                dist_matrix[(u, v)] = {"cost": 0.0, "path": [u]}
                continue
            
            # Apply heading only if we are starting from 'start'
            h = initial_heading if u == start else None
            path, cost = compute_shortest_path(graph_manager, u, v, algorithm, initial_heading=h)
            
            if not path:
                cost = float("inf")
            dist_matrix[(u, v)] = {"cost": cost, "path": path}

    # Check reachability
    for u in all_nodes:
        for v in all_nodes:
            if u != v and dist_matrix[(u, v)]["cost"] == float("inf"):
                return [], 0.0, [], True  # unreachable pair

    # Step 2: solve TSP
    n_waypoints = len(set(waypoints) - {start})
    is_exact = n_waypoints <= 12

    visit_order, total_cost = solve_tsp_tour(dist_matrix, all_nodes, start)

    if total_cost == float("inf"):
        return [], 0.0, [], is_exact

    # Step 3: stitch together the full path
    full_sequence = [start] + visit_order
    full_path = []

    for i in range(len(full_sequence) - 1):
        u = full_sequence[i]
        v = full_sequence[i + 1]
        sub_path = dist_matrix[(u, v)]["path"]
        if not full_path:
            full_path.extend(sub_path)
        else:
            # Avoid duplicating the junction node
            full_path.extend(sub_path[1:])

    return full_path, total_cost, visit_order, is_exact


def compute_sequential_path(graph_manager, start: str, waypoints: list, algorithm: str = "Dijkstra", initial_heading: float = None):
    """
    Compute a multi-stop path that visits waypoints in exactly the order provided.
    Respects initial_heading for the first segment.
    """
    all_sequence = [start] + waypoints
    full_path = []
    total_cost = 0.0
    
    for i in range(len(all_sequence) - 1):
        u, v = all_sequence[i], all_sequence[i+1]
        
        # Apply heading only to the first leg
        h = initial_heading if i == 0 else None
        path, cost = compute_shortest_path(graph_manager, u, v, algorithm, initial_heading=h)
        
        if not path:
            return [], 0.0, waypoints
        
        if not full_path:
            full_path.extend(path)
        else:
            full_path.extend(path[1:])
        total_cost += cost
        
    return full_path, total_cost, waypoints
