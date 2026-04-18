import json
import math
import os
import networkx as nx

class GraphManager:
    DEFAULT_SAVE_PATH = "data/graph.json"

    def __init__(self):
        self.graph: nx.Graph = nx.Graph()
        self._counter: int = 0
        self._waypoint_counter: int = 0
        self.speed_px_per_sec: float = 5.0

    # ─────────────────────────── Persistence ────────────────────────────────

    def load_from_json(self, path: str = DEFAULT_SAVE_PATH):
        """Load nodes, edges, and settings from a JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        self.graph.clear()
        self._counter = data.get("counter", 0)
        self._waypoint_counter = data.get("waypoint_counter", 0)
        self.speed_px_per_sec = data.get("speed_px_per_sec", 5.0)

        for node_id, attrs in data.get("nodes", {}).items():
            # Migration: Ensure each node has a 'uids' list
            if "uids" not in attrs:
                attrs["uids"] = [node_id]
            self.graph.add_node(node_id, **attrs)

        for edge in data.get("edges", []):
            self.graph.add_edge(
                edge["u"], edge["v"],
                weight=edge.get("weight", 1.0),
                label=edge.get("label", "")
            )

    def save_to_json(self, path: str = DEFAULT_SAVE_PATH):
        """Persist the current graph state to a JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        nodes = {n: dict(d) for n, d in self.graph.nodes(data=True)}
        edges = [
            {"u": u, "v": v, "weight": d.get("weight", 1.0), "label": d.get("label", "")}
            for u, v, d in self.graph.edges(data=True)
        ]
        payload = {
            "nodes": nodes,
            "edges": edges,
            "counter": self._counter,
            "waypoint_counter": self._waypoint_counter,
            "speed_px_per_sec": self.speed_px_per_sec,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=4)

    def _auto_save(self):
        self.save_to_json(self.DEFAULT_SAVE_PATH)

    # ──────────────────────────── Node CRUD ─────────────────────────────────

    def add_node(self, x: float, y: float, node_id: str, label: str = "", is_rfid: bool = True):
        """Add a new node (RFID or Waypoint) at the given map coordinates."""
        uids = [node_id] if is_rfid else []
        self.graph.add_node(node_id, x=float(x), y=float(y), label=label, uids=uids, is_rfid=is_rfid)
        self._auto_save()

    def edit_node(self, node_id: str, x: float, y: float, label: str):
        """Update position and label of an existing node."""
        if node_id not in self.graph:
            return
        self.graph.nodes[node_id]["x"] = float(x)
        self.graph.nodes[node_id]["y"] = float(y)
        self.graph.nodes[node_id]["label"] = label
        self._auto_save()

    def delete_node(self, node_id: str):
        """Remove a node and all its incident edges."""
        if node_id in self.graph:
            self.graph.remove_node(node_id)
            self._auto_save()

    def add_secondary_uid(self, node_id: str, new_uid: str) -> bool:
        """Associate another UID with an existing node."""
        if node_id not in self.graph:
            return False
        
        # Check if UID is already in use
        if self.is_uid_used(new_uid):
            return False
            
        uids = self.graph.nodes[node_id].get("uids", [node_id])
        if new_uid not in uids:
            uids.append(new_uid)
            self.graph.nodes[node_id]["uids"] = uids
            self._auto_save()
            return True
        return False

    def remove_secondary_uid(self, node_id: str, uid: str) -> bool:
        """Remove a secondary UID (cannot remove the primary node_id)."""
        if node_id not in self.graph or uid == node_id:
            return False
            
        uids = self.graph.nodes[node_id].get("uids", [node_id])
        if uid in uids:
            uids.remove(uid)
            self.graph.nodes[node_id]["uids"] = uids
            self._auto_save()
            return True
        return False

    def is_uid_used(self, uid: str) -> bool:
        """Check if a UID is already assigned to any node (primary or secondary)."""
        for n, d in self.graph.nodes(data=True):
            if uid == n:
                return True
            if uid in d.get("uids", []):
                return True
        return False

    def get_node_by_uid(self, uid: str) -> str | None:
        """Search for a node ID that owns the given UID."""
        for n, d in self.graph.nodes(data=True):
            if uid == n or uid in d.get("uids", []):
                return n
        return None

    def delete_virtual_node(self, node_id: str):
        """
        Remove a single virtual node (V_*) and restore the original edge
        between its two neighbours (if exactly 2 neighbours exist).
        Restored edge weight = sum of both split-edge weights.
        Virtual nodes are NOT auto-saved (transient per session).
        """
        if node_id not in self.graph or not str(node_id).startswith("V_"):
            return
        neighbours = list(self.graph.neighbors(node_id))
        if len(neighbours) == 2:
            u, v = neighbours
            w1 = self.graph[node_id][u].get("weight", 1.0)
            w2 = self.graph[node_id][v].get("weight", 1.0)
            if not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v, weight=w1 + w2, label="")
        self.graph.remove_node(node_id)

    def clear_virtual_nodes(self) -> int:
        """
        Remove ALL virtual nodes (ID starts with 'V_').
        Restores original edges where possible.
        Returns the number of nodes removed.
        """
        virtual_ids = [n for n in list(self.graph.nodes()) if str(n).startswith("V_")]
        for vid in virtual_ids:
            self.delete_virtual_node(vid)
        return len(virtual_ids)

    # ──────────────────────────── Edge CRUD ─────────────────────────────────

    def add_edge(self, u: str, v: str, weight: float = 1.0, label: str = ""):
        """Add an undirected edge between two nodes."""
        if u in self.graph and v in self.graph:
            self.graph.add_edge(u, v, weight=weight, label=label)
            self._auto_save()

    def delete_edge(self, u: str, v: str):
        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)
            self._auto_save()

    # ───────────────────────── Virtual Nodes ────────────────────────────────

    def _new_virtual_id(self) -> str:
        self._counter += 1
        return f"V_{self._counter}"

    def _new_waypoint_id(self) -> str:
        self._waypoint_counter += 1
        return f"W_{self._waypoint_counter}"

    def get_closest_edge(self, px: float, py: float, max_dist: float = 20.0):
        """
        Find the closest graph edge to point (px, py).
        Returns (edge, projected_point) or (None, None) if nothing is close enough.
        edge = (u, v); projected_point = (x, y) on the segment.
        """
        best_edge = None
        best_proj = None
        best_dist = max_dist

        for u, v in self.graph.edges():
            x0, y0 = self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]
            x1, y1 = self.graph.nodes[v]["x"], self.graph.nodes[v]["y"]
            proj, dist = self._point_to_segment(px, py, x0, y0, x1, y1)
            if dist < best_dist:
                best_dist = dist
                best_edge = (u, v)
                best_proj = proj

        return best_edge, best_proj

    @staticmethod
    def _point_to_segment(px, py, x0, y0, x1, y1):
        """Project point (px,py) onto line segment (x0,y0)-(x1,y1). Returns (point, dist)."""
        dx, dy = x1 - x0, y1 - y0
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0:
            return (x0, y0), math.hypot(px - x0, py - y0)
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / seg_len_sq))
        proj_x = x0 + t * dx
        proj_y = y0 + t * dy
        return (proj_x, proj_y), math.hypot(px - proj_x, py - proj_y)

    def create_virtual_node_on_edge(self, x: float, y: float, u: str, v: str) -> str:
        """
        Split edge (u,v) by inserting a new virtual node at (x,y).
        The original edge is removed and replaced by two new edges weighted
        proportionally by Euclidean distance.
        Returns the new virtual node ID.
        """
        vid = self._new_virtual_id()
        original_weight = self.graph[u][v].get("weight", 1.0)

        # Compute proportional weights
        x0, y0 = self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]
        x1, y1 = self.graph.nodes[v]["x"], self.graph.nodes[v]["y"]
        total_len = math.hypot(x1 - x0, y1 - y0) or 1.0
        w1 = original_weight * math.hypot(x - x0, y - y0) / total_len
        w2 = original_weight * math.hypot(x1 - x, y1 - y) / total_len

        self.graph.remove_edge(u, v)
        self.graph.add_node(vid, x=float(x), y=float(y), label="")
        self.graph.add_edge(u, vid, weight=w1, label="")
        self.graph.add_edge(vid, v, weight=w2, label="")
        # NOTE: virtual nodes are NOT auto-saved; they are transient per session.
        return vid
