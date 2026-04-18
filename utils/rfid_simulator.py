"""
rfid_simulator.py
Simulates RFID tag scanning along a pre-planned route for UI preview / testing
when physical RFID hardware is not available.
"""

import threading
import time


class RFIDSimulator:
    """
    Lightweight in-process simulator that tracks which node on a route
    the 'vehicle' is currently at.

    Usage
    -----
    sim = RFIDSimulator()
    sim.start_route(["A", "B", "C"])   # begin simulated travel
    sim.get_current_node()             # returns current node ID or None
    sim.force_scan("B")               # teleport the vehicle to a specific node
    sim.reset()                       # clear everything
    """

    def __init__(self, step_delay: float = 2.0):
        self._route: list = []
        self._idx: int = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.step_delay = step_delay      # seconds between automatic node advances
        self.is_running: bool = False

    # ─────────────────────────── Public API ─────────────────────────────────

    def start_route(self, path: list):
        """Begin simulated travel along *path*. Advances one node every step_delay sec."""
        self.reset()
        if not path:
            return
        with self._lock:
            self._route = list(path)
            self._idx = 0
            self.is_running = True

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def reset(self):
        """Stop simulation and clear state."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with self._lock:
            self._route = []
            self._idx = 0
            self.is_running = False

    def get_current_node(self):
        """Return the node ID the vehicle is currently at, or None."""
        with self._lock:
            if not self._route:
                return None
            return self._route[self._idx]

    def force_scan(self, node_id: str) -> bool:
        """
        Manually set the vehicle position to *node_id* if it exists in the
        current route. Returns True on success, False otherwise.
        """
        with self._lock:
            if node_id in self._route:
                self._idx = self._route.index(node_id)
                return True
            return False

    def get_progress(self):
        """Return (current_index, total_nodes) for progress display."""
        with self._lock:
            return self._idx, len(self._route)

    # ──────────────────────────── Internal ──────────────────────────────────

    def _run(self):
        while not self._stop_event.is_set():
            time.sleep(self.step_delay)
            with self._lock:
                if self._idx < len(self._route) - 1:
                    self._idx += 1
                else:
                    self.is_running = False
                    break
