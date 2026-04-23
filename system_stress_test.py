import time
import json
import os
import signal
import sys
import threading
import cv2
from hardware.camera import camera_manager
from graph.graph_manager import GraphManager
from autonomous_main import AutonomousCar
import hardware.motor as motor

def _load_turn_config() -> dict:
    defaults = {"90_DEG_RIGHT": 1.3, "90_DEG_LEFT": 2.0, "180_DEG": 3.6, "STRAIGHT": 2.0}
    config_path = "data/turn_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                data = json.load(f)
                return {**defaults, **data}
        except Exception as e:
            print(f"Error loading turn_config.json: {e}")
    return defaults

def main():
    print("=== SYSTEM STRESS TEST ===")
    print("Initializing camera...")
    camera_manager.start()

    print("Loading graph data...")
    gm = GraphManager()
    if os.path.exists("data/graph.json"):
        gm.load_from_json("data/graph.json")

    # Build graph adjacency matrix and RFID map
    graph_adj = {
        'nodes': {n: {'x': gm.graph.nodes[n]['x'], 'y': gm.graph.nodes[n]['y']} for n in gm.graph.nodes()},
        'edges': {}
    }
    for u in gm.graph.nodes():
        graph_adj['edges'][u] = {}
        for v in gm.graph.neighbors(u):
            graph_adj['edges'][u][v] = gm.graph[u][v].get('weight', 1)

    rfid_map = {}
    for n, d in gm.graph.nodes(data=True):
        node_uids = d.get('uids', [n])
        for u in node_uids:
            rfid_map[u] = n

    turn_config = _load_turn_config()
    speed_px_per_sec = float(gm.speed_px_per_sec) if hasattr(gm, 'speed_px_per_sec') else 65.0

    print("Initializing AutonomousCar...")
    # The target node is deliberately set to an unreachable string
    # so the car continuously falls back to the MOVING state (lane following)
    # and keeps detecting signs and running servos indefinitely.
    car = AutonomousCar(
        target_node="DUMMY_STRESS_TEST_NODE_FOREVER",
        graph=graph_adj,
        turn_table={},
        rfid_map=rfid_map,
        turn_config=turn_config,
        predefined_path=[],
        initial_heading=0,
        speed_px_per_sec=speed_px_per_sec,
        lane_mode="sliding_window",
        perspective_src=[[120, 160], [200, 160], [300, 230], [20, 230]]
    )

    def signal_handler(sig, frame):
        print("\n[STRESS TEST] Stopping system...")
        car.stop_system()
        camera_manager.stop()
        cv2.destroyAllWindows()
        sys.exit(0)

    # Catch Ctrl+C to safely stop the motors and background threads
    signal.signal(signal.SIGINT, signal_handler)

    print("Starting execution loop in background...")
    car_thread = threading.Thread(target=car.execute, daemon=True)
    car_thread.start()

    print("Opening camera window... Press 'q' on the image window or Ctrl+C to stop.")
    try:
        while True:
            if hasattr(car, 'debug_frame') and car.debug_frame is not None:
                cv2.imshow("Autonomous Cam", car.debug_frame)
            
            # Wait for 20ms and check if 'q' is pressed
            if cv2.waitKey(20) & 0xFF == ord('q'):
                print("\n[STRESS TEST] 'q' pressed. Stopping...")
                break
    except KeyboardInterrupt:
        pass
    finally:
        car.stop_system()
        camera_manager.stop()
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == "__main__":
    main()
