import time
from enum import Enum, auto
from core.navigation import NavEngine
from hardware.rfid import RFIDReader
import hardware.motor as motor
import cv2
import numpy as np
from hardware.camera import camera_manager
from utils.traffic_sign_recognition import TrafficSignRecognition

class CarState(Enum):
    IDLE = auto()
    PLANNING = auto()
    MOVING = auto()
    TURNING = auto()
    ARRIVED = auto()

class AutonomousCar:
    def __init__(self, target_node, graph, turn_table, rfid_map, turn_config, predefined_path=None):
        self.nav = NavEngine(graph, turn_table)
        self.target_node = target_node
        self.rfid_map = rfid_map
        self.turn_config = turn_config
        self.predefined_path = predefined_path
        self.tsr = TrafficSignRecognition()
        self.prev_node = None
        self.current_node = predefined_path[0] if predefined_path else None
        self.next_node = None
        self.state = CarState.MOVING
        self._rfid = None
        self.is_active = False
        self.log_history = []          # Live console logs for the web UI
        self.last_lane = "right"       # Memory for search direction
        self.last_left_x = -1
        self.last_right_x = -1
        self.base_speed = 120
        self.target_right = 300
        self.target_left = 50
        self.scan_y_left = 0.45        # New ratio from HD
        self.scan_y_right = 0.62       # New ratio from HD
        self.scan_search_up_rows = 35  # Scanning range
        self.debug_frame = None        # Stores visualized frame for the UI

    @property
    def rfid(self):
        if self._rfid is None:
            self._rfid = RFIDReader()
        return self._rfid

    def execute(self):
        self.is_active = True
        self._log("[START] Mission started. Target: " + str(self.target_node))
        try:
            while self.is_active and self.state != CarState.ARRIVED:
                self.update_state()
                time.sleep(0.1)
            motor.stop()
            if self.state == CarState.ARRIVED:
                self._log("[ARRIVED] Đã tới đích: " + str(self.target_node))
            else:
                self._log("[STOPPED] Mission dừng sớm.")
        except Exception as e:
            motor.stop()
            self._log(f"[ERROR] {e}")

    def update_state(self):
        uid = self.rfid.read_uid_hex(timeout=0.05)
        if uid in self.rfid_map:
            detected = self.rfid_map[uid]
            if detected != self.current_node:
                self.prev_node = self.current_node
                self.current_node = detected
                self._log(f"[RFID] Phát hiện node: {detected}")
                self.state = CarState.PLANNING

        if self.state == CarState.PLANNING:
            if self.current_node == self.target_node:
                self.state = CarState.ARRIVED
            else:
                path = None
                if self.predefined_path and self.current_node in self.predefined_path:
                    try:
                        idx = self.predefined_path.index(self.current_node)
                        self.predefined_path = self.predefined_path[idx:]
                        if len(self.predefined_path) > 1:
                            path = self.predefined_path
                        else:
                            self.state = CarState.ARRIVED
                            return
                    except ValueError:
                        pass
                
                if not path:
                    path = self.nav.get_shortest_path(self.current_node, self.target_node)
                    
                if path and len(path) > 1:
                    self.next_node = path[1]
                    self._log(f"[NAV] {self.current_node} → {self.next_node}")
                    self.state = CarState.TURNING
                else:
                    self._log(f"[WARN] Không tìm thấy đường tới {self.target_node}")
                    self.state = CarState.MOVING

        elif self.state == CarState.TURNING:
            if self.prev_node and self.current_node and self.next_node:
                action = self.nav.get_action(self.prev_node, self.current_node, self.next_node)
            else:
                action = "STRAIGHT"
            self._log(f"[TURN] Hành động: {action}")
            self.execute_motor_action(action)
            self.state = CarState.MOVING

        elif self.state == CarState.MOVING:
            self.follow_lane()

    def follow_lane(self):
        raw_frame = camera_manager.get_frame()
        if raw_frame is None:
            motor.move_straight()
            return

        # Resize for CV processing (320x240 is efficient)
        frame = cv2.resize(raw_frame, (320, 240))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)

        # Draw edge visualization overlay
        self._draw_edge_overlay(frame, edges)

        left_x, y_left = self._find_left_lane(edges)
        right_x, y_right = self._find_right_lane(edges)
        
        # Improvement: Use last known lane position if current detection is missing
        if left_x != -1:
            self.last_left_x = left_x
        elif self.last_left_x != -1:
            left_x = self.last_left_x

        if right_x != -1:
            self.last_right_x = right_x
        elif self.last_right_x != -1:
            right_x = self.last_right_x

        # --- Visualization Logic (HD) ---
        display_frame = frame.copy()
        self._draw_lane_overlay(display_frame, left_x, y_left, right_x, y_right)
        
        # Metadata drawings
        cv2.line(display_frame, (0, y_left), (320, y_left), (80, 80, 80), 1)
        cv2.line(display_frame, (0, y_right), (320, y_right), (60, 60, 60), 1)
        cv2.line(display_frame, (160, 0), (160, 240), (200, 200, 200), 1)

        # Target markers
        for tx, label in [(self.target_left, "TL"), (self.target_right, "TR")]:
            cv2.line(display_frame, (tx, y_left-10), (tx, y_left+10), (0, 255, 255), 2)
            cv2.putText(display_frame, label, (tx-10, y_left-14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        if left_x != -1:
            cv2.circle(display_frame, (left_x, y_left), 6, (0, 140, 255), -1)
            cv2.putText(display_frame, f"L:{left_x}", (left_x + 6, y_left - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)
        if right_x != -1:
            cv2.circle(display_frame, (right_x, y_right), 6, (255, 80, 0), -1)
            cv2.putText(display_frame, f"R:{right_x}", (right_x - 40, y_right - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 120, 0), 1)

        steering = 0
        mode = "search"
        if right_x != -1:
            mode = "right"
            self.last_lane = "right"
            steering = -(right_x - self.target_right) * 3
        elif left_x != -1:
            mode = "left"
            self.last_lane = "left"
            steering = -(left_x - self.target_left) * 3
        
        # HUD Orientation Mark (Enhanced)
        self._draw_orientation_mark(display_frame, steering)
        
        if mode in ["right", "left"]:
            cv2.putText(display_frame, f"Mode: {mode} lane", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 80), 1)
        else:
            cv2.putText(display_frame, f"Mode: {mode}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)
            
        # Update debug frame for Streamlit
        self.debug_frame = display_frame

        # --- Traffic Sign Detection ---
        signs = self.tsr.detect_and_classify(frame)
        if signs:
            self.debug_frame = self.tsr.draw_detections(self.debug_frame, signs)
            for s in signs:
                self._log(f"[SIGN] Phát hiện: {s['label'].upper()} ({s['confidence']*100:.1f}%)")

        # Motor control
        if mode in ["right", "left"]:
            motor.drive(self.base_speed, steering)
        else:
            spin_steering = 80 if self.last_lane == "right" else -80
            motor.drive(0, spin_steering)
            time.sleep(0.05)

    def _draw_lane_overlay(self, frame, left_x, y_left, right_x, y_right):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        bot = h
        if left_x != -1 and right_x != -1:
            # Fill lane between two lines
            pts = np.array([[left_x, y_left], [right_x, y_right], [right_x, bot], [left_x, bot]], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 200, 80))
            cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
            cv2.line(frame, (left_x, y_left), (left_x, bot), (0, 140, 255), 2)
            cv2.line(frame, (right_x, y_right), (right_x, bot), (255, 80, 0), 2)
        elif left_x != -1:
            # Only left lane detected
            cv2.line(frame, (left_x, y_left), (left_x, bot), (0, 140, 255), 2)
            pts = np.array([[left_x, y_left], [w // 2, y_left], [w // 2, bot], [left_x, bot]], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 140, 255))
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
        elif right_x != -1:
            # Only right lane detected
            cv2.line(frame, (right_x, y_right), (right_x, bot), (255, 80, 0), 2)
            pts = np.array([[w // 2, y_right], [right_x, y_right], [right_x, bot], [w // 2, bot]], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (255, 80, 0))
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    def _draw_edge_overlay(self, frame, edges):
        """Draw Canny edges as colored semi-transparent overlay."""
        edge_overlay = np.zeros_like(frame)
        edge_overlay[:, :, 0] = edges  # Blue channel
        edge_overlay[:, :, 2] = edges // 2  # Red channel -> Purple tint
        cv2.addWeighted(edge_overlay, 0.15, frame, 0.85, 0, frame)

    def _draw_orientation_mark(self, frame, steering):
        """Draw enhanced steering indicator arrow HUD."""
        h, w = frame.shape[:2]
        base_y = int(h * 0.82)
        cx = w // 2
        max_steer = 200
        clamped = max(-max_steer, min(max_steer, steering))
        arrow_len = int((clamped / max_steer) * (w // 4))
        tip_x = cx + arrow_len

        ratio = abs(clamped) / max_steer
        color = (int(50 + ratio * 200), int(230 - ratio * 150), int(ratio * 255))

        bar_h = 28
        cv2.rectangle(frame, (cx - w // 4, base_y - bar_h // 2), (cx + w // 4, base_y + bar_h // 2), (30, 30, 30), -1)
        cv2.rectangle(frame, (cx, base_y - bar_h // 4), (tip_x, base_y + bar_h // 4), color, -1)
        if arrow_len != 0:
            cv2.arrowedLine(frame, (cx, base_y), (tip_x, base_y), color, 2, tipLength=0.35)
        cv2.line(frame, (cx, base_y - bar_h // 2), (cx, base_y + bar_h // 2), (200, 200, 200), 1)
        label = f"Steer: {int(steering):+d}"
        cv2.putText(frame, label, (cx - w // 4, base_y - bar_h // 2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

    def _find_left_lane(self, edges):
        height, width = edges.shape
        base_y = int(height * self.scan_y_left)
        for offset in range(self.scan_search_up_rows + 1):
            y = base_y - offset
            if y < 0: break
            line = edges[y]
            mid = len(line) // 2
            left_half = line[:mid]
            if left_half.sum() != 0:
                inner_x = mid - 1 - np.argmax(left_half[::-1])
                if inner_x > 0:
                    return inner_x, y
        return -1, base_y

    def _find_right_lane(self, edges):
        height, width = edges.shape
        base_y = int(height * self.scan_y_right)
        for offset in range(self.scan_search_up_rows + 1):
            y = base_y - offset
            if y < 0: break
            line = edges[y]
            mid = len(line) // 2
            right_half = line[mid:]
            if right_half.sum() != 0:
                inner_x = mid + np.argmax(right_half)
                if inner_x < width - 1:
                    return inner_x, y
        return -1, base_y

    def execute_motor_action(self, action):
        if action == "RIGHT":
            motor.turn_right()
            time.sleep(self.turn_config["90_DEG"])
        elif action == "LEFT":
            motor.turn_left()
            time.sleep(self.turn_config["90_DEG"])
        elif action =="TURN_AROUND":
            motor.turn_right()
            time.sleep(self.turn_config["90_DEG"])
            motor.turn_right()
            time.sleep(self.turn_config["90_DEG"])
        
        motor.stop()
        time.sleep(0.1)

    def stop_system(self):
        self.is_active = False
        motor.stop()
        self._log("[STOP] Dừng khẩn cấp!")

    def _log(self, msg: str):
        """Append a timestamped log line to log_history."""
        entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.log_history.append(entry)
        if len(self.log_history) > 200:
            self.log_history = self.log_history[-200:]