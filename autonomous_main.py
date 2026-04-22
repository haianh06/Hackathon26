import time
import threading
from enum import Enum, auto
from collections import deque
from core.navigation import NavEngine
from hardware.rfid import RFIDReader
import hardware.motor as motor
import cv2
import numpy as np
from hardware.camera import camera_manager
import math
import os
import shutil

# ============================================================
# Configuration — Lane Following (Standalone)
# ============================================================
BASE_SPEED          = 120
TARGET_RIGHT        = 300
TARGET_LEFT         = 20
SCAN_Y_RIGHT        = 0.65     # Fraction of frame height to scan at
SCAN_SEARCH_UP_ROWS = 35       # How many rows upward to search for lane edge
STEERING_GAIN       = 3        # Proportional gain



# ============================================================
# Helper Functions (Standalone)
# ============================================================

def _find_right_lane(edges, last_x=-1):
    h, w = edges.shape
    target_y = int(h * SCAN_Y_RIGHT)
    
    # 1. Tìm điểm gốc ở đáy ảnh (ưu tiên mép TRONG - gần tâm nhất)
    # Quét từ 45% chiều rộng (lấn sang trái một chút) đến 90%
    start_search = int(w * 0.45)
    end_search = int(w * 0.9)
    bottom_roi = edges[int(h*0.8):, start_search : end_search]
    
    if bottom_roi.sum() == 0:
        fallback = edges[target_y, start_search : end_search]
        if fallback.sum() == 0: return -1, target_y
        # Lấy điểm đầu tiên bên trái (mép trong)
        first_pixel = np.where(fallback > 0)[0][0]
        return start_search + first_pixel, target_y

    histogram = np.sum(bottom_roi, axis=0)
    # Bắt mọi điểm có vạch (giúp không bỏ sót nét đứt mờ ở mép trong)
    peaks = np.where(histogram > 0)[0]
    
    if len(peaks) > 0:
        current_x = start_search + peaks[0] # Chọn peak ĐẦU TIÊN từ trái sang (mép trong)
    else:
        return -1, target_y
    
    # 2. Trượt cửa sổ lên trên bám theo mép trong
    n_windows = 5
    win_h = int(h * 0.12)
    margin = 40 
    
    lane_points = []
    for i in range(n_windows):
        y_low = h - (i+1) * win_h
        y_high = h - i * win_h
        if y_low < 0: break
        
        x_left = max(0, current_x - margin)
        x_right = min(w, current_x + margin)
        
        win_slice = edges[y_low:y_high, x_left:x_right]
        if win_slice.sum() > 0:
            win_hist = np.sum(win_slice, axis=0)
            win_peaks = np.where(win_hist > 0)[0]
            if len(win_peaks) > 0:
                current_x = x_left + win_peaks[0]
                lane_points.append((current_x, (y_low + y_high) // 2))

    if not lane_points:
        return -1, target_y
        
    best_point = min(lane_points, key=lambda p: abs(p[1] - target_y))
    return best_point[0], target_y

def _find_left_lane(edges, last_x=-1):
    h, w = edges.shape
    target_y = int(h * SCAN_Y_RIGHT)
    
    # Quét từ 10% đến 55% (lấn sang phải một chút)
    start_search = int(w * 0.1)
    end_search = int(w * 0.55)
    bottom_roi = edges[int(h*0.8):, start_search : end_search]
    
    if bottom_roi.sum() == 0:
        fallback = edges[target_y, start_search : end_search]
        if fallback.sum() == 0: return -1, target_y
        # Lấy điểm đầu tiên bên PHẢI (mép trong của vạch trái)
        last_pixel = np.where(fallback > 0)[0][-1]
        return start_search + last_pixel, target_y

    histogram = np.sum(bottom_roi, axis=0)
    peaks = np.where(histogram > 0)[0]
    
    if len(peaks) > 0:
        current_x = start_search + peaks[-1] # Chọn peak CUỐI CÙNG từ trái sang (mép trong)
    else:
        return -1, target_y
    
    n_windows = 5
    win_h = int(h * 0.12)
    margin = 40 
    
    lane_points = []
    for i in range(n_windows):
        y_low = h - (i+1) * win_h
        y_high = h - i * win_h
        if y_low < 0: break
        
        x_left = max(0, current_x - margin)
        x_right = min(w, current_x + margin)
        
        win_slice = edges[y_low:y_high, x_left:x_right]
        if win_slice.sum() > 0:
            win_hist = np.sum(win_slice, axis=0)
            win_peaks = np.where(win_hist > 0)[0]
            if len(win_peaks) > 0:
                current_x = x_left + win_peaks[-1] # Mép trong
                lane_points.append((current_x, (y_low + y_high) // 2))

    if not lane_points:
        return -1, target_y
        
    best_point = min(lane_points, key=lambda p: abs(p[1] - target_y))
    return best_point[0], target_y

def _draw_lane_overlay(frame, lane_x, lane_y, active_lane):
    h, w = frame.shape[:2]
    if lane_x != -1:
        color = (255, 80, 0) if active_lane == "right" else (0, 255, 80)
        cv2.line(frame, (lane_x, lane_y), (lane_x, h), color, 2)
        overlay = frame.copy()
        pts = np.array([[w // 2, lane_y], [lane_x, lane_y], [lane_x, h], [w // 2, h]], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

def _draw_edge_overlay(frame, edges):
    edge_overlay = np.zeros_like(frame)
    edge_overlay[:, :, 0] = edges; edge_overlay[:, :, 2] = edges // 2
    cv2.addWeighted(edge_overlay, 0.15, frame, 0.85, 0, frame)

def _draw_orientation_mark(frame, steering):
    h, w = frame.shape[:2]; base_y = int(h * 0.82); cx = w // 2; max_steer = 200
    clamped = max(-max_steer, min(max_steer, steering))
    arrow_len = int((clamped / max_steer) * (w // 4)); tip_x = cx + arrow_len
    ratio = abs(clamped) / max_steer
    color = (int(50 + ratio * 200), int(230 - ratio * 150), int(ratio * 255))
    bar_h = 28
    cv2.rectangle(frame, (cx - w // 4, base_y - bar_h // 2), (cx + w // 4, base_y + bar_h // 2), (30, 30, 30), -1)
    cv2.rectangle(frame, (cx, base_y - bar_h // 4), (tip_x, base_y + bar_h // 4), color, -1)
    if arrow_len != 0: cv2.arrowedLine(frame, (cx, base_y), (tip_x, base_y), color, 2, tipLength=0.35)
    cv2.line(frame, (cx, base_y - bar_h // 2), (cx, base_y + bar_h // 2), (200, 200, 200), 1)
    cv2.putText(frame, f"Steer: {int(steering):+d}", (cx - w // 4, base_y - bar_h // 2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

def follow_lane_frame(frame, last_steering, pos_history, max_history=5):
    img_small = cv2.resize(frame, (320, 240))
    
    # Preprocessing
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 70, 160)
    edges = cv2.bitwise_and(edges, white_mask)
    
    _draw_edge_overlay(img_small, edges)
    
    right_x_raw, y_lane = _find_right_lane(edges, -1)
    
    # 1. Quản lý trạng thái làn và Lọc nhiễu vị trí
    if right_x_raw != -1:
        active_lane = "right"
        if len(pos_history) > 0 and pos_history[0] < 160: # Chuyển từ trái sang phải -> Xóa lịch sử cũ
            pos_history.clear()
        pos_history.append(right_x_raw)
        if len(pos_history) > max_history: pos_history.pop(0)
        lane_x = int(np.mean(pos_history))
        memory_used = False
    else:
        # Nếu mất làn phải, thử tìm làn trái
        left_x_raw, y_lane = _find_left_lane(edges, -1)
        if left_x_raw != -1:
            active_lane = "left"
            if len(pos_history) > 0 and pos_history[0] >= 160: # Chuyển từ phải sang trái -> Xóa lịch sử cũ
                pos_history.clear()
            pos_history.append(left_x_raw)
            if len(pos_history) > max_history: pos_history.pop(0)
            lane_x = int(np.mean(pos_history))
            memory_used = False
        else:
            # Mất cả 2 làn, dùng lại giá trị cũ
            lane_x = pos_history[-1] if pos_history else -1
            active_lane = "right" if lane_x >= 160 else "left"
            memory_used = True
    
    steering = 0
    mode = "search"
    if lane_x != -1:
        mode = "follow"
        # 2. Tính toán góc lái theo làn đang bám
        if active_lane == "right":
            target_error = lane_x - TARGET_RIGHT
        else:
            target_error = lane_x - TARGET_LEFT
        steering = -target_error * 2.5 
        
    # 3. Bộ lọc thông thấp cho góc lái
    if last_steering is not None:
        steering = 0.8 * last_steering + 0.2 * steering
        
    display = img_small.copy()
    _draw_lane_overlay(display, lane_x if lane_x != -1 else -1, y_lane, active_lane)
    
    if active_lane == "right":
        cv2.line(display, (TARGET_RIGHT, y_lane - 10), (TARGET_RIGHT, y_lane + 10), (0, 255, 255), 2)
    else:
        cv2.line(display, (TARGET_LEFT, y_lane - 10), (TARGET_LEFT, y_lane + 10), (0, 255, 255), 2)
        
    _draw_orientation_mark(display, steering)
    
    return lane_x, y_lane, steering, memory_used, mode, display, edges

# ============================================================
# Main Classes
# ============================================================
class CarState(Enum):
    IDLE = auto()
    PLANNING = auto()
    MOVING = auto()
    TURNING = auto()
    ARRIVED = auto()

class AutonomousCar:
    def __init__(self, target_node, graph, turn_table, rfid_map, turn_config, predefined_path=None, initial_heading=0, speed_px_per_sec=65.0, lane_mode="basic", perspective_src=None):
        self.nav = NavEngine(graph, turn_table)
        self.target_node = target_node
        self.rfid_map = rfid_map
        self.turn_config = turn_config
        self.predefined_path = predefined_path
        self.initial_heading = initial_heading
        self.speed_px_per_sec = speed_px_per_sec

        self.prev_node = None
        self.current_node = predefined_path[0] if predefined_path else None
        self.next_node = None
        self.state = CarState.PLANNING
        self._rfid = None
        self.is_active = False
        self.log_history = []
        self.pos_history = []
        self.last_steering = 0
        self.base_speed = 120
        self.debug_frame = None
        self.blind_run_end_time = None
        self.blind_run_target_node = None
        self._last_blind_log = 0

        # RFID Threading state
        self._latest_uid = None
        self._rfid_lock = threading.Lock()
        self._rfid_thread = None
        self._rfid_lock = threading.Lock()
        self._rfid_thread = None

        try:
            from core.detector import SignDetector
            from core.classifier import SignClassifier
            self.sign_detector = SignDetector()
            self.sign_classifier = SignClassifier(templates_dir='templates')
            self.has_sign_detection = True
            print("[INIT] Sign detection module loaded (Asynchronous).")
        except Exception as e:
            print(f"[INIT] Sign detection error: {e}")
            self.has_sign_detection = False
            
        self.last_sign_detect_time = 0.0
        self.sign_detect_interval = 0.1 # High frequency background scan
        self.last_detected_signs = []
        self._sign_lock = threading.Lock()
        self._sign_thread = None
        
        # Temporal Buffer: {sign_type: hit_count}
        self.sign_counts = {'straight': 0, 'left': 0, 'right': 0, 'parking': 0}
        self.CONFIRM_THRESHOLD = 3 # Consecutive hits to confirm
        
        self.snapshot_dir = "data/sign_snapshots"
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.sign_snapshots = []
        self.last_saved_sign_time = {}

    @property
    def rfid(self):
        if self._rfid is None:
            self._rfid = RFIDReader()
        return self._rfid

    def _rfid_background_loop(self):
        """Dedicated high-frequency RFID scanning loop running in a background thread."""
        self._log("[RFID] Background reader thread started.")
        # Ensure rfid is initialized
        _ = self.rfid
        
        while self.is_active:
            try:
                # Polling for UID with a reasonable timeout
                uid = self.rfid.read_uid_hex(timeout=0.2)
                if uid:
                    with self._rfid_lock:
                        # Store the latest UID. Main thread will consume it.
                        self._latest_uid = uid
            except Exception as e:
                print(f"RFID Thread Exception: {e}")
            
            # Tiny sleep to yield control
            time.sleep(0.01)
        self._log("[RFID] Background reader thread stopped.")

    def _sign_detection_loop(self):
        """Background thread for sign detection to keep main loop at 10 FPS."""
        self._log("[VISION] Sign detection thread started.")
        while self.is_active:
            if not self.has_sign_detection:
                break
                
            raw_frame = camera_manager.get_frame()
            if raw_frame is not None:
                # Convert to RGB once for detector
                rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                
                # Detect signs (uses the optimized SignDetector)
                detected = self.sign_detector.detect_signs(rgb_frame, use_roi=True)
                
                current_results = []
                frame_hits = set()
                if detected:
                    for roi, bbox in detected:
                        sign_type, conf = self.sign_classifier.classify(roi)
                        # Lower threshold for single frame (0.4) because we have the temporal buffer
                        if conf >= 0.4:
                            frame_hits.add(sign_type)
                            self.sign_counts[sign_type] = min(5, self.sign_counts.get(sign_type, 0) + 1)
                            
                            # Log every detection for debugging
                            if conf >= 0.5:
                                self._log(f"[VISION-DEBUG] Seen {sign_type.upper()} ({conf*100:.1f}%) count={self.sign_counts[sign_type]}")
                            
                            # Only confirm if we have enough hits
                            if self.sign_counts[sign_type] >= self.CONFIRM_THRESHOLD:
                                current_results.append((sign_type, conf, bbox))
                                
                                # Snapshot saving logic (safe to do in background)
                                now = time.time()
                                if now - self.last_saved_sign_time.get(sign_type, 0) > 4.0:
                                    self.last_saved_sign_time[sign_type] = now
                                    filename = f"{sign_type}_{int(now)}.jpg"
                                    filepath = os.path.join(self.snapshot_dir, filename)
                                    
                                    snap_frame = raw_frame.copy()
                                    rx, ry, rw, rh = bbox
                                    cv2.rectangle(snap_frame, (rx, ry), (rx+rw, ry+rh), (0,255,0), 2)
                                    cv2.putText(snap_frame, f"{sign_type} CONFIRMED", (rx, max(25, ry-10)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                                    cv2.imwrite(filepath, snap_frame)
                                    self.sign_snapshots.append(filepath)
                                    if len(self.sign_snapshots) > 10: self.sign_snapshots.pop(0)

                # Decay counts for signs NOT seen in this frame
                for st in self.sign_counts:
                    if st not in frame_hits:
                        self.sign_counts[st] = max(0, self.sign_counts[st] - 1)

                # Update shared state
                with self._sign_lock:
                    self.last_detected_signs = current_results
            
            # Yield control - no need to thrash CPU if no new frame
            time.sleep(self.sign_detect_interval)
            
        self._log("[VISION] Sign detection thread stopped.")

    def execute(self):
        self.is_active = True
        self._log("[START] Mission started. Target: " + str(self.target_node))
        
        # Start RFID background thread
        self._rfid_thread = threading.Thread(target=self._rfid_background_loop, daemon=True)
        self._rfid_thread.start()
        
        # Start Sign Detection background thread
        if self.has_sign_detection:
            self._sign_thread = threading.Thread(target=self._sign_detection_loop, daemon=True)
            self._sign_thread.start()
        
        try:
            while self.is_active and self.state != CarState.ARRIVED:
                self.update_state()
                time.sleep(0.01)
            motor.stop()
            self.is_active = False
            if self.state == CarState.ARRIVED:
                self._log("[ARRIVED] Đã tới đích: " + str(self.target_node))
            else:
                self._log("[STOPPED] Mission dừng sớm.")
        except Exception as e:
            self._log(f"[ERROR] {e}")
            print(f"AutonomousCar Thread Error: {e}")
        finally:
            self.is_active = False
            try: motor.stop()
            except: pass
            self._log("[STOP] Mission terminated and resources released.")

    def update_state(self):
        # Consume UID from background thread
        uid = None
        with self._rfid_lock:
            if self._latest_uid:
                uid = self._latest_uid
                self._latest_uid = None
        
        if uid:
            print(f"📡 [SCANNER] Found UID: {uid}")
            self._log(f"[RFID-SCAN] UID scanned: {uid}")
        if uid in self.rfid_map:
            detected = self.rfid_map[uid]
            if detected != self.current_node:
                if self.predefined_path and detected in self.predefined_path:
                    try:
                        idx = self.predefined_path.index(detected)
                        self.prev_node = self.predefined_path[idx-1] if idx > 0 else self.current_node
                    except ValueError:
                        self.prev_node = self.current_node
                else:
                    self.prev_node = self.current_node
                self.current_node = detected
                self._log(f"[RFID] Node detected: {self._get_node_label(detected)}")
                self.state = CarState.PLANNING

        if self.state == CarState.PLANNING:
            # --- 1. Sincronize predefined path if available ---
            if self.predefined_path and self.current_node in self.predefined_path:
                try:
                    idx = self.predefined_path.index(self.current_node)
                    self.predefined_path = self.predefined_path[idx:]
                except ValueError:
                    pass

            # --- 2. Check for Arrival ---
            if self.current_node == self.target_node:
                # If we have a loop path (Start == End), only arrive if we've traversed the intermediate points.
                if not self.predefined_path or len(self.predefined_path) <= 1:
                    self.state = CarState.ARRIVED
                    motor.stop()
                    self.is_active = False
                    return

            # --- 3. Plan next move ---
            path = None
            if self.predefined_path and len(self.predefined_path) > 1:
                path = self.predefined_path
            
            if not path:
                path = self.nav.get_shortest_path(self.current_node, self.target_node)

            if path and len(path) > 1:
                self.next_node = path[1]
                self._log(f"[NAV] {self._get_node_label(self.current_node)} → {self._get_node_label(self.next_node)}")
                remaining_path = path[1:]
                if all(str(node).startswith("V_") or str(node).startswith("W_") for node in remaining_path):
                    try:
                        t_run = 0.0
                        dist0 = self._get_dist(self.current_node, remaining_path[0])
                        t_run += dist0 / self.speed_px_per_sec
                        if self.prev_node and self.prev_node in self.nav.nodes:
                            dx_h = self.nav.nodes[self.current_node]['x'] - self.nav.nodes[self.prev_node]['x']
                            dy_h = self.nav.nodes[self.current_node]['y'] - self.nav.nodes[self.prev_node]['y']
                            current_heading = math.degrees(math.atan2(dy_h, dx_h))
                            initial_action = self.nav.get_initial_action(self.current_node, remaining_path[0], current_heading)
                        else:
                            initial_action = self.nav.get_initial_action(self.current_node, remaining_path[0], self.initial_heading)
                        if initial_action != "STRAIGHT": t_run += self.turn_config.get("90_DEG", 1.2)
                        for i in range(len(remaining_path) - 1):
                            u, v = remaining_path[i], remaining_path[i+1]
                            dist = self._get_dist(u, v)
                            t_run += dist / self.speed_px_per_sec
                            prev = remaining_path[i-1] if i > 0 else self.current_node
                            action = self.nav.get_action(prev, u, v)
                            if action != "STRAIGHT": t_run += self.turn_config.get("90_DEG", 1.2)
                        self.blind_run_end_time = time.time() + t_run
                        self.blind_run_target_node = remaining_path[-1]
                        self._log(f"[BLIND-RUN] Đích cuối ({self._get_node_label(self.blind_run_target_node)}) là Waypoint. Tổng thời gian dự kiến: {t_run:.2f}s")
                    except Exception as e: self._log(f"[BLIND-RUN ERROR] {e}")
                self.state = CarState.TURNING
            else:
                self._log(f"[WARN] Không tìm thấy đường tới {self.target_node}")
                self.state = CarState.MOVING

        elif self.state == CarState.TURNING:
            if self.prev_node is None and self.current_node and self.next_node:
                action = self.nav.get_initial_action(self.current_node, self.next_node, self.initial_heading)
                self._log(f"[INIT-TURN] Hướng: {self.initial_heading}° | Hành động: {action}")
            elif self.prev_node and self.current_node and self.next_node:
                action = self.nav.get_action(self.prev_node, self.current_node, self.next_node)
            else: action = "STRAIGHT"
            self._log(f"[TURN] Hành động: {action}")
            self.execute_motor_action(action)
            self.state = CarState.MOVING

        elif self.state == CarState.MOVING:
            if self.blind_run_end_time is not None:
                rem = self.blind_run_end_time - time.time()
                if rem <= 0:
                    self._log(f"[BLIND-RUN] Đã đến mục tiêu cuối ({self._get_node_label(self.blind_run_target_node)}).")
                    self.current_node = self.blind_run_target_node
                    self.state = CarState.PLANNING
                    self.blind_run_end_time = None
                    self.blind_run_target_node = None
                else:
                    if not hasattr(self, '_last_blind_log') or time.time() - self._last_blind_log >= 0.5:
                        self._log(f"[BLIND-RUN] Estimated arrival in {rem:.1f}s")
                        self._last_blind_log = time.time()
                    self.follow_lane()
            else: self.follow_lane()

    def follow_lane(self):
        raw_frame = camera_manager.get_frame()
        if raw_frame is None:
            motor.move_straight()
            return

        # Gọi thuật toán standalone cải tiến
        right_x, y_right, steering, memory_used, mode, display_frame, edges = \
            follow_lane_frame(raw_frame, self.last_steering, self.pos_history)

        self.last_steering = steering

        # --- Traffic Sign Visualization (Asynchronous results) ---
        # No more detection here, just reading results from the background thread
        with self._sign_lock:
            current_signs = list(self.last_detected_signs)
            
        # Draw HUD
        h, w = display_frame.shape[:2]
        state_color = (0, 255, 0) if "MOVING" in self.state.name or "FOLLOW" in self.state.name else (0, 0, 255)
        cv2.putText(display_frame, f"STATE: {self.state.name}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)
        
        raw_h, raw_w = raw_frame.shape[:2]
        h_ratio, w_ratio = 240 / raw_h, 320 / raw_w
        for sign_type, conf, (rx, ry, rw, rh) in current_signs:
            # Adjust coordinates if ROI window was used, but detect_signs returns relative to top-left of image passed
            # In our case we passed the whole rgb_frame to detect_signs (which 내부적으로 crops)
            # So rx, ry are relative to 640x480.
            x, y = int(rx * w_ratio), int(ry * h_ratio)
            sw, sh = int(rw * w_ratio), int(rh * h_ratio)
            cv2.rectangle(display_frame, (x, y), (x + sw, y + sh), (0, 255, 255), 2)
            cv2.putText(display_frame, f"{sign_type} {int(conf*100)}%", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        self.debug_frame = display_frame
        self.edges_frame = edges

        # Motor control
        if mode != "search": motor.drive(self.base_speed, steering)
        else:
            motor.drive(100, 0)
            time.sleep(0.05)

    def execute_motor_action(self, action):
        t_straight = self.turn_config.get("STRAIGHT", 1.0)
        t_90 = self.turn_config.get("90_DEG", 1.2)
        t_180 = self.turn_config.get("180_DEG", 2.4)

        if action == "RIGHT":
            motor.move_straight(); time.sleep(t_straight)
            motor.turn_right(); time.sleep(t_90)
        elif action == "LEFT":
            motor.move_straight(); time.sleep(t_straight)
            motor.turn_left(); time.sleep(t_90)
        elif action =="TURN_AROUND":
            motor.turn_right(); time.sleep(t_180)
        motor.stop(); time.sleep(0.1)

    def _get_dist(self, u, v):
        try:
            n1, n2 = self.nav.nodes[u], self.nav.nodes[v]
            return math.hypot(n2['x'] - n1['x'], n2['y'] - n1['y'])
        except KeyError: return 0.0

    def _get_node_label(self, node_id):
        if not node_id: return "None"
        if node_id in self.nav.graph.get('nodes', {}):
            lbl = self.nav.graph['nodes'][node_id].get('label', '')
            return f"{lbl} [{node_id}]" if lbl else f"[{node_id}]"
        return f"[{node_id}]"

    def stop_system(self):
        """Safely stop all background threads and release resources."""
        self.is_active = False
        try:
            motor.stop()
        except:
            pass
            
        # Join RFID thread
        if self._rfid_thread and self._rfid_thread.is_alive():
            self._rfid_thread.join(timeout=0.5)
            
        # Join Sign Detection thread
        if hasattr(self, '_sign_thread') and self._sign_thread and self._sign_thread.is_alive():
            self._sign_thread.join(timeout=0.5)
            
        self._log("[STOP] Hệ thống đã dừng hoàn toàn.")

    def _log(self, msg: str):
        entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.log_history.append(entry)
        if len(self.log_history) > 200: self.log_history = self.log_history[-200:]