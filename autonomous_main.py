import time
from enum import Enum, auto
from core.navigation import NavEngine
from hardware.rfid import RFIDReader
import hardware.motor as motor
import cv2
import numpy as np
from hardware.camera import camera_manager
import math

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
        self.log_history = []          # Live console logs for the web UI
        self.last_right_x = -1
        self.base_speed = 120
        self.target_right = 300
        self.scan_y_right = 0.6       # New ratio from HD
        self.scan_search_up_rows = 35  # Scanning range
        self.debug_frame = None        # Stores visualized frame for the UI
        self.blind_run_end_time = None # Timer value for Waypoints without RFID
        self.blind_run_target_node = None # The destination of the timer
        self._last_blind_log = 0       # Throttling log

        # Lane Detection Algorithm Config
        self.detection_mode = lane_mode  # "basic", "sliding_window"
        
        # Perspective Transform Points (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        # Defaults for 320x240 frame if not provided
        if perspective_src is not None:
            self.perspective_src = np.float32(perspective_src)
        else:
            self.perspective_src = np.float32([
                [120, 160], [200, 160], 
                [300, 230], [20, 230]
            ])
        
        self.perspective_dst = np.float32([
            [80, 0], [240, 0], 
            [240, 240], [80, 240]
        ])

        # Traffic Sign Detection Integration
        try:
            from core.detector import SignDetector
            from core.classifier import SignClassifier
            self.sign_detector = SignDetector()
            self.sign_classifier = SignClassifier(templates_dir='templates')
            self.has_sign_detection = True
            # self._log will be used when actived, we can just print here
            print("[INIT] Traffic Sign Detection module loaded successfully.")
        except Exception as e:
            print(f"Failed to initialize sign detection: {e}")
            self.has_sign_detection = False
            
        self.last_sign_detect_time = 0.0
        self.sign_detect_interval = 0.5  # Run detection every 0.5 seconds
        self.last_detected_signs = []    # Cache to display on UI between detections

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
            try:
                motor.stop()
            except:
                pass
            self._log("[STOP] Mission terminated and resources released.")

    def update_state(self):
        uid = self.rfid.read_uid_hex(timeout=0.05)
        if uid in self.rfid_map:
            detected = self.rfid_map[uid]
            if detected != self.current_node:
                # Optimized prev_node logic: Use actual predecessor in path for accurate turn geometry
                if self.predefined_path and detected in self.predefined_path:
                    try:
                        idx = self.predefined_path.index(detected)
                        if idx > 0:
                            self.prev_node = self.predefined_path[idx-1]
                        else:
                            self.prev_node = self.current_node
                    except ValueError:
                        self.prev_node = self.current_node
                else:
                    self.prev_node = self.current_node
                
                self.current_node = detected
                self._log(f"[RFID] Phát hiện node: {self._get_node_label(detected)}")
                self.state = CarState.PLANNING

        if self.state == CarState.PLANNING:
            if self.current_node == self.target_node:
                self.state = CarState.ARRIVED
                motor.stop()
                self.is_active = False
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
                            motor.stop()
                            self.is_active = False
                            return
                    except ValueError:
                        pass
                
                if not path:
                    path = self.nav.get_shortest_path(self.current_node, self.target_node)
                    
                if path and len(path) > 1:
                    self.next_node = path[1]
                    self._log(f"[NAV] {self._get_node_label(self.current_node)} → {self._get_node_label(self.next_node)}")
                    
                    # New: Logic for Waypoint-only final segments (Blind Run)
                    # Support both Virtual nodes (V_) and Manual Waypoints (W_)
                    remaining_path = path[1:]
                    if all(str(node).startswith("V_") or str(node).startswith("W_") for node in remaining_path):
                        try:
                            t_run = 0.0
                            # Leg 1: from current (RFID) to first waypoint
                            dist0 = self._get_dist(self.current_node, remaining_path[0])
                            t_run += dist0 / self.speed_px_per_sec
                            
                            # Add initial turn penalty if any
                            # Use vector from prev_node to current_node to find current heading
                            if self.prev_node and self.prev_node in self.nav.nodes:
                                dx_h = self.nav.nodes[self.current_node]['x'] - self.nav.nodes[self.prev_node]['x']
                                dy_h = self.nav.nodes[self.current_node]['y'] - self.nav.nodes[self.prev_node]['y']
                                current_heading = math.degrees(math.atan2(dy_h, dx_h))
                                initial_action = self.nav.get_initial_action(self.current_node, remaining_path[0], current_heading)
                            else:
                                initial_action = self.nav.get_initial_action(self.current_node, remaining_path[0], self.initial_heading)
                            
                            if initial_action != "STRAIGHT":
                                t_run += self.turn_config.get("90_DEG", 1.2)

                            # Subsequent legs
                            for i in range(len(remaining_path) - 1):
                                u, v = remaining_path[i], remaining_path[i+1]
                                dist = self._get_dist(u, v)
                                t_run += dist / self.speed_px_per_sec
                                
                                # Check for turns at intermediate waypoints
                                prev = remaining_path[i-1] if i > 0 else self.current_node
                                action = self.nav.get_action(prev, u, v)
                                if action != "STRAIGHT":
                                    t_run += self.turn_config.get("90_DEG", 1.2)
                            
                            self.blind_run_end_time = time.time() + t_run
                            self.blind_run_target_node = remaining_path[-1]
                            self._log(f"[BLIND-RUN] Đích cuối ({self._get_node_label(self.blind_run_target_node)}) là Waypoint. Tổng thời gian dự kiến: {t_run:.2f}s")
                        except Exception as e:
                            self._log(f"[BLIND-RUN ERROR] {e}")

                    self.state = CarState.TURNING
                else:
                    self._log(f"[WARN] Không tìm thấy đường tới {self.target_node}")
                    self.state = CarState.MOVING

        elif self.state == CarState.TURNING:
            if self.prev_node is None and self.current_node and self.next_node:
                # First node: handle initial orientation
                action = self.nav.get_initial_action(self.current_node, self.next_node, self.initial_heading)
                self._log(f"[INIT-TURN] Hướng: {self.initial_heading}° | Hành động: {action}")
            elif self.prev_node and self.current_node and self.next_node:
                action = self.nav.get_action(self.prev_node, self.current_node, self.next_node)
            else:
                action = "STRAIGHT"
            
            self._log(f"[TURN] Hành động: {action}")
            self.execute_motor_action(action)
            self.state = CarState.MOVING
            
            # Kiểm tra xem đích đến tiếp theo có phải là Waypoint (đích cuối) không
            is_final = False
            if self.predefined_path and self.next_node == self.predefined_path[-1]:
                is_final = True
            elif self.next_node == self.target_node:
                is_final = True
                
            # Handled in PLANNING now for better accuracy
            pass

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
                    # Log countdown every 0.5s to avoid spamming
                    if not hasattr(self, '_last_blind_log') or time.time() - self._last_blind_log >= 0.5:
                        self._log(f"[BLIND-RUN] Đang về đích... Còn lại {rem:.1f}s")
                        self._last_blind_log = time.time()
                    self.follow_lane()
            else:
                self.follow_lane()

    def follow_lane(self):
        raw_frame = camera_manager.get_frame()
        if raw_frame is None:
            motor.move_straight()
            return

        # Resize for CV processing (320x240 is efficient)
        frame = cv2.resize(raw_frame, (320, 240))
        
        if self.detection_mode == "sliding_window":
            right_x_raw, y_right, debug_ovl = self._sliding_window_lane(frame)
            # Override frame with visual feedback if needed
            self._draw_edge_overlay(frame, cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0), 100, 200))
            if debug_ovl is not None:
                frame = cv2.addWeighted(frame, 1.0, debug_ovl, 0.8, 0)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 100, 200)
            # Draw edge visualization overlay
            self._draw_edge_overlay(frame, edges)
            right_x_raw, y_right = self._find_right_lane(edges)

        y_left = y_right # For potential visualization constraints, or just use y_right
        
        # Lưu toạ độ mới nhất khi phát hiện được
        if right_x_raw != -1:
            self.last_right_x = right_x_raw

        right_x = right_x_raw
        memory_used = False

        # Nếu mất làn phải, dùng memory của làn phải
        if right_x == -1 and self.last_right_x != -1:
            right_x = self.last_right_x
            memory_used = True

        # --- Traffic Sign Detection ---
        current_time = time.time()
        if self.has_sign_detection and (current_time - self.last_sign_detect_time) > self.sign_detect_interval:
            self.last_sign_detect_time = current_time
            # detect_signs expects RGB and raw_frame is natively BGR from camera module
            rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            detected_signs = self.sign_detector.detect_signs(rgb_frame)
            
            self.last_detected_signs = []
            if detected_signs:
                for idx, (roi, bbox) in enumerate(detected_signs):
                    sign_type, confidence = self.sign_classifier.classify(roi)
                    if confidence >= 0.5: # Only consider reasonably confident detections
                        self.last_detected_signs.append((sign_type, confidence, bbox))
                        log_msg = f"[VISION] Detected {sign_type.upper()} sign ({confidence*100:.1f}%)"
                        self._log(log_msg)
                        print(log_msg)

        # --- Visualization Logic (HD) ---
        display_frame = frame.copy()
        
        # Overlay Traffic Signs (Scale bbox from raw_frame to display_frame)
        raw_h, raw_w = raw_frame.shape[:2]
        h_ratio = 240 / raw_h
        w_ratio = 320 / raw_w
        for sign_type, conf, (rx, ry, rw, rh) in self.last_detected_signs:
            x = int(rx * w_ratio)
            y = int(ry * h_ratio)
            w = int(rw * w_ratio)
            h = int(rh * h_ratio)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(display_frame, f"{sign_type} {int(conf*100)}%", (x, max(15, y - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        self._draw_lane_overlay(display_frame, -1, y_left, right_x, y_right)
        
        # Metadata drawings
        cv2.line(display_frame, (0, y_right), (320, y_right), (60, 60, 60), 1)
        cv2.line(display_frame, (160, 0), (160, 240), (200, 200, 200), 1)

        # Target markers (Only right)
        for tx, label in [(self.target_right, "TR")]:
            cv2.line(display_frame, (tx, y_right-10), (tx, y_right+10), (0, 255, 255), 2)
            cv2.putText(display_frame, label, (tx-10, y_right-14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        if right_x != -1:
            color = (255, 80, 0) if not memory_used else (0, 200, 255)
            cv2.circle(display_frame, (right_x, y_right), 6, color, -1)
            cv2.putText(display_frame, f"R:{right_x}", (right_x - 40, y_right - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        steering = 0
        mode = "search"
        if right_x != -1:
            mode = "right (mem)" if memory_used else "right"
            steering = -(right_x - self.target_right) * 3
        
        # HUD Orientation Mark (Enhanced)
        self._draw_orientation_mark(display_frame, steering)
        
        if mode != "search":
            color = (50, 255, 80) if not memory_used else (0, 230, 255)
            cv2.putText(display_frame, f"Mode: {mode} lane", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(display_frame, f"Mode: {mode}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)
            
        # Update debug frame for Streamlit
        self.debug_frame = display_frame

        # Motor control
        if mode != "search":
            motor.drive(self.base_speed, steering)
        else:
            # Fallback when even right memory is gone
            motor.drive(100, 0) # Spin right to find lane
            time.sleep(0.05)

    def _draw_lane_overlay(self, frame, left_x, y_left, right_x, y_right):
        # Simplified: Only draw right lane line
        h, w = frame.shape[:2]
        bot = h
        if right_x != -1:
            cv2.line(frame, (right_x, y_right), (right_x, bot), (255, 80, 0), 2)
            # Optional: subtle glow on the right side
            overlay = frame.copy()
            pts = np.array([[w // 2, y_right], [right_x, y_right], [right_x, bot], [w // 2, bot]], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (255, 80, 0))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

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
            motor.move_straight()
            time.sleep(self.turn_config["STRAIGHT"])
            motor.turn_right()
            time.sleep(self.turn_config["90_DEG"])
        elif action == "LEFT":
            motor.move_straight()
            time.sleep(self.turn_config["STRAIGHT"])
            motor.turn_left()
            time.sleep(self.turn_config["90_DEG"])
        elif action =="TURN_AROUND":
            motor.turn_right()
            time.sleep(self.turn_config["180_DEG"])
        
        motor.stop()
        time.sleep(0.1)

    def _get_birdseye_view(self, frame):
        """Warp frame to bird's-eye view using perspective transform."""
        h, w = frame.shape[:2]
        M = cv2.getPerspectiveTransform(self.perspective_src, self.perspective_dst)
        warped = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)
        return warped

    def _sliding_window_lane(self, frame):
        """Advanced lane detection using Sliding Window algorithm."""
        h, w = frame.shape[:2]
        warped = self._get_birdseye_view(frame)
        
        # Binary thresholding (Using saturation channel for better line detection)
        hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # Combine S channel and Gradient
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        
        binary = np.zeros_like(s_channel)
        binary[((s_channel > 100) & (s_channel <= 255)) | ((scaled_sobel > 50) & (scaled_sobel <= 255))] = 255
        
        # Histogram to find line starts
        histogram = np.sum(binary[h//2:, :], axis=0)
        midpoint = int(histogram.shape[0] // 2)
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Sliding Window Settings
        nwindows = 9
        window_height = int(h // nwindows)
        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        current_x = rightx_base
        margin = 40
        minpix = 50
        right_lane_inds = []
        
        debug_ovl = np.zeros_like(frame)
        
        for window in range(nwindows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height
            win_x_low = current_x - margin
            win_x_high = current_x + margin
            
            # Draw for debug
            cv2.rectangle(debug_ovl, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)
            
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            right_lane_inds.append(good_right_inds)
            
            if len(good_right_inds) > minpix:
                current_x = int(np.mean(nonzerox[good_right_inds]))
                
        right_lane_inds = np.concatenate(right_lane_inds)
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit Polynomial
        if len(rightx) > 0:
            right_fit = np.polyfit(righty, rightx, 2)
            # Calculate x at target y (scan_y_right)
            target_y = h * self.scan_y_right
            fit_x = right_fit[0]*target_y**2 + right_fit[1]*target_y + right_fit[2]
            
            # Map back to original perspective (Approximate)
            # For simplicity in this HUD, we just return the x value as if it's in original perspective 
            # after inverse warping or just use it as relative steering.
            # Real implementation would use M_inv.
            
            # Draw fitted line on debug overlay
            ploty = np.linspace(0, h-1, h)
            fit_x_pts = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            pts = np.array([np.transpose(np.vstack([fit_x_pts, ploty]))], np.int32)
            cv2.polylines(debug_ovl, pts, isClosed=False, color=(0, 255, 255), thickness=3)
            
            # Inverse Warp for visualization on the main frame
            M_inv = cv2.getPerspectiveTransform(self.perspective_dst, self.perspective_src)
            debug_ovl_unwarped = cv2.warpPerspective(debug_ovl, M_inv, (w, h))
            
            # Return result (scale relative to center)
            # We want current_x at a target_y. For basic logic, let's keep it simple:
            return int(fit_x), int(target_y), debug_ovl_unwarped
        
        return -1, int(h * self.scan_y_right), None

    def _get_dist(self, u, v):
        """Calculate Euclidean distance between two nodes."""
        try:
            import math
            n1 = self.nav.nodes[u]
            n2 = self.nav.nodes[v]
            return math.hypot(n2['x'] - n1['x'], n2['y'] - n1['y'])
        except KeyError:
            return 0.0

    def _get_node_label(self, node_id):
        """Helper to get 'Label [ID]' or just 'ID' for logs."""
        if not node_id: return "None"
        # Check if node exists in graph to avoid crashes
        if node_id in self.nav.graph.get('nodes', {}):
            lbl = self.nav.graph['nodes'][node_id].get('label', '')
            return f"{lbl} [{node_id}]" if lbl else f"[{node_id}]"
        return f"[{node_id}]"

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