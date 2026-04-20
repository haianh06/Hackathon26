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
SCAN_Y_RIGHT        = 0.73     # Fraction of frame height to scan at
SCAN_SEARCH_UP_ROWS = 35       # How many rows upward to search for lane edge
STEERING_GAIN       = 3        # Proportional gain

PERSPECTIVE_SRC = np.float32([[120, 160], [200, 160], [300, 230], [20, 230]])
PERSPECTIVE_DST = np.float32([[80, 0],   [240, 0],   [240, 240], [80, 240]])

# ============================================================
# Configuration — Dead-End / Turn Detection
# ============================================================
SIGNAL_WEIGHTS = [0.40, 0.35, 0.15, 0.10]
GATE_MIN_FOLLOW_TIME = 2.5    
GATE_UPPER_DENSITY   = 0.04   
UPPER_ZONE_FRAC      = 0.75
UPPER_WALL_THRESH    = 0.10   
UPPER_WALL_MAX       = 0.25   
LANE_BAND_LO         = 0.70
LANE_BAND_HI         = 0.90  
LANE_LINE_MIN_SPAN   = 0.38   
LANE_LINE_MAX_ANGLE  = 15.0   
DIAG_SCORE_THRESH    = 0.28   
ASYMMETRY_THRESH     = 0.25
DEAD_END_SCORE_THRESH  = 0.55
SMOOTH_WINDOW_SIZE     = 8     
CONFIRM_CONSECUTIVE    = 5     
HOUGH_THRESHOLD  = 40
HOUGH_MIN_LENGTH = 50
HOUGH_MAX_GAP    = 25
ROI_SIDE_STRIP = 0.08
TURN_DURATION = 1.2
COOLDOWN_TIME = 2.0

# ============================================================
# Dead-End Detector (Standalone)
# ============================================================
class DeadEndDetector:
    def __init__(self):
        self._score_history  = deque(maxlen=SMOOTH_WINDOW_SIZE)
        self._consec_count   = 0
        self._follow_start   = None

    def notify_follow_start(self):
        self._follow_start  = time.time()
        self._consec_count  = 0
        self._score_history.clear()

    def detect(self, canny_frame):
        h, w = canny_frame.shape
        strip = int(w * ROI_SIDE_STRIP)
        roi = canny_frame.copy()
        roi[:, :strip]     = 0
        roi[:, w - strip:] = 0

        upper_end   = int(h * UPPER_ZONE_FRAC)
        lane_lo     = int(h * LANE_BAND_LO)
        lane_hi     = int(h * LANE_BAND_HI)

        t_in_follow = (time.time() - self._follow_start if self._follow_start is not None else 0.0)
        gate_time_ok = t_in_follow >= GATE_MIN_FOLLOW_TIME
        upper_zone     = roi[:upper_end, :]
        upper_density  = np.count_nonzero(upper_zone) / (upper_zone.size + 1e-6)
        gate_density_ok = upper_density >= GATE_UPPER_DENSITY
        gate_open = gate_time_ok and gate_density_ok

        signals = {
            "gate_open": gate_open, "gate_time_ok": gate_time_ok, "gate_density_ok": gate_density_ok,
            "t_in_follow": t_in_follow, "upper_density": upper_density, "lower_density": 0.0,
            "ul_ratio": 0.0, "ul_score": 0.0, "lane_long_horiz": 0, "lane_loss": 0.0,
            "upper_total": 0, "upper_diagonal": 0, "upper_diag_ratio": 0.0, "diag_score": 0.0,
            "asymmetry": 0.0, "total_lines": 0, "raw_votes": [0.0, 0.0, 0.0, 0.0], "raw_score": 0.0,
            "smoothed_score": float(np.mean(self._score_history)) if self._score_history else 0.0,
            "consec_count": self._consec_count,
        }
        if not gate_open:
            self._consec_count = 0
            return False, None, signals, signals["smoothed_score"]

        wall_score = min(1.0, max(0.0, (upper_density - UPPER_WALL_THRESH) / (UPPER_WALL_MAX - UPPER_WALL_THRESH + 1e-6)))
        lower_zone    = roi[lane_hi:, :]
        lower_density = np.count_nonzero(lower_zone) / (lower_zone.size + 1e-6)
        ul_ratio      = upper_density / (lower_density + 1e-6)
        ul_score      = min(1.0, max(0.0, (ul_ratio - 0.8) / 1.2))

        lines = cv2.HoughLinesP(roi, 1, np.pi/180, HOUGH_THRESHOLD, minLineLength=HOUGH_MIN_LENGTH, maxLineGap=HOUGH_MAX_GAP)
        lane_long_horiz = upper_total = upper_diagonal = left_cnt = right_cnt = total_cnt = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]; total_cnt += 1
                length = float(np.hypot(x2 - x1, y2 - y1))
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) if x2 != x1 else 90.0
                cy_line = (y1 + y2) / 2.0; cx_line = (x1 + x2) / 2.0
                if (lane_lo <= cy_line <= lane_hi and angle < LANE_LINE_MAX_ANGLE and length > w * LANE_LINE_MIN_SPAN):
                    lane_long_horiz += 1
                if cy_line < upper_end:
                    upper_total += 1
                    if 15.0 < angle < 75.0: upper_diagonal += 1
                if cx_line < w / 2: left_cnt += 1
                else: right_cnt += 1

        lane_loss = float(np.exp(-1.5 * lane_long_horiz))
        upper_diag_ratio = upper_diagonal / (upper_total + 1e-6)
        diag_score = min(1.0, upper_diag_ratio / (DIAG_SCORE_THRESH + 1e-6))
        asymmetry = (right_cnt - left_cnt) / (total_cnt + 1e-6)
        raw_votes = [wall_score, lane_loss, diag_score, min(1.0, abs(asymmetry) / (ASYMMETRY_THRESH + 1e-6))]
        raw_score = sum(wt * v for wt, v in zip(SIGNAL_WEIGHTS, raw_votes))
        self._score_history.append(raw_score)
        smoothed_score = float(np.mean(self._score_history))
        if smoothed_score >= DEAD_END_SCORE_THRESH: self._consec_count += 1
        else: self._consec_count = 0
        is_dead_end = self._consec_count >= CONFIRM_CONSECUTIVE
        turn_dir = ("LEFT" if asymmetry > 0 else "RIGHT") if is_dead_end else None
        signals.update({
            "lower_density": lower_density, "ul_ratio": ul_ratio, "ul_score": ul_score,
            "wall_score": wall_score, "lane_long_horiz": lane_long_horiz, "lane_loss": lane_loss,
            "upper_total": upper_total, "upper_diagonal": upper_diagonal, "upper_diag_ratio": upper_diag_ratio,
            "diag_score": diag_score, "asymmetry": asymmetry, "total_lines": total_cnt,
            "raw_votes": raw_votes, "raw_score": raw_score, "smoothed_score": smoothed_score, "consec_count": self._consec_count,
        })
        return is_dead_end, turn_dir, signals, smoothed_score

    def reset(self):
        self._score_history.clear()
        self._consec_count = 0
        self._follow_start = None

# ============================================================
# Helper Functions (Standalone)
# ============================================================
def _get_birdseye(frame):
    h, w = frame.shape[:2]
    M = cv2.getPerspectiveTransform(PERSPECTIVE_SRC, PERSPECTIVE_DST)
    return cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)

def _find_right_lane(edges):
    height, width = edges.shape
    base_y = int(height * SCAN_Y_RIGHT)
    for offset in range(SCAN_SEARCH_UP_ROWS + 1):
        y = base_y - offset
        if y < 0: break
        right_half = edges[y, width // 2 :]
        if right_half.sum() != 0:
            inner_x = width // 2 + int(np.argmax(right_half))
            if inner_x < width - 1: return inner_x, y
    return -1, base_y

def _sliding_window_lane(frame):
    h, w = frame.shape[:2]
    warped = _get_birdseye(frame)
    hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel = np.absolute(sobelx)
    max_sobel = np.max(abs_sobel)
    scaled = np.uint8(255 * abs_sobel / max_sobel) if max_sobel > 0 else np.zeros_like(gray)
    binary = np.zeros_like(s_channel)
    binary[((s_channel > 100) & (s_channel <= 255)) | ((scaled > 50) & (scaled <= 255))] = 255
    histogram = np.sum(binary[h // 2 :, :], axis=0)
    midpoint = histogram.shape[0] // 2
    rightx_base = int(np.argmax(histogram[midpoint:])) + midpoint
    nwindows = 9
    window_height = h // nwindows
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0]); nonzerox = np.array(nonzero[1])
    current_x = rightx_base; margin = 40; minpix = 50
    right_lane_inds = []; debug_ovl = np.zeros_like(frame)
    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height; win_y_high = h - window * window_height
        win_x_low = current_x - margin; win_x_high = current_x + margin
        cv2.rectangle(debug_ovl, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)
        good = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        right_lane_inds.append(good)
        if len(good) > minpix: current_x = int(np.mean(nonzerox[good]))
    right_lane_inds = np.concatenate(right_lane_inds)
    rightx = nonzerox[right_lane_inds]; righty = nonzeroy[right_lane_inds]
    if len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
        target_y = h * SCAN_Y_RIGHT
        fit_x = right_fit[0] * target_y ** 2 + right_fit[1] * target_y + right_fit[2]
        ploty = np.linspace(0, h - 1, h)
        fit_pts = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        pts = np.array([np.transpose(np.vstack([fit_pts, ploty]))], np.int32)
        cv2.polylines(debug_ovl, pts, isClosed=False, color=(0, 255, 255), thickness=3)
        M_inv = cv2.getPerspectiveTransform(PERSPECTIVE_DST, PERSPECTIVE_SRC)
        ovl_unwarped = cv2.warpPerspective(debug_ovl, M_inv, (w, h))
        return int(fit_x), int(target_y), ovl_unwarped
    return -1, int(h * SCAN_Y_RIGHT), None

def _draw_lane_overlay(frame, right_x, y_right):
    h, w = frame.shape[:2]
    if right_x != -1:
        cv2.line(frame, (right_x, y_right), (right_x, h), (255, 80, 0), 2)
        overlay = frame.copy()
        pts = np.array([[w // 2, y_right], [right_x, y_right], [right_x, h], [w // 2, h]], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (255, 80, 0))
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

def _draw_dead_end_hud(display, signals, smoothed_score, current_state_name):
    h, w = display.shape[:2]
    state_color = (0, 255, 0) if "MOVING" in current_state_name or "FOLLOW" in current_state_name else (0, 0, 255)
    cv2.putText(display, f"STATE: {current_state_name}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)
    gate_open = signals.get("gate_open", False); t_fol = signals.get("t_in_follow", 0.0)
    g1_ok = signals.get("gate_time_ok", False); g2_ok = signals.get("gate_density_ok", False)
    gate_color = (0, 220, 50) if gate_open else (0, 100, 200)
    gate_lbl = "GATE: OPEN" if gate_open else f"GATE: CLOSED (G1={'OK' if g1_ok else f'{t_fol:.1f}s'} G2={'OK' if g2_ok else 'low'})"
    cv2.putText(display, gate_lbl, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.32, gate_color, 1)
    bar_w = int(min(1.0, smoothed_score) * (w - 16))
    bar_color = (0, 0, 255) if smoothed_score >= DEAD_END_SCORE_THRESH else (0, 200, 50)
    cv2.rectangle(display, (8, 44), (8 + bar_w, 54), bar_color, -1)
    cv2.rectangle(display, (8, 44), (w - 8, 54), (180, 180, 180), 1)
    cv2.putText(display, f"Score raw={signals.get('raw_score',0.0):.2f} smooth={smoothed_score:.2f} consec={signals.get('consec_count',0)}/{CONFIRM_CONSECUTIVE}", (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1)
    labels = [
        f"S1 wall_scr  : {signals.get('wall_score',0):.2f} (up={signals.get('upper_density',0):.3f})",
        f"S2 lane_loss : {signals.get('lane_loss',0):.2f} (ln={signals.get('lane_long_horiz',0)})",
        f"S3 diag_scr  : {signals.get('diag_score',0):.2f}",
        f"S4 asymm     : {signals.get('asymmetry',0):+.2f}",
    ]
    for i, lbl in enumerate(labels):
        cv2.putText(display, lbl, (8, 74 + i * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1)
    up_y = int(h * UPPER_ZONE_FRAC); ln_lo = int(h * LANE_BAND_LO); ln_hi = int(h * LANE_BAND_HI)
    cv2.line(display, (0, up_y), (w, up_y), (100, 100, 255), 1)
    cv2.line(display, (0, ln_lo), (w, ln_lo), (255, 180, 0), 1)
    cv2.line(display, (0, ln_hi), (w, ln_hi), (255, 180, 0), 1)
    strip = int(w * ROI_SIDE_STRIP)
    cv2.line(display, (strip, 0), (strip, h), (80, 80, 0), 1)
    cv2.line(display, (w - strip, 0), (w - strip, h), (80, 80, 0), 1)

def follow_lane_frame(frame, last_right_x, detection_mode="basic"):
    img_small = cv2.resize(frame, (320, 240))
    if detection_mode == "sliding_window":
        right_x_raw, y_right, debug_ovl = _sliding_window_lane(img_small)
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY); blurred = cv2.GaussianBlur(gray, (5, 5), 0); edges = cv2.Canny(blurred, 100, 200)
        _draw_edge_overlay(img_small, edges)
        if debug_ovl is not None: img_small = cv2.addWeighted(img_small, 1.0, debug_ovl, 0.8, 0)
    else:
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY); blurred = cv2.GaussianBlur(gray, (5, 5), 0); edges = cv2.Canny(blurred, 100, 200)
        _draw_edge_overlay(img_small, edges)
        right_x_raw, y_right = _find_right_lane(edges)
    right_x = right_x_raw; memory_used = False
    if right_x == -1 and last_right_x != -1: right_x = last_right_x; memory_used = True
    steering = 0; mode = "search"
    if right_x != -1:
        mode = "right (mem)" if memory_used else "right"
        steering = -(right_x - TARGET_RIGHT) * STEERING_GAIN
    display = img_small.copy()
    _draw_lane_overlay(display, right_x if right_x != -1 else -1, y_right)
    cv2.line(display, (TARGET_RIGHT, y_right - 10), (TARGET_RIGHT, y_right + 10), (0, 255, 255), 2)
    cv2.line(display, (0, y_right), (320, y_right), (60, 60, 60), 1)
    cv2.line(display, (160, 0), (160, 240), (200, 200, 200), 1)
    if right_x != -1:
        color = (255, 80, 0) if not memory_used else (0, 200, 255)
        cv2.circle(display, (right_x, y_right), 6, color, -1)
    _draw_orientation_mark(display, steering)
    return right_x, y_right, steering, memory_used, mode, display, edges

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
        self.last_right_x = -1
        self.base_speed = 120
        self.target_right = 310        
        self.scan_y_right = 0.4
        self.scan_search_up_rows = 35
        self.debug_frame = None
        self.blind_run_end_time = None
        self.blind_run_target_node = None
        self._last_blind_log = 0

        # RFID Threading state
        self._latest_uid = None
        self._rfid_lock = threading.Lock()
        self._rfid_thread = None

        self.dead_end_detector = DeadEndDetector()
        self._dead_end_signals  = {}   
        self.detection_mode = lane_mode

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
        self.dead_end_detector.notify_follow_start()
        
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
            self.dead_end_detector.notify_follow_start()

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

        # Gọi thuật toán standalone
        right_x, y_right, steering, memory_used, mode, display_frame, edges = \
            follow_lane_frame(raw_frame, self.last_right_x, detection_mode=self.detection_mode)

        if right_x != -1 and not memory_used:
            self.last_right_x = right_x

        # ── Dead-End Detection ────────────────────────────────────────────────
        is_dead_end, turn_dir, de_signals, de_score = self.dead_end_detector.detect(edges)
        self._dead_end_signals = de_signals

        if is_dead_end:
            self._log(f"[DEAD-END] score={de_score:.2f} → TURN {turn_dir}")
            motor.stop(); time.sleep(0.1)
            if turn_dir == "LEFT": motor.turn_left()
            else: motor.turn_right()
            time.sleep(self.turn_config.get("90_DEG", 1.2))
            motor.stop(); time.sleep(0.1)
            self.dead_end_detector.notify_follow_start()
            return

        # --- Traffic Sign Visualization (Asynchronous results) ---
        # No more detection here, just reading results from the background thread
        with self._sign_lock:
            current_signs = list(self.last_detected_signs)
            
        # Draw Dead-End HUD & Sign boxes
        _draw_dead_end_hud(display_frame, de_signals, de_score, self.state.name)
        
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