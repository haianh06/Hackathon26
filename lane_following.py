import cv2
import numpy as np
import time
from collections import deque
from hardware.camera import camera_manager
import hardware.motor as motor

# ============================================================
# Configuration — Lane Following  (from autonomous_main.py)
# ============================================================
BASE_SPEED          = 120
TARGET_RIGHT        = 310
SCAN_Y_RIGHT        = 0.4      # Fraction of frame height to scan at
SCAN_SEARCH_UP_ROWS = 35       # How many rows upward to search for lane edge
STEERING_GAIN       = 3        # Proportional gain

PERSPECTIVE_SRC = np.float32([[120, 160], [200, 160], [300, 230], [20, 230]])
PERSPECTIVE_DST = np.float32([[80, 0],   [240, 0],   [240, 240], [80, 240]])

# ============================================================
# Configuration — Dead-End / Turn Detection
# ============================================================
#
# Architecture: 3-layer detection
#
#   LAYER 1 — GATE (prerequisite, cheap check):
#     [G1] Time gate : robot must have been in FOLLOW state >= GATE_MIN_FOLLOW_TIME
#                      (avoids triggering immediately after startup or a turn)
#     [G2] Density gate : upper zone must have minimal edge density
#                         (something must actually be visible ahead)
#     If gate fails → score = 0, no turn command, no false positive.
#
#   LAYER 2 — SIGNALS (only run when gate is open):
#     [S1] upper_wall_score  (0.40) : dense edges in top 35% = wall face is close
#     [S2] lane_loss_lower   (0.35) : long horizontal lane lines MISSING from
#                                     the lower band where they normally live
#                                     (45-83% from top = where camera sees lane)
#     [S3] upper_diagonal    (0.15) : diagonal fraction in upper zone = wall corner
#     [S4] asymmetry         (0.10) : turn direction only
#
#   LAYER 3 — CONFIRMATION:
#     [OPT-1] Rolling average over SMOOTH_WINDOW_SIZE frames
#     [C]     Score must stay >= threshold for CONFIRM_CONSECUTIVE frames in a row
#
# Why the old MIDDLE_BAND was wrong:
#   Lane lines from a forward-looking camera appear at y=70-85% of the frame
#   height (car sees road beneath it).  The old 30-70% band MISSED them entirely,
#   causing middle_long_horiz=0 on straight roads → lane_loss=1.0 always on.

# Weights: [upper_wall, lane_loss_lower, upper_diagonal, asymmetry]
SIGNAL_WEIGHTS = [0.40, 0.35, 0.15, 0.10]

# ── Gate parameters (Layer 1) ────────────────────────────────────────────────
GATE_MIN_FOLLOW_TIME = 2.5    # [G1] seconds in FOLLOW before detection starts
GATE_UPPER_DENSITY   = 0.04   # [G2] min edge density in upper zone to proceed

# ── Signal parameters (Layer 2) ───────────────────────────────────────────
# [S1] Upper zone: top UPPER_ZONE_FRAC of frame
UPPER_ZONE_FRAC      = 0.38
UPPER_WALL_THRESH    = 0.10   # density above this → wall is close
UPPER_WALL_MAX       = 0.25   # density at or above this → full score

# [S2] Lower-lane band: where the lane strip actually appears in camera view
#      Camera looks forward → lane is in the BOTTOM portion of the frame
LANE_BAND_LO         = 0.45   # row 45% (far edge of lane strip)
LANE_BAND_HI         = 0.83   # row 83% (near edge of lane strip)
LANE_LINE_MIN_SPAN   = 0.38   # must span 38% of width to count as lane line
LANE_LINE_MAX_ANGLE  = 15.0   # degrees from horizontal

# [S3] Upper diagonal fraction
DIAG_SCORE_THRESH    = 0.28   # fraction of diagonal lines in upper zone

# [S4] Asymmetry
ASYMMETRY_THRESH     = 0.25

# ── Scoring & confirmation (Layer 3) ─────────────────────────────────────────
DEAD_END_SCORE_THRESH  = 0.55
SMOOTH_WINDOW_SIZE     = 8     # [OPT-1] rolling average window
CONFIRM_CONSECUTIVE    = 5     # [C] frames above threshold IN A ROW before acting

# ── Hough parameters ─────────────────────────────────────────────────────────
HOUGH_THRESHOLD  = 40
HOUGH_MIN_LENGTH = 50
HOUGH_MAX_GAP    = 25

# ── ROI side strip (OPT-2) ─────────────────────────────────────────────────────
ROI_SIDE_STRIP = 0.08

TURN_DURATION = 1.2
COOLDOWN_TIME = 2.0

# ============================================================
# States
# ============================================================
STATE_FOLLOW   = "FOLLOWING"
STATE_TURN     = "TURNING"
STATE_COOLDOWN = "COOLDOWN"


# ============================================================
# Dead-End Detector  (3-layer: Gate → Signals → Confirmation)
# ============================================================
class DeadEndDetector:
    """
    3-layer dead-end detection for a white-strip lane.

    LAYER 1 — GATE (cheap prerequisite, runs every frame)
    ------------------------------------------------------
    [G1] Time gate  : robot must have been in FOLLOW state for at least
                      GATE_MIN_FOLLOW_TIME seconds.  This prevents triggering
                      right after startup or right after exiting a turn.
    [G2] Density gate : the upper zone must contain a minimum amount of edges
                        (GATE_UPPER_DENSITY).  On an open straight road the upper
                        zone is nearly empty — no wall, no obstacle.  If this gate
                        fails, we know we're on an open road and skip detection.
    When the gate fails: raw_score = 0, smoothing buffer is NOT updated,
    consecutive counter resets to 0.  Zero false positives on open roads.

    LAYER 2 — SIGNALS (only when gate is open)
    -------------------------------------------
    [S1] upper_wall_score (0.40)
         Density of white pixels in top UPPER_ZONE_FRAC of frame, normalised
         between UPPER_WALL_THRESH and UPPER_WALL_MAX.
         Dense upper edges = wall face is very close.

    [S2] lane_loss_lower (0.35)
         Count of long (>= LANE_LINE_MIN_SPAN) near-horizontal Hough lines
         inside the LANE_BAND_LO–LANE_BAND_HI zone.
         WHY NEW BAND: the camera on this robot places lane lines at y=45-83%
         of the frame (road is under the car, not in the middle of frame).
         The original 30-70% band missed them entirely, so lane_loss was always
         1.0 even on straight roads.
         score = exp(-1.5 * count): 0 lines→1.0, 1 line→0.22, 2→0.05

    [S3] upper_diagonal (0.15)
         Fraction of lines in upper zone that are diagonal (15°-75°).
         A wall corner seen close-up creates many diagonal edges.

    [S4] asymmetry (0.10)
         (right_lines - left_lines) / total → turn direction only.

    LAYER 3 — CONFIRMATION
    ----------------------
    [OPT-1] Rolling mean over SMOOTH_WINDOW_SIZE frames.
    [C]     Score must stay >= DEAD_END_SCORE_THRESH for CONFIRM_CONSECUTIVE
            consecutive frames before a turn is issued.
            Single-spike noise cannot trigger a turn.
    """

    def __init__(self):
        self._score_history  = deque(maxlen=SMOOTH_WINDOW_SIZE)  # [OPT-1]
        self._consec_count   = 0    # [C] consecutive frames above threshold
        self._follow_start   = None # timestamp when FOLLOW state began

    def notify_follow_start(self):
        """Call this when STATE_FOLLOW begins (or resets after cooldown)."""
        self._follow_start  = time.time()
        self._consec_count  = 0
        self._score_history.clear()

    def detect(self, canny_frame):
        """
        Parameters
        ----------
        canny_frame : np.ndarray  (grayscale uint8, output of cv2.Canny)

        Returns
        -------
        is_dead_end    : bool
        turn_dir       : str | None
        signals        : dict
        smoothed_score : float
        """
        h, w = canny_frame.shape

        # ── [OPT-2] ROI: strip noisy side edges ─────────────────────────────────
        strip = int(w * ROI_SIDE_STRIP)
        roi = canny_frame.copy()
        roi[:, :strip]     = 0
        roi[:, w - strip:] = 0

        # ── Zone boundaries ─────────────────────────────────────────────────
        upper_end   = int(h * UPPER_ZONE_FRAC)   # top zone for S1 & S3
        lane_lo     = int(h * LANE_BAND_LO)       # lane band for S2
        lane_hi     = int(h * LANE_BAND_HI)

        # ── LAYER 1: GATE ─────────────────────────────────────────────────────
        # [G1] Time gate
        t_in_follow = (
            time.time() - self._follow_start
            if self._follow_start is not None else 0.0
        )
        gate_time_ok = t_in_follow >= GATE_MIN_FOLLOW_TIME

        # [G2] Upper density gate (quick pixel count, no Hough needed)
        upper_zone     = roi[:upper_end, :]
        upper_density  = np.count_nonzero(upper_zone) / (upper_zone.size + 1e-6)
        gate_density_ok = upper_density >= GATE_UPPER_DENSITY

        gate_open = gate_time_ok and gate_density_ok

        # Build a minimal signals dict regardless (so HUD always has data)
        signals = {
            "gate_open"         : gate_open,
            "gate_time_ok"      : gate_time_ok,
            "gate_density_ok"   : gate_density_ok,
            "t_in_follow"       : t_in_follow,
            "upper_density"     : upper_density,
            # placeholders — filled below if gate is open
            "lower_density"     : 0.0,
            "ul_ratio"          : 0.0,
            "ul_score"          : 0.0,
            "lane_long_horiz"   : 0,
            "lane_loss"         : 0.0,
            "upper_total"       : 0,
            "upper_diagonal"    : 0,
            "upper_diag_ratio"  : 0.0,
            "diag_score"        : 0.0,
            "asymmetry"         : 0.0,
            "total_lines"       : 0,
            "raw_votes"         : [0.0, 0.0, 0.0, 0.0],
            "raw_score"         : 0.0,
            "smoothed_score"    : float(np.mean(self._score_history)) if self._score_history else 0.0,
            "consec_count"      : self._consec_count,
        }

        if not gate_open:
            # Gate blocked — do NOT update score history; reset consecutive counter
            self._consec_count = 0
            return False, None, signals, signals["smoothed_score"]

        # ── LAYER 2: SIGNALS ───────────────────────────────────────────────────

        # [S1] Upper wall density (normalised)
        wall_score = min(1.0, max(0.0,
            (upper_density - UPPER_WALL_THRESH) / (UPPER_WALL_MAX - UPPER_WALL_THRESH + 1e-6)
        ))

        # Lower zone density (for ul_ratio display)
        lower_zone    = roi[lane_hi:, :]
        lower_density = np.count_nonzero(lower_zone) / (lower_zone.size + 1e-6)
        ul_ratio      = upper_density / (lower_density + 1e-6)
        ul_score      = min(1.0, max(0.0, (ul_ratio - 0.8) / 1.2))  # display only

        # Hough lines
        lines = cv2.HoughLinesP(
            roi,
            rho=1, theta=np.pi / 180,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LENGTH,
            maxLineGap=HOUGH_MAX_GAP,
        )

        lane_long_horiz = 0   # [S2]: long horizontal lines in the lane band
        upper_total     = 0   # [S3]: lines in upper zone
        upper_diagonal  = 0
        left_cnt = right_cnt = total_cnt = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                total_cnt += 1
                length   = float(np.hypot(x2 - x1, y2 - y1))
                angle    = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) if x2 != x1 else 90.0
                cy_line  = (y1 + y2) / 2.0
                cx_line  = (x1 + x2) / 2.0

                # [S2] Is this a long horizontal line inside the lane band?
                if (lane_lo <= cy_line <= lane_hi
                        and angle < LANE_LINE_MAX_ANGLE
                        and length > w * LANE_LINE_MIN_SPAN):
                    lane_long_horiz += 1

                # [S3] Diagonal lines in upper zone
                if cy_line < upper_end:
                    upper_total += 1
                    if 15.0 < angle < 75.0:
                        upper_diagonal += 1

                if cx_line < w / 2:
                    left_cnt += 1
                else:
                    right_cnt += 1

        # [S2] Lane-loss score: no long horizontal lines in lane band = lane gone
        lane_loss    = float(np.exp(-1.5 * lane_long_horiz))  # 0 lines→1.0

        # [S3] Upper diagonal score
        upper_diag_ratio = upper_diagonal / (upper_total + 1e-6)
        diag_score       = min(1.0, upper_diag_ratio / (DIAG_SCORE_THRESH + 1e-6))

        # [S4] Asymmetry (direction only)
        asymmetry = (right_cnt - left_cnt) / (total_cnt + 1e-6)

        raw_votes = [
            wall_score,
            lane_loss,
            diag_score,
            min(1.0, abs(asymmetry) / (ASYMMETRY_THRESH + 1e-6)),
        ]
        raw_score = sum(wt * v for wt, v in zip(SIGNAL_WEIGHTS, raw_votes))

        # ── LAYER 3: CONFIRMATION ─────────────────────────────────────────────
        # [OPT-1] Rolling average
        self._score_history.append(raw_score)
        smoothed_score = float(np.mean(self._score_history))

        # [C] Consecutive frames above threshold
        if smoothed_score >= DEAD_END_SCORE_THRESH:
            self._consec_count += 1
        else:
            self._consec_count = 0

        is_dead_end = self._consec_count >= CONFIRM_CONSECUTIVE

        turn_dir = None
        if is_dead_end:
            turn_dir = "LEFT" if asymmetry > 0 else "RIGHT"

        # Update signals dict with full values
        signals.update({
            "lower_density"  : lower_density,
            "ul_ratio"       : ul_ratio,
            "ul_score"       : ul_score,
            "wall_score"     : wall_score,
            "lane_long_horiz": lane_long_horiz,
            "lane_loss"      : lane_loss,
            "upper_total"    : upper_total,
            "upper_diagonal" : upper_diagonal,
            "upper_diag_ratio": upper_diag_ratio,
            "diag_score"     : diag_score,
            "asymmetry"      : asymmetry,
            "total_lines"    : total_cnt,
            "raw_votes"      : raw_votes,
            "raw_score"      : raw_score,
            "smoothed_score" : smoothed_score,
            "consec_count"   : self._consec_count,
        })
        return is_dead_end, turn_dir, signals, smoothed_score

    def reset(self):
        """Clear all state (call after a turn completes)."""
        self._score_history.clear()
        self._consec_count = 0
        self._follow_start = None


# ============================================================
# Lane Following helpers  (from autonomous_main.py)
# ============================================================

def _get_birdseye(frame):
    h, w = frame.shape[:2]
    M = cv2.getPerspectiveTransform(PERSPECTIVE_SRC, PERSPECTIVE_DST)
    return cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)


def _find_right_lane(edges):
    """Scan from SCAN_Y_RIGHT upward for the first non-zero pixel in the right half."""
    height, width = edges.shape
    base_y = int(height * SCAN_Y_RIGHT)
    for offset in range(SCAN_SEARCH_UP_ROWS + 1):
        y = base_y - offset
        if y < 0:
            break
        right_half = edges[y, width // 2 :]
        if right_half.sum() != 0:
            inner_x = width // 2 + int(np.argmax(right_half))
            if inner_x < width - 1:
                return inner_x, y
    return -1, base_y


def _sliding_window_lane(frame):
    """Sliding-window lane detection (from autonomous_main.py)."""
    h, w = frame.shape[:2]
    warped = _get_birdseye(frame)

    hls       = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    gray      = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    sobelx     = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel  = np.absolute(sobelx)
    max_sobel  = np.max(abs_sobel)
    scaled     = np.uint8(255 * abs_sobel / max_sobel) if max_sobel > 0 else np.zeros_like(gray)

    binary = np.zeros_like(s_channel)
    binary[((s_channel > 100) & (s_channel <= 255)) | ((scaled > 50) & (scaled <= 255))] = 255

    histogram   = np.sum(binary[h // 2 :, :], axis=0)
    midpoint    = histogram.shape[0] // 2
    rightx_base = int(np.argmax(histogram[midpoint:])) + midpoint

    nwindows      = 9
    window_height = h // nwindows
    nonzero       = binary.nonzero()
    nonzeroy      = np.array(nonzero[0])
    nonzerox      = np.array(nonzero[1])

    current_x       = rightx_base
    margin          = 40
    minpix          = 50
    right_lane_inds = []
    debug_ovl       = np.zeros_like(frame)

    for window in range(nwindows):
        win_y_low  = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_x_low  = current_x - margin
        win_x_high = current_x + margin
        cv2.rectangle(debug_ovl, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

        good = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_x_low) & (nonzerox < win_x_high)
        ).nonzero()[0]
        right_lane_inds.append(good)
        if len(good) > minpix:
            current_x = int(np.mean(nonzerox[good]))

    right_lane_inds = np.concatenate(right_lane_inds)
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
        target_y  = h * SCAN_Y_RIGHT
        fit_x     = right_fit[0] * target_y ** 2 + right_fit[1] * target_y + right_fit[2]

        ploty    = np.linspace(0, h - 1, h)
        fit_pts  = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        pts      = np.array([np.transpose(np.vstack([fit_pts, ploty]))], np.int32)
        cv2.polylines(debug_ovl, pts, isClosed=False, color=(0, 255, 255), thickness=3)

        M_inv       = cv2.getPerspectiveTransform(PERSPECTIVE_DST, PERSPECTIVE_SRC)
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
    edge_overlay             = np.zeros_like(frame)
    edge_overlay[:, :, 0]   = edges
    edge_overlay[:, :, 2]   = edges // 2
    cv2.addWeighted(edge_overlay, 0.15, frame, 0.85, 0, frame)


def _draw_orientation_mark(frame, steering):
    h, w      = frame.shape[:2]
    base_y    = int(h * 0.82)
    cx        = w // 2
    max_steer = 200
    clamped   = max(-max_steer, min(max_steer, steering))
    arrow_len = int((clamped / max_steer) * (w // 4))
    tip_x     = cx + arrow_len
    ratio     = abs(clamped) / max_steer
    color     = (int(50 + ratio * 200), int(230 - ratio * 150), int(ratio * 255))
    bar_h     = 28
    cv2.rectangle(frame, (cx - w // 4, base_y - bar_h // 2), (cx + w // 4, base_y + bar_h // 2), (30, 30, 30), -1)
    cv2.rectangle(frame, (cx, base_y - bar_h // 4), (tip_x, base_y + bar_h // 4), color, -1)
    if arrow_len != 0:
        cv2.arrowedLine(frame, (cx, base_y), (tip_x, base_y), color, 2, tipLength=0.35)
    cv2.line(frame, (cx, base_y - bar_h // 2), (cx, base_y + bar_h // 2), (200, 200, 200), 1)
    cv2.putText(frame, f"Steer: {int(steering):+d}", (cx - w // 4, base_y - bar_h // 2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)


# ============================================================
# Debug overlay — Dead-End signals HUD
# ============================================================
def _draw_dead_end_hud(display, signals, smoothed_score, current_state):
    h, w = display.shape[:2]

    # State label
    state_color = {
        STATE_FOLLOW  : (0, 255, 0),
        STATE_TURN    : (0, 0, 255),
        STATE_COOLDOWN: (0, 255, 255),
    }.get(current_state, (255, 255, 255))
    cv2.putText(display, f"STATE: {current_state}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)

    # Gate status
    gate_open = signals.get("gate_open", False)
    t_fol     = signals.get("t_in_follow", 0.0)
    g1_ok     = signals.get("gate_time_ok", False)
    g2_ok     = signals.get("gate_density_ok", False)
    gate_color = (0, 220, 50) if gate_open else (0, 100, 200)
    gate_lbl   = "GATE: OPEN" if gate_open else f"GATE: CLOSED (G1={'OK' if g1_ok else f'{t_fol:.1f}s'} G2={'OK' if g2_ok else 'low'})"
    cv2.putText(display, gate_lbl, (8, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, gate_color, 1)

    # Smoothed score bar
    bar_w     = int(min(1.0, smoothed_score) * (w - 16))
    bar_color = (0, 0, 255) if smoothed_score >= DEAD_END_SCORE_THRESH else (0, 200, 50)
    cv2.rectangle(display, (8, 44), (8 + bar_w, 54), bar_color, -1)
    cv2.rectangle(display, (8, 44), (w - 8, 54), (180, 180, 180), 1)

    raw_s    = signals.get("raw_score", 0.0)
    consec   = signals.get("consec_count", 0)
    cv2.putText(display,
                f"Score raw={raw_s:.2f} smooth={smoothed_score:.2f} consec={consec}/{CONFIRM_CONSECUTIVE}",
                (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1)

    # Per-signal readouts
    votes = signals.get("raw_votes", [0, 0, 0, 0])
    upper_d = signals.get("upper_density", 0.0)
    labels = [
        f"S1 wall_scr  : {signals.get('wall_score',0):.2f}  upper={upper_d:.3f}  v={votes[0]:.2f}",
        f"S2 lane_loss : {signals.get('lane_loss',0):.2f}  (lines={signals.get('lane_long_horiz',0)})  v={votes[1]:.2f}",
        f"S3 diag_scr  : {signals.get('diag_score',0):.2f}  ({signals.get('upper_diagonal',0)}/{signals.get('upper_total',0)})  v={votes[2]:.2f}",
        f"S4 asymm     : {signals.get('asymmetry',0):+.2f}  v={votes[3]:.2f}",
        f"lines={signals.get('total_lines',0)}  ul={signals.get('ul_ratio',0):.2f}",
    ]
    for i, lbl in enumerate(labels):
        cv2.putText(display, lbl, (8, 74 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1)

    # Zone boundary lines (upper + lane band)
    upper_y  = int(h * UPPER_ZONE_FRAC)
    lane_lo_y = int(h * LANE_BAND_LO)
    lane_hi_y = int(h * LANE_BAND_HI)
    cv2.line(display, (0, upper_y),   (w, upper_y),   (100, 100, 255), 1)  # upper zone
    cv2.line(display, (0, lane_lo_y), (w, lane_lo_y), (255, 180,   0), 1)  # lane band top
    cv2.line(display, (0, lane_hi_y), (w, lane_hi_y), (255, 180,   0), 1)  # lane band bot
    cv2.putText(display, "up",    (4, upper_y   - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (100, 100, 255), 1)
    cv2.putText(display, "ln-lo", (4, lane_lo_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (255, 180,   0), 1)
    cv2.putText(display, "ln-hi", (4, lane_hi_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (255, 180,   0), 1)

    # ROI strip markers
    strip = int(w * ROI_SIDE_STRIP)
    cv2.line(display, (strip,     0), (strip,     h), (80, 80, 0), 1)
    cv2.line(display, (w - strip, 0), (w - strip, h), (80, 80, 0), 1)


# ============================================================
# Main loop
# ============================================================
def follow_lane_frame(frame, last_right_x, detection_mode="basic"):
    """
    Process one frame for lane following.
    Returns: (right_x_used, y_right, steering, memory_used, display_frame, edges)
    """
    img_small = cv2.resize(frame, (320, 240))

    if detection_mode == "sliding_window":
        right_x_raw, y_right, debug_ovl = _sliding_window_lane(img_small)
        gray    = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 100, 200)
        _draw_edge_overlay(img_small, edges)
        if debug_ovl is not None:
            img_small = cv2.addWeighted(img_small, 1.0, debug_ovl, 0.8, 0)
    else:
        gray    = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 100, 200)
        _draw_edge_overlay(img_small, edges)
        right_x_raw, y_right = _find_right_lane(edges)

    # Memory fallback
    right_x     = right_x_raw
    memory_used = False
    if right_x == -1 and last_right_x != -1:
        right_x     = last_right_x
        memory_used = True

    # Steering
    steering = 0
    mode     = "search"
    if right_x != -1:
        mode     = "right (mem)" if memory_used else "right"
        steering = -(right_x - TARGET_RIGHT) * STEERING_GAIN

    # Display frame
    display = img_small.copy()
    _draw_lane_overlay(display, right_x if right_x != -1 else -1, y_right)

    # Target marker
    cv2.line(display, (TARGET_RIGHT, y_right - 10), (TARGET_RIGHT, y_right + 10), (0, 255, 255), 2)
    cv2.putText(display, "TR", (TARGET_RIGHT - 10, y_right - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.line(display, (0, y_right), (320, y_right), (60, 60, 60), 1)
    cv2.line(display, (160, 0), (160, 240), (200, 200, 200), 1)

    if right_x != -1:
        color = (255, 80, 0) if not memory_used else (0, 200, 255)
        cv2.circle(display, (right_x, y_right), 6, color, -1)
        cv2.putText(display, f"R:{right_x}", (right_x - 40, y_right - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    if mode != "search":
        color = (50, 255, 80) if not memory_used else (0, 230, 255)
        cv2.putText(display, f"Mode: {mode} lane", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        cv2.putText(display, "Mode: search", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)

    _draw_orientation_mark(display, steering)

    return right_x, y_right, steering, memory_used, mode, display, edges


def main():
    print("=== Lane Following + Dead-End Detection (3-layer Gate) ===")
    print(f"  [G1] Time gate          : {GATE_MIN_FOLLOW_TIME}s in FOLLOW before detection")
    print(f"  [G2] Upper density gate : {GATE_UPPER_DENSITY} min edge density ahead")
    print(f"  [C]  Consecutive frames : {CONFIRM_CONSECUTIVE} frames above {DEAD_END_SCORE_THRESH} to confirm")
    print(f"  [OPT-1] Smooth window   : {SMOOTH_WINDOW_SIZE} frames")
    print(f"  [OPT-2] ROI side strip  : {int(ROI_SIDE_STRIP*100)}% each side")
    print(f"  Lane band               : {int(LANE_BAND_LO*100)}-{int(LANE_BAND_HI*100)}% from top")
    print()

    camera_manager.start()
    motor.stop()
    time.sleep(2.0)

    detector      = DeadEndDetector()
    current_state = STATE_FOLLOW
    turn_dir_lock = None
    state_start   = 0.0
    last_right_x  = -1

    # [G1] Start the follow timer from launch
    detector.notify_follow_start()

    try:
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                continue

            # ── Lane following processing ─────────────────────────────────────
            right_x, y_right, steering, memory_used, mode, display, edges = \
                follow_lane_frame(frame, last_right_x, detection_mode="basic")

            if right_x != -1 and not memory_used:
                last_right_x = right_x

            # ── Dead-end detection (only in FOLLOW state) — [OPT-4] state lock
            signals       = {}
            smoothed_score = 0.0
            is_dead_end   = False
            turn_dir      = None

            if current_state == STATE_FOLLOW:
                is_dead_end, turn_dir, signals, smoothed_score = detector.detect(edges)
            else:
                # Keep HUD values from last live frame
                smoothed_score = float(np.mean(detector._score_history)) if detector._score_history else 0.0

            # ── State machine ─────────────────────────────────────────────────
            now = time.time()

            if current_state == STATE_FOLLOW:
                if is_dead_end:
                    turn_dir_lock = turn_dir
                    print(
                        f"[DEAD-END] smooth={smoothed_score:.2f} consec={signals.get('consec_count',0)} "
                        f"→ TURN {turn_dir_lock} | "
                        f"wall={signals.get('wall_score',0):.2f} "
                        f"lane_loss={signals.get('lane_loss',0):.2f}({signals.get('lane_long_horiz',0)}lines) "
                        f"diag={signals.get('diag_score',0):.2f} "
                        f"asym={signals.get('asymmetry',0):+.2f}"
                    )
                    current_state = STATE_TURN
                    state_start   = now
                    if turn_dir_lock == "LEFT":
                        motor.turn_left()
                    else:
                        motor.turn_right()
                else:
                    # Lane following motor command
                    if mode != "search":
                        motor.drive(BASE_SPEED, int(steering))
                    else:
                        motor.drive(100, 0)

            elif current_state == STATE_TURN:
                if now - state_start >= TURN_DURATION:
                    print("Turn complete. → COOLDOWN")
                    motor.stop()
                    detector.reset()   # [OPT-1] reset smoothing after turn
                    current_state = STATE_COOLDOWN
                    state_start   = now

            elif current_state == STATE_COOLDOWN:
                if now - state_start >= COOLDOWN_TIME:
                    print("Cooldown done. → FOLLOWING")
                    turn_dir_lock = None
                    current_state = STATE_FOLLOW
                    detector.notify_follow_start()  # [G1] restart time gate
                else:
                    if mode != "search":
                        motor.drive(BASE_SPEED, int(steering))
                    else:
                        motor.drive(100, 0)

            # ── Draw dead-end HUD on top of lane display ──────────────────────
            _draw_dead_end_hud(display, signals, smoothed_score, current_state)

            cv2.imshow("Lane Following", display)
            cv2.imshow("Canny Edges",   edges)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        motor.stop()
        camera_manager.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
