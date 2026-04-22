import cv2
import numpy as np
import time
import hardware.motor as motor
from hardware.camera import camera_manager

# ============================================================
# Configuration — Lane Following (Restored from autonomous_main.py)
# ============================================================
BASE_SPEED          = 120
TARGET_RIGHT        = 300
TARGET_LEFT         = 20
SCAN_Y_RIGHT        = 0.65     # Fraction of frame height to scan at
SCAN_SEARCH_UP_ROWS = 35       # How many rows upward to search for lane edge
STEERING_GAIN       = 3        # Proportional gain



def _find_right_lane(edges, last_x):
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

def main():
    print("Initializing Camera...")
    camera_manager.start()
    time.sleep(1.0)
    
    print("Press SPACE to start driving, ESC to exit.")
    
    last_steering = 0
    pos_history = []
    motor.stop()
    
    # Pre-run camera view
    while True:
        frame = camera_manager.get_frame()
        if frame is not None:
            # For preview, we don't update last_steering yet
            _, _, steering, _, _, display, _ = follow_lane_frame(frame, last_steering, pos_history)
            
            img_disp = cv2.resize(display, (640, 480))
            cv2.putText(img_disp, "PREVIEW MODE - Press SPACE to START", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Lane Following - Preview", img_disp)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            print("Starting motor control...")
            cv2.destroyWindow("Lane Following - Preview")
            break
        elif key == 27: # ESC
            camera_manager.stop()
            return

    try:
        while True:
            raw_frame = camera_manager.get_frame()
            if raw_frame is None: continue
            
            lane_x, y_lane, steering, memory_used, mode, display, edges = follow_lane_frame(raw_frame, last_steering, pos_history)
            last_steering = steering
            
            if mode != "search":
                motor.drive(BASE_SPEED, steering)
            else:
                motor.drive(80, 0) # Slow search
            
            cv2.imshow("Lane Following", display)
            cv2.imshow("Edges", edges)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        motor.stop()
        camera_manager.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
