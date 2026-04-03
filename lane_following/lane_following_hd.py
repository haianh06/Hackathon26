import os
import signal
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manual_control.servo import (
    LEFT_PIN,
    RIGHT_PIN,
    STOP_L,
    STOP_R,
    _set_pwm,
    get_servo_monitor_state,
    log_servo_monitor,
    release,
    shutdown_motors,
    stop,
)

# ============ CONTROL PARAMETERS ============
# Keep steering behavior close to the old version: steering = -(error * 3).
Kp = 3.0
Kd = 0.0
Ki = 0.0

BASE_SPEED = 120
MIN_SPEED = 80
STEERING_LIMIT = 200
STEERING_SLEW_LIMIT = 35
DEADBAND_PX = 4
INTEGRAL_LIMIT = 400
SPEED_REDUCTION_GAIN = 0.18

LEFT_PWM_TRIM_US = 0
RIGHT_PWM_TRIM_US = 0

WAIT_FOR_INITIAL_LANE_LOCK = True
LOST_HOLD_FRAMES = 8
LOST_RIGHT_TURN_MIN = 18
LOST_RIGHT_TURN_MAX = 72
LOST_RIGHT_TURN_RAMP_FRAMES = 18

TARGET_LEFT = 50
TARGET_RIGHT = 300
TARGET_CENTER = (TARGET_LEFT + TARGET_RIGHT) // 2

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
CAMERA_FLIP_CODE = None

ROI_TOP_RATIO = 0.45
CONTROL_Y_RATIO = 0.88
SCAN_Y_RATIOS = (0.92, 0.88, 0.84, 0.80, 0.76, 0.72, 0.68, 0.64, 0.60)
MIN_SEGMENT_WIDTH = 6
MIN_VALID_LANE_WIDTH = 90
MAX_VALID_LANE_WIDTH = 290

WHITE_GRAY_MIN = 160
HSV_WHITE_LOW = np.array([0, 0, 150], dtype=np.uint8)
HSV_WHITE_HIGH = np.array([180, 80, 255], dtype=np.uint8)
HLS_WHITE_LOW = np.array([0, 150, 0], dtype=np.uint8)
HLS_WHITE_HIGH = np.array([180, 255, 120], dtype=np.uint8)

running = True


def drive_motor(speed, steering):
    steering = int(np.clip(steering, -STEERING_LIMIT, STEERING_LIMIT))
    left_us = STOP_L + LEFT_PWM_TRIM_US - int(speed) - steering
    right_us = STOP_R + RIGHT_PWM_TRIM_US + int(speed) - steering
    _set_pwm(LEFT_PIN, left_us)
    _set_pwm(RIGHT_PIN, right_us)
    return left_us, right_us


def release_motor_output():
    release()
    return 0, 0


def build_lane_mask(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

    mask_gray = cv2.inRange(gray, WHITE_GRAY_MIN, 255)
    mask_hsv = cv2.inRange(hsv, HSV_WHITE_LOW, HSV_WHITE_HIGH)
    mask_hls = cv2.inRange(hls, HLS_WHITE_LOW, HLS_WHITE_HIGH)
    mask = cv2.bitwise_and(mask_gray, cv2.bitwise_or(mask_hsv, mask_hls))

    roi_top = int(FRAME_HEIGHT * ROI_TOP_RATIO)
    mask[:roi_top, :] = 0

    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    return mask, roi_top


def extract_segments(line):
    binary = line > 0
    if not np.any(binary):
        return []

    changes = np.diff(binary.astype(np.int8))
    starts = list(np.where(changes == 1)[0] + 1)
    ends = list(np.where(changes == -1)[0])

    if binary[0]:
        starts.insert(0, 0)
    if binary[-1]:
        ends.append(len(binary) - 1)

    segments = []
    for start, end in zip(starts, ends):
        if end - start + 1 >= MIN_SEGMENT_WIDTH:
            segments.append((int(start), int(end)))
    return segments


def fit_weighted_line(points, control_y):
    if len(points) < 2:
        if len(points) == 1:
            return int(points[0][1]), None
        return None, None

    ys = np.array([p[0] for p in points], dtype=np.float32)
    xs = np.array([p[1] for p in points], dtype=np.float32)
    ws = np.array([p[2] for p in points], dtype=np.float32)
    coeffs = np.polyfit(ys, xs, deg=1, w=ws)
    x_at_control = int(np.polyval(coeffs, control_y))
    return x_at_control, coeffs


def collect_inner_lane_points(mask):
    center_x = FRAME_WIDTH // 2
    control_y = int(FRAME_HEIGHT * CONTROL_Y_RATIO)

    left_points = []
    right_points = []
    debug_rows = []

    for index, ratio in enumerate(SCAN_Y_RATIOS):
        y = int(FRAME_HEIGHT * ratio)
        line = mask[y]
        segments = extract_segments(line)

        left_segments = [seg for seg in segments if (seg[0] + seg[1]) // 2 < center_x]
        right_segments = [seg for seg in segments if (seg[0] + seg[1]) // 2 > center_x]

        left_segment = max(left_segments, key=lambda seg: seg[1]) if left_segments else None
        right_segment = min(right_segments, key=lambda seg: seg[0]) if right_segments else None

        weight = len(SCAN_Y_RATIOS) - index
        if left_segment is not None:
            left_points.append((y, left_segment[1], weight))
        if right_segment is not None:
            right_points.append((y, right_segment[0], weight))

        debug_rows.append(
            {
                "y": y,
                "segments": segments,
                "left_segment": left_segment,
                "right_segment": right_segment,
            }
        )

    left_x, left_fit = fit_weighted_line(left_points, control_y)
    right_x, right_fit = fit_weighted_line(right_points, control_y)

    return {
        "control_y": control_y,
        "left_x": left_x,
        "right_x": right_x,
        "left_fit": left_fit,
        "right_fit": right_fit,
        "left_points": left_points,
        "right_points": right_points,
        "debug_rows": debug_rows,
    }


def evaluate_lane_measurement(lane_state):
    left_x = lane_state["left_x"]
    right_x = lane_state["right_x"]

    if left_x is not None and right_x is not None:
        lane_width = right_x - left_x
        if MIN_VALID_LANE_WIDTH <= lane_width <= MAX_VALID_LANE_WIDTH:
            return {
                "mode": "inner-center",
                "measurement_x": (left_x + right_x) // 2,
                "target_x": TARGET_CENTER,
                "lane_width": lane_width,
            }

    if right_x is not None:
        return {
            "mode": "right-inner",
            "measurement_x": right_x,
            "target_x": TARGET_RIGHT,
            "lane_width": None,
        }

    if left_x is not None:
        return {
            "mode": "left-inner",
            "measurement_x": left_x,
            "target_x": TARGET_LEFT,
            "lane_width": None,
        }

    return {
        "mode": "search",
        "measurement_x": None,
        "target_x": None,
        "lane_width": None,
    }


def compute_speed(steering):
    reduction = min(abs(steering) * SPEED_REDUCTION_GAIN, BASE_SPEED - MIN_SPEED)
    return int(BASE_SPEED - reduction)


def limit_slew(target_value, previous_value, max_step):
    lower = previous_value - max_step
    upper = previous_value + max_step
    return int(np.clip(target_value, lower, upper))


def draw_fit_line(frame, coeffs, color, y_start, y_end):
    if coeffs is None:
        return

    y_values = np.linspace(y_start, y_end, 12).astype(int)
    points = []
    for y in y_values:
        x = int(np.polyval(coeffs, y))
        if 0 <= x < FRAME_WIDTH:
            points.append((x, y))

    if len(points) >= 2:
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, color, 2)


def draw_debug(frame, mask, edges, lane_state, measurement, roi_top, speed, steering, lane_lock_active):
    overlay = frame.copy()
    control_y = lane_state["control_y"]
    servo_state = get_servo_monitor_state()

    cv2.rectangle(overlay, (0, roi_top), (FRAME_WIDTH - 1, FRAME_HEIGHT - 1), (80, 80, 80), 1)
    cv2.line(overlay, (0, control_y), (FRAME_WIDTH, control_y), (80, 80, 80), 1)

    for row in lane_state["debug_rows"]:
        y = row["y"]
        cv2.line(overlay, (0, y), (FRAME_WIDTH, y), (50, 50, 50), 1)

        if row["left_segment"] is not None:
            seg = row["left_segment"]
            cv2.line(overlay, (seg[0], y), (seg[1], y), (0, 255, 255), 1)
            cv2.circle(overlay, (seg[1], y), 3, (0, 140, 255), -1)

        if row["right_segment"] is not None:
            seg = row["right_segment"]
            cv2.line(overlay, (seg[0], y), (seg[1], y), (0, 255, 255), 1)
            cv2.circle(overlay, (seg[0], y), 3, (0, 255, 0), -1)

    draw_fit_line(overlay, lane_state["left_fit"], (0, 140, 255), roi_top, control_y)
    draw_fit_line(overlay, lane_state["right_fit"], (0, 255, 0), roi_top, control_y)

    if lane_state["left_x"] is not None:
        cv2.circle(overlay, (lane_state["left_x"], control_y), 5, (0, 140, 255), -1)
        cv2.line(overlay, (TARGET_LEFT, control_y - 10), (TARGET_LEFT, control_y + 10), (255, 255, 0), 2)

    if lane_state["right_x"] is not None:
        cv2.circle(overlay, (lane_state["right_x"], control_y), 5, (0, 255, 0), -1)
        cv2.line(overlay, (TARGET_RIGHT, control_y - 10), (TARGET_RIGHT, control_y + 10), (255, 255, 0), 2)

    cv2.line(overlay, (TARGET_CENTER, roi_top), (TARGET_CENTER, FRAME_HEIGHT), (255, 0, 0), 1)

    if measurement["measurement_x"] is not None:
        cv2.circle(overlay, (measurement["measurement_x"], control_y), 6, (0, 0, 255), -1)

    cv2.putText(overlay, f"Mode: {measurement['mode']}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
    cv2.putText(overlay, f"Speed: {speed}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
    cv2.putText(overlay, f"Steer: {int(steering)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
    cv2.putText(
        overlay,
        f"Lane lock: {'ON' if lane_lock_active else 'WAIT'}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 255, 20),
        1,
    )

    if measurement["lane_width"] is not None:
        cv2.putText(
            overlay,
            f"Lane width: {int(measurement['lane_width'])}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 255, 20),
            1,
        )

    cv2.putText(
        overlay,
        f"Servo L: {servo_state['left_us']}us ({servo_state['left_duty']:.2f}%)",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
    )
    cv2.putText(
        overlay,
        f"Servo R: {servo_state['right_us']}us ({servo_state['right_duty']:.2f}%)",
        (10, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
    )

    return overlay, mask, edges


def get_camera_frame(picam2, cap):
    if picam2:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cap.read()
        if not ret:
            return None

    if CAMERA_FLIP_CODE is not None:
        frame = cv2.flip(frame, CAMERA_FLIP_CODE)

    return frame


def _request_stop(signum=None, frame=None):
    global running
    shutdown_motors(hold_seconds=0.2, release=True, close_gpio_handle=False)
    running = False


def main():
    global running

    if "QT_QPA_PLATFORM" in os.environ:
        del os.environ["QT_QPA_PLATFORM"]

    for sig_name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _request_stop)
        except ValueError:
            pass

    picam2 = None
    cap = None

    try:
        from picamera2 import Picamera2

        picam2 = Picamera2()
        sensor_res = picam2.sensor_resolution
        scale = 800 / sensor_res[0] if sensor_res[0] > 800 else 1.0
        config = picam2.create_preview_configuration(
            main={
                "size": (int(sensor_res[0] * scale), int(sensor_res[1] * scale)),
                "format": "RGB888",
            }
        )
        picam2.configure(config)
        picam2.start()
    except Exception:
        pipeline = "libcamerasrc ! video/x-raw, width=800, height=600, framerate=30/1 ! videoconvert ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not picam2 and (not cap or not cap.isOpened()):
        print("Cannot open camera")
        return

    running = True
    release()

    print("Camera preview started. Press 's' to start lane following, 'q' to quit.")
    preview_mode = True

    while preview_mode and running:
        frame = get_camera_frame(picam2, cap)
        if frame is None:
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        cv2.putText(frame, "Press 's' to start", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Camera Preview", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("s"):
            stop(hold_seconds=0.2, release=False)
            preview_mode = False
        elif key == ord("q"):
            running = False
            preview_mode = False

    cv2.destroyWindow("Camera Preview")

    if not running:
        print("Stopped by user request.")
        shutdown_motors(hold_seconds=0.2, release=True, close_gpio_handle=True)
        if picam2:
            picam2.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        return

    print("Starting lane following using the inner edge of the white lane.")

    prev_error = 0.0
    integral = 0.0
    prev_steering = 0
    lost_frames = 0
    lane_lock_active = not WAIT_FOR_INITIAL_LANE_LOCK
    try:
        while running:
            frame = get_camera_frame(picam2, cap)
            if frame is None:
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            mask, roi_top = build_lane_mask(frame)
            edges = cv2.Canny(mask, 50, 150)
            lane_state = collect_inner_lane_points(mask)
            measurement = evaluate_lane_measurement(lane_state)

            speed = 0
            steering = prev_steering

            if measurement["measurement_x"] is not None:
                if not lane_lock_active:
                    lane_lock_active = True
                    prev_error = 0.0
                    integral = 0.0
                    prev_steering = 0

                error = measurement["measurement_x"] - measurement["target_x"]
                if abs(error) <= DEADBAND_PX:
                    error = 0

                integral = float(np.clip(integral + error, -INTEGRAL_LIMIT, INTEGRAL_LIMIT))
                derivative = error - prev_error

                raw_steering = -(
                    (Kp * error)
                    + (Kd * derivative)
                    + (Ki * integral)
                )
                steering = limit_slew(raw_steering, prev_steering, STEERING_SLEW_LIMIT)
                speed = compute_speed(steering)
                drive_motor(speed, steering)
                log_servo_monitor(prefix=f"lane_following_hd:{measurement['mode']}")

                prev_error = error
                prev_steering = steering
                lost_frames = 0
            else:
                integral = 0.0
                prev_error = 0.0
                lost_frames += 1

                if not lane_lock_active:
                    speed = 0
                    steering = 0
                    measurement = {**measurement, "mode": "waiting-lock"}
                    release_motor_output()
                elif lost_frames <= LOST_HOLD_FRAMES:
                    steering = 0
                    speed = 0
                    measurement = {**measurement, "mode": "lost-hold"}
                    release_motor_output()
                else:
                    ramp_frames = min(
                        lost_frames - LOST_HOLD_FRAMES,
                        LOST_RIGHT_TURN_RAMP_FRAMES,
                    )
                    turn_ratio = ramp_frames / LOST_RIGHT_TURN_RAMP_FRAMES
                    right_turn = LOST_RIGHT_TURN_MIN + (
                        (LOST_RIGHT_TURN_MAX - LOST_RIGHT_TURN_MIN) * turn_ratio
                    )
                    steering = -int(right_turn)
                    speed = 0
                    measurement = {**measurement, "mode": "lost-right-search"}

                prev_steering = steering
                if measurement["mode"] == "lost-right-search":
                    drive_motor(speed, steering)
                log_servo_monitor(prefix=f"lane_following_hd:{measurement['mode']}")

            debug_frame, debug_mask, debug_edges = draw_debug(
                frame,
                mask,
                edges,
                lane_state,
                measurement,
                roi_top,
                speed,
                steering,
                lane_lock_active,
            )

            cv2.imshow("Lane following inner line", debug_frame)
            cv2.imshow("Lane mask", debug_mask)
            cv2.imshow("Lane edge", debug_edges)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                running = False
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping the car and releasing resources...")
        shutdown_motors(hold_seconds=0.3, release=True, close_gpio_handle=True)
        if picam2:
            picam2.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
