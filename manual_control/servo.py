import time

import lgpio
import pygame

try:
    from gpio_handle import gpio_close, gpio_open
except ImportError:
    from manual_control.gpio_handle import gpio_close, gpio_open

LEFT_PIN = 12
RIGHT_PIN = 13
PWM_FREQ = 50

# Neutral pulse widths should be calibrated per wheel.
STOP_L = 1500
STOP_R = 1500

DRIVE_SPEED = 300
TURN_SPEED = 200

MONITOR_INTERVAL = 0.25

h = None
_monitor_state = {
    "left_us": 0,
    "right_us": 0,
    "left_duty": 0.0,
    "right_duty": 0.0,
    "updated_at": 0.0,
    "last_print_at": 0.0,
}


def _claim_output_pins(handle):
    lgpio.gpio_claim_output(handle, LEFT_PIN)
    lgpio.gpio_claim_output(handle, RIGHT_PIN)


def _ensure_gpio_ready():
    global h
    if h is None:
        h = gpio_open()
        if h is not None:
            _claim_output_pins(h)
    return h


def _update_monitor(pin, us):
    duty = 0.0 if us == 0 else (us / 20000.0) * 100.0
    now = time.time()

    if pin == LEFT_PIN:
        _monitor_state["left_us"] = int(us)
        _monitor_state["left_duty"] = duty
    elif pin == RIGHT_PIN:
        _monitor_state["right_us"] = int(us)
        _monitor_state["right_duty"] = duty

    _monitor_state["updated_at"] = now


def _set_pwm(pin, us):
    try:
        handle = _ensure_gpio_ready()
        if handle is not None:
            duty = 0.0 if us == 0 else (us / 20000.0) * 100.0
            lgpio.tx_pwm(handle, pin, PWM_FREQ, duty)
            _update_monitor(pin, us)
    except Exception:
        pass


def get_servo_monitor_state():
    return dict(_monitor_state)


def format_servo_monitor(prefix="Servo"):
    state = get_servo_monitor_state()
    return (
        f"{prefix} L={state['left_us']}us ({state['left_duty']:.2f}%) "
        f"R={state['right_us']}us ({state['right_duty']:.2f}%)"
    )


def log_servo_monitor(prefix="Servo", force=False, min_interval=MONITOR_INTERVAL):
    now = time.time()
    if force or (now - _monitor_state["last_print_at"] >= min_interval):
        print(format_servo_monitor(prefix))
        _monitor_state["last_print_at"] = now


def drive(speed, steering):
    left_us = STOP_L - int(speed) - int(steering)
    right_us = STOP_R + int(speed) - int(steering)
    _set_pwm(LEFT_PIN, left_us)
    _set_pwm(RIGHT_PIN, right_us)
    return left_us, right_us


def drive_us(left_us, right_us):
    _set_pwm(LEFT_PIN, int(left_us))
    _set_pwm(RIGHT_PIN, int(right_us))
    return int(left_us), int(right_us)


def release():
    _set_pwm(LEFT_PIN, 0)
    _set_pwm(RIGHT_PIN, 0)


def stop(hold_seconds=0.0, release=False):
    _set_pwm(LEFT_PIN, STOP_L)
    _set_pwm(RIGHT_PIN, STOP_R)
    if hold_seconds > 0:
        time.sleep(hold_seconds)
    if release:
        release_motors()


def release_motors():
    _set_pwm(LEFT_PIN, 0)
    _set_pwm(RIGHT_PIN, 0)


def shutdown_motors(hold_seconds=0.0, release=False, close_gpio_handle=False):
    global h
    stop(hold_seconds=hold_seconds, release=release)
    if close_gpio_handle:
        gpio_close()
        h = None


try:
    _ensure_gpio_ready()
except Exception:
    h = None


def joystick_loop(exit_callback=None):
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No joystick found")

    joy = pygame.joystick.Joystick(0)
    joy.init()

    stop()
    print("Motor control active (B to exit)")
    import __main__

    while True:
        if hasattr(__main__, "running") and not __main__.running:
            break
        pygame.event.pump()

        if joy.get_button(1):
            stop()
            if exit_callback:
                exit_callback()
            break

        x = joy.get_axis(0)
        y = -joy.get_axis(1)

        if abs(x) < 0.2 and abs(y) < 0.2:
            stop()
        elif abs(y) > abs(x):
            if y > 0:
                _set_pwm(LEFT_PIN, STOP_L - DRIVE_SPEED)
                _set_pwm(RIGHT_PIN, STOP_R + DRIVE_SPEED + 100)
            else:
                _set_pwm(LEFT_PIN, STOP_L + DRIVE_SPEED)
                _set_pwm(RIGHT_PIN, STOP_R - DRIVE_SPEED)
        else:
            if x < 0:
                _set_pwm(LEFT_PIN, STOP_L - TURN_SPEED)
                _set_pwm(RIGHT_PIN, STOP_R - TURN_SPEED)
            else:
                _set_pwm(LEFT_PIN, STOP_L + TURN_SPEED)
                _set_pwm(RIGHT_PIN, STOP_R + TURN_SPEED)

        time.sleep(0.01)
