import lgpio
from hardware.gpio_handle import gpio_open

LEFT_PIN, RIGHT_PIN = 12, 13
PWM_FREQ = 50
STOP_VAL_LEFT = 1500
STOP_VAL_RIGHT = 1500
DRIVE_SPEED = 300
TURN_SPEED = 200
TURN_TIME = 1.8
_claimed = False

def init_motors():
    global _claimed
    h = gpio_open()
    if not _claimed:
        try:
            lgpio.gpio_claim_output(h, LEFT_PIN)
            lgpio.gpio_claim_output(h, RIGHT_PIN)
            _claimed = True
        except lgpio.error:
            pass
    return h

def _set_pwm(pin, us):
    try:
        h = init_motors()
        duty = (us / 20000) * 100
        lgpio.tx_pwm(h, pin, PWM_FREQ, duty)
    except lgpio.error as e:
        if "not set as an output" in str(e).lower() or "not a pwm" in str(e).lower():
            h = gpio_open()
            try:
                lgpio.gpio_claim_output(h, pin)
                duty = (us / 20000) * 100
                lgpio.tx_pwm(h, pin, PWM_FREQ, duty)
            except:
                pass
        else:
            raise e

def move_straight():
    _set_pwm(LEFT_PIN, STOP_VAL_LEFT - DRIVE_SPEED)
    _set_pwm(RIGHT_PIN, STOP_VAL_RIGHT + DRIVE_SPEED)

def turn_right():
    _set_pwm(LEFT_PIN, STOP_VAL_LEFT + TURN_SPEED)
    _set_pwm(RIGHT_PIN, STOP_VAL_RIGHT + TURN_SPEED)

def turn_left():
    _set_pwm(LEFT_PIN, STOP_VAL_LEFT - TURN_SPEED)
    _set_pwm(RIGHT_PIN, STOP_VAL_RIGHT - TURN_SPEED)

def drive(speed, steering):
    """
    Drive with proportional steering.
    speed: base speed (0-400)
    steering: steering offset (-200 to 200)
    """
    steering = max(-200, min(200, steering))
    left_us = STOP_VAL_LEFT - speed - steering
    right_us = STOP_VAL_RIGHT + speed - steering
    _set_pwm(LEFT_PIN, left_us)
    _set_pwm(RIGHT_PIN, right_us)

def move_backward():
    _set_pwm(LEFT_PIN, STOP_VAL_LEFT + DRIVE_SPEED)
    _set_pwm(RIGHT_PIN, STOP_VAL_RIGHT - DRIVE_SPEED)

def stop():
    _set_pwm(LEFT_PIN, 0)
    _set_pwm(RIGHT_PIN, 0)
