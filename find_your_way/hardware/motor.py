import lgpio
from hardware.gpio_handle import gpio_open

LEFT_PIN, RIGHT_PIN = 12, 13
PWM_FREQ = 50
STOP_VAL = 1500
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
    h = init_motors()
    duty = (us / 20000) * 100
    lgpio.tx_pwm(h, pin, PWM_FREQ, duty)

def move_straight():
    _set_pwm(LEFT_PIN, STOP_VAL - DRIVE_SPEED)
    _set_pwm(RIGHT_PIN, STOP_VAL + DRIVE_SPEED)

def turn_right():
    _set_pwm(LEFT_PIN, STOP_VAL + TURN_SPEED)
    _set_pwm(RIGHT_PIN, STOP_VAL + TURN_SPEED)

def turn_left():
    _set_pwm(LEFT_PIN, STOP_VAL - TURN_SPEED)
    _set_pwm(RIGHT_PIN, STOP_VAL - TURN_SPEED)

def stop():
    _set_pwm(LEFT_PIN, 0)
    _set_pwm(RIGHT_PIN, 0)
