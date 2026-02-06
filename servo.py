import lgpio
import pygame
import time
from gpio_handle import gpio_open

LEFT_PIN  = 12
RIGHT_PIN = 13
PWM_FREQ  = 50

STOP_L = 1500
STOP_R = 1500

DRIVE_SPEED = 300
TURN_SPEED  = 200

h = gpio_open()
lgpio.gpio_claim_output(h, LEFT_PIN)
lgpio.gpio_claim_output(h, RIGHT_PIN)

def _set_pwm(pin, us):
    duty = (us / 20000) * 100
    lgpio.tx_pwm(h, pin, PWM_FREQ, duty)

def stop():
    _set_pwm(LEFT_PIN, 0)    
    _set_pwm(RIGHT_PIN, 0)

def joystick_loop(exit_callback=None):
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No joystick found")

    joy = pygame.joystick.Joystick(0)
    joy.init()

    stop()
    print("🎮 Motor control active (B to exit)")

    while True:
        pygame.event.pump()

        if joy.get_button(1):  # B
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
                _set_pwm(LEFT_PIN,  STOP_L + DRIVE_SPEED)
                _set_pwm(RIGHT_PIN, STOP_R - DRIVE_SPEED)
            else:
                _set_pwm(LEFT_PIN,  STOP_L - DRIVE_SPEED)
                _set_pwm(RIGHT_PIN, STOP_R + DRIVE_SPEED)
        else:
            if x < 0:
                _set_pwm(LEFT_PIN,  STOP_L - TURN_SPEED)
                _set_pwm(RIGHT_PIN, STOP_R - TURN_SPEED)
            else:
                _set_pwm(LEFT_PIN,  STOP_L + TURN_SPEED)
                _set_pwm(RIGHT_PIN, STOP_R + TURN_SPEED)

        time.sleep(0.01)
