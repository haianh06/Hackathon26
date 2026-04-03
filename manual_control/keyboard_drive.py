import sys

import pygame

try:
    from gpio_handle import gpio_close
    from servo import format_servo_monitor, drive, release, stop
except ImportError:
    from manual_control.gpio_handle import gpio_close
    from manual_control.servo import format_servo_monitor, drive, release, stop

WINDOW_WIDTH = 560
WINDOW_HEIGHT = 260
FORWARD_SPEED = 70
REVERSE_SPEED = 60
TURN_SPEED = 70
LOOP_DELAY = 0.02


def compute_command(keys):
    up = keys[pygame.K_UP]
    down = keys[pygame.K_DOWN]
    left = keys[pygame.K_LEFT]
    right = keys[pygame.K_RIGHT]

    speed = 0
    steering = 0
    mode = "RELEASED"

    if up and not down:
        speed = FORWARD_SPEED
        mode = "FORWARD"
    elif down and not up:
        speed = -REVERSE_SPEED
        mode = "REVERSE"

    if left and not right:
        steering = -TURN_SPEED
        mode = "LEFT" if speed == 0 else f"{mode} + LEFT"
    elif right and not left:
        steering = TURN_SPEED
        mode = "RIGHT" if speed == 0 else f"{mode} + RIGHT"

    return speed, steering, mode


def draw_ui(screen, font, mode, speed, steering):
    screen.fill((18, 22, 30))

    lines = [
        "Arrow Keys Drive",
        "UP/DOWN: forward/reverse",
        "LEFT/RIGHT: rotate or steer",
        "SPACE: hold calibrated neutral",
        "No arrow key: release PWM",
        "Q or ESC: quit",
        f"Mode: {mode}",
        f"Speed: {speed}",
        f"Steering: {steering}",
        format_servo_monitor(prefix="PWM"),
    ]

    for idx, text in enumerate(lines):
        color = (240, 240, 240) if idx < 6 else (120, 255, 160)
        surface = font.render(text, True, color)
        screen.blit(surface, (24, 18 + idx * 22))

    pygame.display.flip()


def main():
    pygame.init()
    pygame.display.set_caption("Keyboard Drive")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    font = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()
    target_fps = int(1 / LOOP_DELAY)

    release()

    mode = "RELEASED"
    speed = 0
    steering = 0
    neutral_hold_active = False
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        speed = 0
                        steering = 0
                        mode = "NEUTRAL"
                        neutral_hold_active = True
                        stop()

            keys = pygame.key.get_pressed()
            speed, steering, computed_mode = compute_command(keys)

            if speed == 0 and steering == 0:
                if neutral_hold_active:
                    mode = "NEUTRAL"
                    stop()
                else:
                    release()
                    mode = "RELEASED"
            else:
                neutral_hold_active = False
                drive(speed=speed, steering=steering)
                mode = computed_mode

            draw_ui(screen, font, mode, speed, steering)
            clock.tick(target_fps)

    except KeyboardInterrupt:
        pass
    finally:
        stop(hold_seconds=0.2, release=True)
        gpio_close()
        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Keyboard drive failed: {exc}", file=sys.stderr)
        raise
