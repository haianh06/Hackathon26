import sys

import pygame

try:
    from gpio_handle import gpio_close
    from servo import (
        STOP_L,
        STOP_R,
        drive_us,
        format_servo_monitor,
        release,
        stop,
    )
except ImportError:
    from manual_control.gpio_handle import gpio_close
    from manual_control.servo import (
        STOP_L,
        STOP_R,
        drive_us,
        format_servo_monitor,
        release,
        stop,
    )

WINDOW_WIDTH = 760
WINDOW_HEIGHT = 340
FPS = 30
US_STEP = 5


def compute_command(keys, neutral_left, neutral_right, test_speed):
    left_us = neutral_left
    right_us = neutral_right
    mode = "NEUTRAL"

    if keys[pygame.K_UP]:
        left_us = neutral_left - test_speed
        right_us = neutral_right + test_speed
        mode = "FORWARD"
    elif keys[pygame.K_DOWN]:
        left_us = neutral_left + test_speed
        right_us = neutral_right - test_speed
        mode = "REVERSE"
    elif keys[pygame.K_LEFT]:
        left_us = neutral_left + test_speed
        right_us = neutral_right + test_speed
        mode = "ROTATE LEFT"
    elif keys[pygame.K_RIGHT]:
        left_us = neutral_left - test_speed
        right_us = neutral_right - test_speed
        mode = "ROTATE RIGHT"

    return left_us, right_us, mode


def draw_ui(screen, font, small_font, neutral_left, neutral_right, test_speed, mode):
    screen.fill((18, 22, 30))

    lines = [
        "Servo Tune",
        "Keep the car lifted off the ground before testing.",
        "Arrow keys: apply motion test",
        "[ / ]: adjust left neutral by 5us",
        "; / ': adjust right neutral by 5us",
        "1/2/3/4: set test speed to 40/60/80/100us",
        "SPACE: hold neutral",
        "R: release PWM",
        "Q or ESC: quit",
        f"Neutral Left:  {neutral_left}us",
        f"Neutral Right: {neutral_right}us",
        f"Test Speed:    {test_speed}us",
        f"Mode:          {mode}",
        format_servo_monitor(prefix="PWM"),
    ]

    for idx, text in enumerate(lines):
        color = (245, 245, 245) if idx < 9 else (120, 255, 160)
        line_font = font if idx == 0 else small_font
        surface = line_font.render(text, True, color)
        screen.blit(surface, (24, 18 + idx * 22))

    hint = "Adjust neutral until both wheels stay still at SPACE."
    hint_surface = small_font.render(hint, True, (255, 220, 120))
    screen.blit(hint_surface, (24, WINDOW_HEIGHT - 34))
    pygame.display.flip()


def main():
    pygame.init()
    pygame.display.set_caption("Servo Tune")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    font = pygame.font.SysFont(None, 34)
    small_font = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()

    neutral_left = STOP_L
    neutral_right = STOP_R
    test_speed = 60
    mode = "RELEASED"

    release()

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_LEFTBRACKET:
                        neutral_left -= US_STEP
                    elif event.key == pygame.K_RIGHTBRACKET:
                        neutral_left += US_STEP
                    elif event.key == pygame.K_SEMICOLON:
                        neutral_right -= US_STEP
                    elif event.key == pygame.K_QUOTE:
                        neutral_right += US_STEP
                    elif event.key == pygame.K_1:
                        test_speed = 40
                    elif event.key == pygame.K_2:
                        test_speed = 60
                    elif event.key == pygame.K_3:
                        test_speed = 80
                    elif event.key == pygame.K_4:
                        test_speed = 100
                    elif event.key == pygame.K_SPACE:
                        drive_us(neutral_left, neutral_right)
                        mode = "NEUTRAL"
                    elif event.key == pygame.K_r:
                        release()
                        mode = "RELEASED"

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_DOWN] or keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
                left_us, right_us, mode = compute_command(keys, neutral_left, neutral_right, test_speed)
                drive_us(left_us, right_us)
            elif mode not in ("RELEASED", "NEUTRAL"):
                drive_us(neutral_left, neutral_right)
                mode = "NEUTRAL"

            draw_ui(screen, font, small_font, neutral_left, neutral_right, test_speed, mode)
            clock.tick(FPS)

    except KeyboardInterrupt:
        pass
    finally:
        stop(hold_seconds=0.2, release=True)
        gpio_close()
        pygame.quit()
        print(f"Suggested STOP_L = {neutral_left}")
        print(f"Suggested STOP_R = {neutral_right}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Servo tune failed: {exc}", file=sys.stderr)
        raise
