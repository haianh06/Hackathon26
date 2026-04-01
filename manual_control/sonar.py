import lgpio
import time
from gpio_handle import gpio_open

TRIG = 23
ECHO = 24

h = gpio_open()
lgpio.gpio_claim_output(h, TRIG)
lgpio.gpio_claim_input(h, ECHO)

def get_distance(timeout=0.03):
    lgpio.gpio_write(h, TRIG, 0)
    time.sleep(0.000002)
    lgpio.gpio_write(h, TRIG, 1)
    time.sleep(0.00001)
    lgpio.gpio_write(h, TRIG, 0)

    start_time = time.time()
    while lgpio.gpio_read(h, ECHO) == 0:
        if time.time() - start_time > timeout:
            return None
    start = time.time()

    while lgpio.gpio_read(h, ECHO) == 1:
        if time.time() - start > timeout:
            return None
    end = time.time()

    return (end - start) * 34300 / 2
