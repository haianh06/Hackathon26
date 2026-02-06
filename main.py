#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time

from servo import joystick_loop, stop
from sonar import get_distance
from rfid import RFIDReader
from gpio_handle import gpio_close

running = True
rfid = RFIDReader()

# shared data
latest_distance = None
distance_lock = threading.Lock()

# ================= SONAR THREAD =================
def sonar_task():
    global latest_distance
    while running:
        d = get_distance()
        if d is not None:
            with distance_lock:
                latest_distance = d

            print(f"📏 Distance: {d:.1f} cm")

            if d < 20:
                print("⚠️ Obstacle detected – STOP")
                stop()

        time.sleep(0.3)

# ================= RFID THREAD =================
def rfid_task():
    while running:
        uid = rfid.read_uid_hex(timeout=0.5)
        if uid:
            with distance_lock:
                d = latest_distance

            if d is not None:
                print(f"🆔 UID: {uid} | 📏 Distance: {d:.1f} cm")
            else:
                print(f"🆔 UID: {uid} | 📏 Distance: N/A")

            time.sleep(1)  # tránh spam khi giữ thẻ

# ================= MAIN =================
try:
    t1 = threading.Thread(target=sonar_task, daemon=True)
    t2 = threading.Thread(target=rfid_task, daemon=True)

    t1.start()
    t2.start()

    joystick_loop()   # blocking loop

except KeyboardInterrupt:
    pass

finally:
    running = False
    stop()
    rfid.cleanup()
    gpio_close()
    print("🛑 System shutdown")
