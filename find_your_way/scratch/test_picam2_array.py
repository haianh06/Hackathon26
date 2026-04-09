from picamera2 import Picamera2
import time

try:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 240), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()
    print("Camera started")
    for i in range(10):
        frame = picam2.capture_array()
        print(f"Captured frame {i}, shape: {frame.shape}")
        time.sleep(0.1)
    picam2.stop()
    print("Done")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
