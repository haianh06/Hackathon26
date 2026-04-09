import threading
import time
from picamera2 import Picamera2

class ThreadedCam:
    def __init__(self):
        self.frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def _loop(self):
        try:
            print("Thread started, initializing Picamera2...")
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"size": (320, 240), "format": "BGR888"})
            picam2.configure(config)
            print("Camera configured, starting...")
            picam2.start()
            time.sleep(1.0) # Grace period
            print("Camera ready")
            
            while self.running:
                try:
                    frame = picam2.capture_array()
                    with self.lock:
                        self.frame = frame
                    time.sleep(0.01)
                except Exception as e:
                    print(f"Capture error: {e}")
                    time.sleep(0.1)
            
            print("Stopping camera...")
            picam2.stop()
        except Exception as e:
            print(f"Thread failed: {e}")
            import traceback
            traceback.print_exc()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

tc = ThreadedCam()
try:
    tc.start()
    for i in range(10):
        time.sleep(1)
        with tc.lock:
            if tc.frame is not None:
                print(f"Captured frame {i}, shape: {tc.frame.shape}")
            else:
                print(f"No frame {i} yet")
    tc.stop()
    print("Done")
except Exception as e:
    print(f"Main error: {e}")
