from picamera2 import Picamera2
import time
import threading

class TestCam:
    def __init__(self):
        self.picam2 = Picamera2()
        self.frame = None
        self.lock = threading.Lock()
        self.running = False

    def on_request_completed(self, request):
        with self.lock:
            self.frame = request.make_array("main")

    def start(self):
        config = self.picam2.create_preview_configuration(main={"size": (320, 240), "format": "BGR888"})
        self.picam2.configure(config)
        self.picam2.request_completed = self.on_request_completed
        self.picam2.start()
        self.running = True

    def stop(self):
        self.running = False
        self.picam2.stop()

tc = TestCam()
try:
    tc.start()
    print("Camera started with listener")
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
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
