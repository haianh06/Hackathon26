import cv2
import threading
import time
import os

try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except Exception:
    HAS_PICAMERA = False

class CameraManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CameraManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.frame = None
        self.resized_frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.picam2 = None  # Persistent camera objects
        self.cap = None
        self._initialized = True

    def start(self):
        with self.lock:
            if self.running and self.thread and self.thread.is_alive():
                return
            self.running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            print("✅ CameraManager: Thread started")

    def _run_loop(self):
        """Background thread that owns the camera lifecycle."""
        print("CameraManager: Loop entering...")
        
        while self.running:
            # 1. Attempt Picamera2
            if HAS_PICAMERA and self.picam2 is None and self.cap is None:
                try:
                    print("CameraManager: Initializing Picamera2...")
                    self.picam2 = Picamera2()
                    
                    sensor_res = self.picam2.sensor_resolution
                    scale = 800 / sensor_res[0] if sensor_res[0] > 800 else 1.0
                    target_size = (int(sensor_res[0] * scale), int(sensor_res[1] * scale))
                    
                    config = self.picam2.create_preview_configuration(
                        main={"size": target_size, "format": "RGB888"}
                    )
                    self.picam2.configure(config)
                    try:
                        if "LensPosition" in self.picam2.controls:
                            self.picam2.set_controls({"LensPosition": 0.5})
                    except Exception:
                        pass
                    self.picam2.start()
                    time.sleep(1.0)
                    print(f"✅ CameraManager: Picamera2 ready ({target_size[0]}x{target_size[1]})")
                except Exception as e:
                    print(f"⚠️ CameraManager: Picamera2 failed: {e}")
                    if self.picam2:
                        try:
                            self.picam2.stop()
                            self.picam2.close()
                        except: pass
                        self.picam2 = None
                    time.sleep(2.0)

            # 2. Fallback to OpenCV
            if not self.picam2 and self.cap is None:
                print("CameraManager: Initializing OpenCV...")
                indices = [0, 1, 2, 4, 6]
                for idx in indices:
                    c = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                    if c.isOpened():
                        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        ret, _ = c.read()
                        if ret:
                            self.cap = c
                            print(f"✅ CameraManager: OpenCV ready at index {idx}")
                            break
                        c.release()
                
                if not self.cap:
                    c = cv2.VideoCapture(0)
                    if c.isOpened():
                        self.cap = c
                    else:
                        print("❌ CameraManager: All backends failed. Retrying in 3s...")
                        time.sleep(3.0)
                        continue

            # 3. Capture Loop
            try:
                raw_frame = None
                if self.picam2:
                    raw_frame = self.picam2.capture_array()
                elif self.cap:
                    ret, frame = self.cap.read()
                    if ret:
                        raw_frame = frame
                    else:
                        print("⚠️ CameraManager: OpenCV frame lost. Resetting...")
                        self.cap.release()
                        self.cap = None
                        time.sleep(1.0)
                
                if raw_frame is not None:
                    # Pre-resize once for all consumers (e.g. 320x240)
                    small = cv2.resize(raw_frame, (320, 240))
                    with self.lock:
                        self.frame = raw_frame
                        self.resized_frame = small
                
                time.sleep(0.01)
            except Exception as e:
                print(f"⚠️ CameraManager capture error: {e}")
                if self.picam2:
                    try: 
                        self.picam2.stop()
                        self.picam2.close()
                    except: pass
                    self.picam2 = None
                time.sleep(1.0)

        print("CameraManager: Shutting down...")
        self._cleanup()

    def _cleanup(self):
        if self.picam2:
            try: 
                self.picam2.stop()
                self.picam2.close()
            except: pass
            self.picam2 = None
        if self.cap:
            try: self.cap.release()
            except: pass
            self.cap = None

    def get_frame(self, copy=True):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy() if copy else self.frame
            return None

    def get_frame_resized(self, size=(320, 240), copy=True):
        """Returns a cached resized frame if size matches, else resizes on demand."""
        with self.lock:
            if self.frame is None:
                return None
            if size == (320, 240) and self.resized_frame is not None:
                return self.resized_frame.copy() if copy else self.resized_frame
            return cv2.resize(self.frame, size)

    def stop(self):
        with self.lock:
            self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

camera_manager = CameraManager()
