import cv2
import time

pipeline = "libcamerasrc ! video/x-raw, width=320, height=240, framerate=30/1 ! videoconvert ! appsink"
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Failed to open GStreamer pipeline")
else:
    print("✅ GStreamer pipeline opened")
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            print(f"Captured frame {i}, shape: {frame.shape}")
        else:
            print(f"Failed to capture frame {i}")
        time.sleep(0.5)
    cap.release()
