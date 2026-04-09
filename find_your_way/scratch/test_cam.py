import cv2
import os

devices = [f"/dev/video{i}" for i in range(10)]
for dev in devices:
    if os.path.exists(dev):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Device {dev} is working!")
                cap.release()
                break
            cap.release()
        else:
            print(f"❌ Device {dev} could not be opened.")
else:
    print("No working device found.")
