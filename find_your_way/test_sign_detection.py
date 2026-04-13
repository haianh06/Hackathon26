import cv2
import numpy as np
import os
import sys
from utils.traffic_sign_recognition import TrafficSignRecognition

def test_tsr():
    tsr = TrafficSignRecognition(
        config_path="find_your_way/data/sign_config.json",
        templates_dir="find_your_way/data/templates"
    )
    tsr.circularity_threshold = 0.5
    tsr.confidence_threshold = 0.4
    
    # Create a dummy frame with a blue circle
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (160, 120), 40, (255, 0, 0), -1) # Blue circle
    
    # Add some text to simulate a sign (matches our placeholder 'up.png')
    cv2.putText(frame, "UP", (145, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    print("Testing detection...")
    results = tsr.detect_and_classify(frame)
    if results:
        for res in results:
            print(f"Found: {res['label']} with confidence {res['confidence']:.2f}")
    else:
        print("No signs detected.")

if __name__ == "__main__":
    test_tsr()
