import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

class TrafficSignRecognition:
    """
    Integrated Traffic Sign Recognition system for autonomous driving.
    Handles detection of blue circular signs and classification using template matching.
    """
    
    def __init__(self, config_path: str = "find_your_way/data/sign_config.json", 
                 templates_dir: str = "find_your_way/data/templates"):
        self.config_path = config_path
        self.templates_dir = Path(templates_dir)
        self.load_config()
        self.load_templates()

    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_path}, using defaults. Error: {e}")
            self.config = {
                "detector": {
                    "hsv_lower": [90, 50, 50],
                    "hsv_upper": [130, 255, 255],
                    "min_area": 500,
                    "max_area": 50000,
                    "circularity_threshold": 0.7
                },
                "classifier": {
                    "template_size": [64, 64],
                    "template_threshold": 0.5
                }
            }
        
        # Detector params
        det = self.config["detector"]
        self.hsv_lower = np.array(det["hsv_lower"], dtype=np.uint8)
        self.hsv_upper = np.array(det["hsv_upper"], dtype=np.uint8)
        self.min_area = det["min_area"]
        self.max_area = det["max_area"]
        self.circularity_threshold = det["circularity_threshold"]
        
        # Classifier params
        cls_cfg = self.config["classifier"]
        self.template_size = tuple(cls_cfg["template_size"])
        self.confidence_threshold = cls_cfg["template_threshold"]

    def save_config(self):
        """Save current configuration to file."""
        self.config["detector"]["hsv_lower"] = self.hsv_lower.tolist()
        self.config["detector"]["hsv_upper"] = self.hsv_upper.tolist()
        self.config["detector"]["min_area"] = self.min_area
        self.config["detector"]["max_area"] = self.max_area
        self.config["detector"]["circularity_threshold"] = self.circularity_threshold
        self.config["classifier"]["template_size"] = list(self.template_size)
        self.config["classifier"]["template_threshold"] = self.confidence_threshold
        
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def load_templates(self):
        """Load templates from directory."""
        self.templates = {}
        template_mapping = {
            'straight': 'up.png',
            'left': 'left.png',
            'right': 'right.png',
            'parking': 'p.png'
        }
        
        if not self.templates_dir.exists():
            print(f"Warning: Templates directory {self.templates_dir} not found.")
            return

        for sign_type, filename in template_mapping.items():
            path = self.templates_dir / filename
            if path.exists():
                img = cv2.imread(str(path))
                if img is not None:
                    self.templates[sign_type] = cv2.resize(img, self.template_size)
                else:
                    print(f"Warning: Failed to load template {path}")
            else:
                self.templates[sign_type] = None

    def detect_and_classify(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and classify signs in a single frame.
        Returns a list of detections with bbox, label, and confidence.
        """
        if frame is None:
            return []

        # 1. Detection
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            if circularity < self.circularity_threshold:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # ROI Extraction
            padding = int(0.1 * max(w, h))
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame.shape[1], x + w + padding)
            y_end = min(frame.shape[0], y + h + padding)
            roi = frame[y_start:y_end, x_start:x_end]
            
            if roi.size == 0: continue
            
            # 2. Classification
            roi_resized = cv2.resize(roi, self.template_size)
            best_match = "unknown"
            best_confidence = 0.0
            
            for sign_type, template in self.templates.items():
                if template is None: continue
                
                res = cv2.matchTemplate(roi_resized, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_match = sign_type
            
            if best_confidence >= self.confidence_threshold:
                results.append({
                    "bbox": (x, y, w, h),
                    "label": best_match,
                    "confidence": float(best_confidence)
                })
                
        return results

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on the frame."""
        annotated = frame.copy()
        for det in detections:
            x, y, w, h = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(annotated, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return annotated
