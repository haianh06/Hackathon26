"""
Sign Detector Module

This module contains the SignDetector class which is responsible for detecting
blue circular traffic signs in an image using HSV color space and contour analysis.
"""

import cv2
import numpy as np
from typing import List, Tuple


class SignDetector:
    """Detects blue circular traffic signs using optimized HSV and contours."""
    
    def __init__(self, 
                 hsv_lower: Tuple[int, int, int] = (90, 50, 50),
                 hsv_upper: Tuple[int, int, int] = (130, 255, 255),
                 min_area: int = 400,
                 max_area: int = 40000,
                 circularity_threshold: float = 0.35):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = min_area
        self.max_area = max_area
        self.circularity_threshold = circularity_threshold
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing with CLAHE normalization to handle varying lighting.
        """
        # 1. Convert to YUV to apply CLAHE only on the Y (intensity) channel
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        
        # 2. Convert back to RGB for subsequent processing
        normalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        
        # 3. Fast blur to reduce noise
        blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        return hsv
    
    def fast_blue_check(self, image: np.ndarray) -> bool:
        """
        Very fast check to see if any blue exists in the frame.
        Uses a downsampled image for speed.
        """
        # Resize to a tiny thumbnail for a quick scan
        small = cv2.resize(image, (80, 60), interpolation=cv2.INTER_NEAREST)
        hsv_small = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_small, self.hsv_lower, self.hsv_upper)
        return cv2.countNonZero(mask) > 10 # At least 10 "blue" pixels in a 80x60 grid

    def create_blue_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """
        Create a binary mask for blue colors.
        """
        mask = cv2.inRange(hsv_image, self.hsv_lower, self.hsv_upper)
        
        # Simplified morphology for speed
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def calculate_circularity(self, area: float, perimeter: float) -> float:
        """Calculate circularity: 4π * Area / Perimeter²"""
        if perimeter == 0:
            return 0.0
        return 4 * np.pi * area / (perimeter ** 2)
    
    def detect_signs(self, image: np.ndarray, use_roi: bool = True) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect blue circular signs with ROI optimization and fast-path check.
        """
        h, w = image.shape[:2]
        
        # 1. Fast existence check
        if not self.fast_blue_check(image):
            return []
            
        # 2. ROI Cropping (Optional but recommended for speed)
        # Usually signs are in the top-right or top-left. We focus on top 2/3 of frame.
        # If use_roi is True, we only look at the upper 60% of the image.
        roi_y_end = int(h * 0.65) if use_roi else h
        roi_img = image[:roi_y_end, :]
        
        # 3. Process the ROI
        hsv_roi = self.preprocess(roi_img)
        mask = self.create_blue_mask(hsv_roi)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_signs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            circularity = self.calculate_circularity(area, perimeter)
            if circularity < self.circularity_threshold:
                continue
            
            # Get bounding box in ROI coordinates
            rx, ry, rw, rh = cv2.boundingRect(contour)
            
            # Extract ROI with padding from the ORIGINAL FULL image
            # (Need to adjust Y coordinate because it was from roi_img)
            padding = int(0.15 * max(rw, rh))
            
            x_start = max(0, rx - padding)
            y_start = max(0, ry - padding)
            x_end = min(image.shape[1], rx + rw + padding)
            y_end = min(image.shape[0], ry + rh + padding)
            
            final_roi = image[y_start:y_end, x_start:x_end]
            
            # Return ROI and original coordinates (x, y, w, h)
            detected_signs.append((final_roi, (rx, ry, rw, rh)))
        
        return detected_signs

    def get_detection_with_mask(self, image: np.ndarray) -> Tuple[List[Tuple[np.ndarray, Tuple[int, int, int, int]]], np.ndarray]:
        """Debugging version that returns mask."""
        hsv = self.preprocess(image)
        mask = self.create_blue_mask(hsv)
        dets = self.detect_signs(image, use_roi=False)
        return dets, mask
