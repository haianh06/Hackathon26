"""
Sign Classifier Module

This module contains the SignClassifier class which is responsible for classifying
detected blue circular traffic signs using Template Matching and HOG features.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class SignClassifier:
    """Classifies traffic signs using Template Matching and HOG."""
    
    def __init__(self, 
                 templates_dir: str = 'templates',
                 template_size: Tuple[int, int] = (32, 32),
                 template_matching_threshold: float = 0.65):
        self.templates_dir = Path(templates_dir)
        self.template_size = template_size
        self.threshold = template_matching_threshold
        
        # Template mapping: sign_type -> template_name
        self.template_mapping = {
            'straight': 'up.png',
            'left': 'left.png',
            'right': 'right.png',
            'parking': 'p.png'
        }
        
        self.templates = {}
        self.load_templates()
    
    def load_templates(self) -> None:
        """
        Load template images in grayscale.
        """
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {self.templates_dir}")
        
        for sign_type, template_name in self.template_mapping.items():
            template_path = self.templates_dir / template_name
            
            if not template_path.exists():
                print(f"Warning: Template file not found: {template_path}")
                self.templates[sign_type] = None
                continue
            
            # Load template image in grayscale
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"Warning: Failed to load template: {template_path}")
                self.templates[sign_type] = None
                continue
            
            # Resize and normalize template
            template_resized = cv2.resize(template, self.template_size)
            
            # Use CLAHE on template as well for consistency
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            template_norm = clahe.apply(template_resized)
            
            self.templates[sign_type] = template_norm
    
    def template_matching(self, roi: np.ndarray) -> Tuple[str, float]:
        """
        Classify a ROI using Grayscale Normalized Cross-Correlation.
        """
        # 1. Resize and Grayscale
        roi_resized = cv2.resize(roi, self.template_size)
        if len(roi_resized.shape) == 3:
            roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2GRAY)
        else:
            roi_gray = roi_resized
            
        # 2. Normalize ROI lighting (matching the template preprocessing)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        roi_norm = clahe.apply(roi_gray)
        
        best_match = None
        best_confidence = -1.0
        
        for sign_type, template in self.templates.items():
            if template is None:
                continue
            
            # Perform template matching on grayscale image
            # TM_CCOEFF_NORMED is excellent for lighting invariance
            result = cv2.matchTemplate(roi_norm, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            # Update best match
            if max_val > best_confidence:
                best_confidence = max_val
                best_match = sign_type
        
        return best_match, best_confidence
    
    def extract_hog_features(self, roi: np.ndarray) -> np.ndarray:
        """
        Extract Histogram of Oriented Gradients (HOG) features from ROI.
        
        HOG features are useful for detecting edges and shapes.
        Can be used with SVM or other classifiers for robust classification.
        
        Args:
            roi (np.ndarray): Region of Interest
            
        Returns:
            np.ndarray: HOG feature vector
        """
        # Resize ROI to standard size
        roi_resized = cv2.resize(roi, self.template_size)
        
        # Convert to grayscale if needed
        if len(roi_resized.shape) == 3:
            roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi_resized
        
        # Compute HOG features using OpenCV
        # Cell size: 8x8, Block size: 16x16, Block stride: 8x8
        hog = cv2.HOGDescriptor(
            (64, 64),           # winSize
            (16, 16),           # blockSize
            (8, 8),             # blockStride
            (8, 8),             # cellSize
            9,                  # nBins
            1,                  # derivAperture
            -1.0,               # winSigma
            0,                  # histogramNormType
            0.2,                # L2HysThreshold
            True                # gammaCorrection
        )
        
        # Compute HOG descriptor
        hog_features = hog.compute(roi_gray)
        
        return hog_features
    
    def classify_hog_features(self, roi: np.ndarray) -> Tuple[str, float]:
        """
        Classify a ROI using HOG features (skeleton implementation).
        
        This is a placeholder for HOG + SVM classification.
        In a complete implementation, you would train an SVM model on HOG features
        and use it for classification here.
        
        Args:
            roi (np.ndarray): Region of Interest to classify
            
        Returns:
            Tuple[str, float]: Classification result (sign_type, confidence)
        """
        # Extract HOG features
        hog_features = self.extract_hog_features(roi)
        
        # Placeholder: Return message indicating this needs training
        # In production, you would:
        # 1. Load pre-trained SVM model
        # 2. Use model.predict(hog_features)
        # 3. Return prediction and confidence scores
        
        return 'hog_not_trained', 0.0
    
    def classify(self, 
                 roi: np.ndarray, 
                 method: str = 'template_matching') -> Tuple[str, float]:
        """
        Classify a detected traffic sign.
        
        Args:
            roi (np.ndarray): Region of Interest to classify
            method (str): Classification method ('template_matching' or 'hog_features')
            
        Returns:
            Tuple[str, float]: Classification result (sign_type, confidence)
        """
        if method == 'template_matching':
            return self.template_matching(roi)
        elif method == 'hog_features':
            return self.classify_hog_features(roi)
        else:
            # Fallback to template matching
            return self.template_matching(roi)
    
    def batch_classify(self, 
                      rois: List[np.ndarray],
                      method: str = 'template_matching') -> List[Tuple[str, float]]:
        """
        Classify multiple ROIs.
        
        Args:
            rois (List[np.ndarray]): List of regions of interest
            method (str): Classification method
            
        Returns:
            List[Tuple[str, float]]: List of classification results
        """
        results = []
        for roi in rois:
            result = self.classify(roi, method=method)
            results.append(result)
        
        return results
    
    def classify_verbose(self, roi: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify with detailed output showing all template scores.
        
        Args:
            roi (np.ndarray): Region of Interest to classify
            
        Returns:
            Tuple containing:
                - Best match sign type
                - Best match confidence
                - Dictionary with all template scores
        """
        # Resize ROI to match template size
        roi_resized = cv2.resize(roi, self.template_size)
        if len(roi_resized.shape) == 3:
            roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2GRAY)
        else:
            roi_gray = roi_resized
            
        _, roi_binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        all_scores = {}
        best_match = None
        best_confidence = -1.0
        
        for sign_type, template in self.templates.items():
            if template is None:
                all_scores[sign_type] = 0.0
                continue
            
            # Perform template matching on binary image
            result = cv2.matchTemplate(roi_binary, template, cv2.TM_CCOEFF_NORMED)
            
            # Get the maximum correlation value
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            # Use absolute value to make it color-agnostic (handles white-on-dark AND dark-on-white)
            max_val = abs(max_val)
            
            all_scores[sign_type] = max_val
            
            # Update best match if this confidence is higher
            if max_val > best_confidence:
                best_confidence = max_val
                best_match = sign_type
        
        # Always return best match (no unknown classification)
        return best_match, best_confidence, all_scores
