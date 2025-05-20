import logging
import os
from pathlib import Path
import supervision as sv
from ultralytics import YOLO
from .base import LicensePlateDetector

logger = logging.getLogger(__name__)


class YOLOv8LicensePlateDetector(LicensePlateDetector):
    """YOLOv8 implementation of license plate detection."""

    def __init__(self, weights_path: str):
        """Initialize YOLOv8 detector.

        Args:
            weights_path: Full path to the ONNX weights file
        """
        logger.info(f"Initializing YOLOv8 detector with weights from {weights_path}")
        try:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found at {weights_path}")

            self.model = YOLO(str(weights_path))
            logger.info("YOLOv8 model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8 detector: {e}")
            raise

    def detect_plates(self, image):
        """Detect license plates using YOLOv8 model.

        Args:
            image: Input image

        Returns:
            sv.Detections: Detected license plates
        """
        try:
            # Run inference
            results = self.model(image)[0]

            # Convert YOLO results to supervision Detections format
            boxes = results.boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format
            confidence = results.boxes.conf.cpu().numpy()  # Get confidence scores
            class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Get class IDs

            return sv.Detections(xyxy=boxes, confidence=confidence, class_id=class_ids)

        except Exception as e:
            logger.error(f"Error during YOLOv8 detection: {e}")
            raise
