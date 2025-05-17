import logging
import numpy as np
from inference_sdk import InferenceHTTPClient
import supervision as sv
from .base import LicensePlateDetector

logger = logging.getLogger(__name__)


class SDKLicensePlateDetector(LicensePlateDetector):
    """Roboflow SDK implementation of license plate detection."""

    def __init__(
        self,
        api_key: str,
        model_id: str,
        api_url: str = "https://serverless.roboflow.com",
    ):
        """Initialize SDK detector.

        Args:
            api_key: Roboflow API key
            model_id: Model ID to use
            api_url: API URL (optional)
        """
        logger.info("Initializing SDK detector")
        try:
            self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
            self.model_id = model_id
        except Exception as e:
            logger.error(f"Failed to initialize SDK detector: {e}")
            raise

    def detect_plates(self, image) -> sv.Detections:
        """Detect license plates using Roboflow SDK.

        Args:
            image: Input image path or numpy array

        Returns:
            sv.Detections: Detected license plates
        """
        try:
            results = self.client.infer(image, model_id=self.model_id)
            # Convert SDK results to sv.Detections format
            predictions = results.get("predictions", [])
            boxes = []
            confidences = []
            class_ids = []

            for pred in predictions:
                x, y, width, height = (
                    pred["x"],
                    pred["y"],
                    pred["width"],
                    pred["height"],
                )
                x1 = x - width / 2
                y1 = y - height / 2
                x2 = x + width / 2
                y2 = y + height / 2
                boxes.append([x1, y1, x2, y2])
                confidences.append(pred.get("confidence", 1.0))
                class_ids.append(0)  # Assuming single class for license plates

            return sv.Detections(
                xyxy=np.array(boxes),
                confidence=np.array(confidences),
                class_id=np.array(class_ids),
            )
        except Exception as e:
            logger.error(f"Error during SDK detection: {e}")
            raise
