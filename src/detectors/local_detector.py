import logging
from inference import get_model
import supervision as sv
from .base import LicensePlateDetector

logger = logging.getLogger(__name__)


class LocalLicensePlateDetector(LicensePlateDetector):
    """Local implementation of license plate detection using inference package."""

    def __init__(self, model_id: str):
        """Initialize local detector.

        Args:
            model_id: The model ID to use for detection
        """
        logger.info(f"Initializing local detector with model {model_id}")
        try:
            self.model = get_model(model_id=model_id)
        except Exception as e:
            logger.error(f"Failed to initialize local detector: {e}")
            raise

    def detect_plates(self, image) -> sv.Detections:
        """Detect license plates using local model.

        Args:
            image: Input image

        Returns:
            sv.Detections: Detected license plates
        """
        try:
            results = self.model.infer(image)[0]
            return sv.Detections.from_inference(results)
        except Exception as e:
            logger.error(f"Error during local detection: {e}")
            raise
