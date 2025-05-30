import logging
from .base import LicensePlateDetector

logger = logging.getLogger(__name__)


class LocalLicensePlateDetector(LicensePlateDetector):
    """Local implementation of license plate detection using inference package."""

    def __init__(self, model_id: str, api_key: str):
        """Initialize local detector.

        Args:
            model_id: The model ID to use for detection
        """
        logger.info(f"Initializing local detector with model {model_id}")
        try:
            # Import heavy dependencies only when needed
            from inference import get_model

            self.model = get_model(model_id=model_id, api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize local detector: {e}")
            raise

    def detect_plates(self, image):
        """Detect license plates using local model.

        Args:
            image: Input image

        Returns:
            sv.Detections: Detected license plates
        """
        try:
            # Import heavy dependencies only when needed
            import supervision as sv

            results = self.model.infer(image)[0]
            return sv.Detections.from_inference(results)
        except Exception as e:
            logger.error(f"Error during local detection: {e}")
            raise
