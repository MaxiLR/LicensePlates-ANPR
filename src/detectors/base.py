from abc import ABC, abstractmethod
import supervision as sv


class LicensePlateDetector(ABC):
    """Abstract base class for license plate detection implementations."""

    @abstractmethod
    def detect_plates(self, image) -> sv.Detections:
        """Detect license plates in the given image.

        Args:
            image: The input image (numpy array)

        Returns:
            sv.Detections: Detected license plates
        """
        pass
