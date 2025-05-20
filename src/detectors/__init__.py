from .base import LicensePlateDetector
from .local_detector import LocalLicensePlateDetector
from .sdk_detector import SDKLicensePlateDetector
from .yolo_detector import YOLOv8LicensePlateDetector

__all__ = [
    "LicensePlateDetector",
    "LocalLicensePlateDetector",
    "SDKLicensePlateDetector",
    "YOLOv8LicensePlateDetector",
]
