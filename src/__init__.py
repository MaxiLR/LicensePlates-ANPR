from .processor import LicensePlateProcessor
from .detectors.local_detector import LocalLicensePlateDetector
from .detectors.sdk_detector import SDKLicensePlateDetector

__version__ = "1.0.0"

__all__ = [
    "LicensePlateProcessor",
    "LocalLicensePlateDetector",
    "SDKLicensePlateDetector",
]
