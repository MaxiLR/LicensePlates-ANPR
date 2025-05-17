import os
import logging
import pathlib
from typing import Dict, List, Tuple, Optional
from .detectors.base import LicensePlateDetector

logger = logging.getLogger(__name__)


class LicensePlateProcessor:
    """Main class for processing license plate images."""

    def __init__(
        self,
        detector: LicensePlateDetector,
        plate_model: str,
        output_dir: Optional[str] = None,
    ):
        """Initialize the processor.

        Args:
            detector: License plate detector implementation
            plate_model: Model name for OCR
            output_dir: Directory to save results (optional)
        """
        self.detector = detector
        try:
            # Import heavy dependencies only when needed
            from fast_plate_ocr import ONNXPlateRecognizer

            self.plate_reader = ONNXPlateRecognizer(plate_model)
            if output_dir:
                pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.output_dir = output_dir
        except Exception as e:
            logger.error(f"Failed to initialize processor: {e}")
            raise

    def process_image(
        self, image_path: str, show_result: bool = False, save_result: bool = True
    ) -> Dict[str, Tuple[int, int, int, int]]:
        """Process an image to detect and read license plates.

        Args:
            image_path: Path to the image
            show_result: Whether to display the result
            save_result: Whether to save the annotated image

        Returns:
            Dict mapping plate text to bounding box coordinates
        """
        try:
            # Import heavy dependencies only when needed
            import cv2
            import supervision as sv

            logger.info(f"Processing image: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            detections = self.detector.detect_plates(image)
            labels = self._read_plates(image, detections)

            if save_result or show_result:
                annotated_image = self._annotate(image.copy(), labels)

                if save_result and self.output_dir:
                    output_path = os.path.join(
                        self.output_dir, f"annotated_{os.path.basename(image_path)}"
                    )
                    cv2.imwrite(output_path, annotated_image)
                    logger.info(f"Saved annotated image to: {output_path}")

                if show_result:
                    sv.plot_image(annotated_image)

            return labels

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise

    def process_batch(
        self,
        image_paths: List[str],
        show_results: bool = False,
        save_results: bool = True,
    ) -> Dict[str, Dict[str, Tuple[int, int, int, int]]]:
        """Process multiple images.

        Args:
            image_paths: List of image paths
            show_results: Whether to display results
            save_results: Whether to save annotated images

        Returns:
            Dict mapping image paths to their results
        """
        results = {}
        for image_path in image_paths:
            try:
                results[image_path] = self.process_image(
                    image_path, show_result=show_results, save_result=save_results
                )
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results[image_path] = None
        return results

    def _read_plates(self, image, detections) -> Dict[str, Tuple[int, int, int, int]]:
        """Read text from detected license plates.

        Args:
            image: Input image
            detections: Detected license plates

        Returns:
            Dict mapping plate text to coordinates
        """
        # Import heavy dependencies only when needed
        import cv2

        labels = {}
        for i in range(len(detections)):
            try:
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                plate_image = image[y1:y2, x1:x2]
                plate_image_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                plate_text = self.plate_reader.run(plate_image_gray)
                plate_text[0] = plate_text[0].replace("_", "")
                labels[plate_text[0]] = (x1, y1, x2, y2)
            except Exception as e:
                logger.warning(f"Failed to read plate {i}: {e}")
        return labels

    def _annotate(self, image, labels):
        """Annotate image with detected plates.

        Args:
            image: Input image
            labels: Dict of plate text and coordinates

        Returns:
            Annotated image
        """
        # Import heavy dependencies only when needed
        import numpy as np
        import supervision as sv

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        for label, (x1, y1, x2, y2) in labels.items():
            xyxy = np.array([[x1, y1, x2, y2]])
            class_id = np.array([0])
            detections = sv.Detections(xyxy=xyxy, class_id=class_id)
            image = bounding_box_annotator.annotate(scene=image, detections=detections)
            image = label_annotator.annotate(
                scene=image, detections=detections, labels=[label]
            )

        return image
