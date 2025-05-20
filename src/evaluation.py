import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def load_ground_truth(image_path: str, annotations_dir: str) -> str:
    """Load ground truth plate number for an image.

    Args:
        image_path: Path to the image file
        annotations_dir: Directory containing annotation files

    Returns:
        Ground truth plate number
    """
    # Get image name without extension
    image_name = Path(image_path).stem

    # Find corresponding annotation file
    ann_file = Path(annotations_dir) / f"{image_name}.txt"

    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    # Read annotation file and extract plate number
    with open(ann_file, "r") as f:
        for line in f:
            if line.startswith("plate:"):
                return line.split(":")[1].strip()

    raise ValueError(f"No plate annotation found in {ann_file}")


def sample_dataset(dataset_dir: str, n_samples: int = 100) -> Dict[str, List[str]]:
    """Sample images from each category in the dataset.

    Args:
        dataset_dir: Root directory of the dataset
        n_samples: Number of samples to take from each category

    Returns:
        Dictionary mapping category names to lists of image paths
    """
    samples = {}
    images_dir = Path(dataset_dir) / "images"

    # Get all category directories
    categories = [d for d in images_dir.iterdir() if d.is_dir()]

    for category in categories:
        # Get all image files in category
        image_files = list(category.glob("*.jpg"))

        # Sample n_samples or all if less available
        n_available = len(image_files)
        n_to_sample = min(n_samples, n_available)

        if n_to_sample < n_samples:
            logger.warning(
                f"Only {n_available} images available in {category.name}, "
                f"using all instead of requested {n_samples}"
            )

        samples[category.name] = random.sample(
            [str(f) for f in image_files], n_to_sample
        )

    return samples


def evaluate_processor(
    processor,
    dataset_dir: str,
    n_samples: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Evaluate processor on dataset.

    Args:
        processor: LicensePlateProcessor instance
        dataset_dir: Root directory of the dataset
        n_samples: Number of samples to evaluate per category

    Returns:
        Dictionary with evaluation metrics per category
    """
    # Sample images from dataset
    samples = sample_dataset(dataset_dir, n_samples)

    results = {}
    for category, image_paths in samples.items():
        logger.info(f"\nEvaluating {category}...")

        n_correct = 0
        n_total = len(image_paths)
        n_detected = 0

        for i, image_path in enumerate(image_paths, 1):
            try:
                # Get ground truth
                gt_plate = load_ground_truth(
                    image_path, os.path.join(dataset_dir, "annotations")
                )

                # Run processor
                pred_results = processor.process_image(image_path)

                if pred_results:  # If any plates were detected
                    n_detected += 1
                    # Check if any of the detected plates match the ground truth
                    for pred_plate in pred_results.keys():
                        if pred_plate.lower() == gt_plate.lower():
                            n_correct += 1
                            break

            except Exception as e:
                logger.debug(
                    f"Error processing {image_path}: {e}"
                )  # Reduced to debug level
                continue

            # Show progress
            if i % 10 == 0:
                logger.info(f"Processed {i}/{n_total} images")

        # Calculate metrics
        detection_rate = n_detected / n_total if n_total > 0 else 0
        accuracy = n_correct / n_total if n_total > 0 else 0

        results[category] = {
            "detection_rate": detection_rate,
            "accuracy": accuracy,
            "n_samples": n_total,
            "n_detected": n_detected,
            "n_correct": n_correct,
        }

    return results


def find_annotation_file(image_path: str, dataset_dir: str) -> str:
    """Find the annotation file for a given image.

    Args:
        image_path: Path to the image file (from split.txt)
        dataset_dir: Root directory of the dataset

    Returns:
        Path to the annotation file
    """
    # Remove './' from the start if present
    if image_path.startswith("./"):
        image_path = image_path[2:]

    # Get the full path to the image
    full_image_path = Path(dataset_dir) / image_path

    # Get image directory and name
    image_dir = full_image_path.parent
    image_name = full_image_path.stem

    # Look for annotation file with the same name
    ann_file = image_dir / f"{image_name}.txt"

    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    return str(ann_file)


def read_plate_number(ann_file: str) -> str:
    """Read the plate number from an annotation file.

    Args:
        ann_file: Path to the annotation file

    Returns:
        The plate number
    """
    with open(ann_file, "r") as f:
        for line in f:
            if line.startswith("plate:"):
                return line.split(":")[1].strip()
    raise ValueError(f"No plate number found in {ann_file}")


def prepare_annotations(dataset_dir: str):
    """Prepare dataset annotations by extracting plate numbers from original annotations.

    Args:
        dataset_dir: Root directory of the dataset
    """
    # Create annotations directory
    annotations_dir = Path(dataset_dir) / "annotations"
    annotations_dir.mkdir(exist_ok=True)

    # Read split.txt
    split_file = Path(dataset_dir) / "split.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    # Process each line
    n_processed = 0
    n_errors = 0

    with open(split_file, "r") as f:
        for line in f:
            try:
                # Get image path and split type
                image_path, split_type = line.strip().split(";")

                # Find original annotation file
                ann_file = find_annotation_file(image_path, dataset_dir)

                # Read plate number
                plate_number = read_plate_number(ann_file)

                # Get image name without extension
                image_name = Path(image_path).stem

                # Create new annotation file
                new_ann_file = annotations_dir / f"{image_name}.txt"

                # Write plate number to new annotation file
                with open(new_ann_file, "w") as af:
                    af.write(f"plate: {plate_number}\n")

                n_processed += 1

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                n_errors += 1
                continue

    logger.info(f"Processed {n_processed} annotations successfully")
    if n_errors > 0:
        logger.warning(f"Encountered {n_errors} errors during processing")
