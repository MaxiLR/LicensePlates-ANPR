import os
import zipfile
from enum import Enum
import typer
from typing_extensions import Annotated
from typing import Optional
from rich.console import Console
from pathlib import Path

# Create CLI app
app = typer.Typer(
    help="License Plate Detection and Recognition CLI", add_completion=False
)

console = Console()


class DetectorType(str, Enum):
    """Enum for detector types."""

    LOCAL = "local"
    SDK = "sdk"
    YOLO = "yolo"


def init_logging():
    """Initialize logging configuration."""
    import logging

    # Set higher level for verbose loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("fast_plate_ocr").setLevel(logging.WARNING)
    logging.getLogger("src.processor").setLevel(logging.WARNING)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",  # Simplified format
    )
    return logging.getLogger(__name__)


def get_processor(
    detector_type: DetectorType,
    model_id: str,
    api_key: str = None,
    plate_model: str = None,
    output_dir: Optional[str] = None,
    weights_path: Optional[str] = None,
):
    """Initialize detector and processor with lazy imports."""
    # Import heavy dependencies only when needed
    from .detectors.local_detector import LocalLicensePlateDetector
    from .detectors.sdk_detector import SDKLicensePlateDetector
    from .detectors.yolo_detector import YOLOv8LicensePlateDetector
    from .processor import LicensePlateProcessor

    if detector_type in [DetectorType.LOCAL, DetectorType.SDK] and not api_key:
        console.print("[red]Error: API key required for SDK and LOCAL detectors[/red]")
        raise typer.Exit(1)

    if detector_type == DetectorType.YOLO and not weights_path:
        console.print("[red]Error: weights_path required for YOLO detector[/red]")
        raise typer.Exit(1)

    if detector_type == DetectorType.LOCAL:
        detector = LocalLicensePlateDetector(model_id=model_id, api_key=api_key)
    elif detector_type == DetectorType.SDK:
        detector = SDKLicensePlateDetector(api_key=api_key, model_id=model_id)
    else:  # YOLO
        detector = YOLOv8LicensePlateDetector(weights_path=weights_path)

    return LicensePlateProcessor(
        detector=detector, plate_model=plate_model, output_dir=output_dir
    )


def process_directory(processor, directory: str, recursive: bool = False):
    """Process all images in a directory."""
    import pathlib

    extensions = [".jpg", ".jpeg", ".png"]
    image_paths = []
    pattern = "**/*" if recursive else "*"

    for ext in extensions:
        image_paths.extend(
            [str(p) for p in pathlib.Path(directory).glob(f"{pattern}{ext}")]
        )

    return processor.process_batch(image_paths)


@app.command()
def detect(
    input_path: Annotated[str, typer.Argument(help="Path to image file or directory")],
    detector_type: Annotated[
        DetectorType, typer.Option(help="Type of detector to use")
    ] = DetectorType.LOCAL,
    recursive: Annotated[
        bool, typer.Option(help="Process subdirectories if input is a directory")
    ] = False,
    show_result: Annotated[bool, typer.Option(help="Show results in window")] = False,
    save_result: Annotated[bool, typer.Option(help="Save annotated images")] = True,
    output_dir: Annotated[
        Optional[str], typer.Option(help="Output directory for annotated images")
    ] = None,
    weights_path: Annotated[
        Optional[str],
        typer.Option(help="Path to YOLO weights file (required for YOLO detector)"),
    ] = None,
):
    """Detect and recognize license plates in images."""
    try:
        # Initialize logging
        logger = init_logging()

        # Import dotenv only when needed
        from dotenv import load_dotenv

        load_dotenv()

        # Get configuration from environment
        api_key = os.getenv("ROBOFLOW_API_KEY")
        model_id = os.getenv("MODEL_ID", "license-plates-xfeyr/1")
        plate_model = os.getenv("PLATE_MODEL", "global-plates-mobile-vit-v2-model")
        default_output = os.getenv("OUTPUT_DIR", "output")
        default_weights_path = os.getenv("WEIGHTS_PATH", "weights")

        # Use provided output_dir or default
        output_directory = output_dir or default_output

        # Use provided weights_path or default
        weights_path = weights_path or default_weights_path

        # Import progress only when needed
        from rich.progress import Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize detector and processor
            progress.add_task(description="Initializing detector...", total=None)

            processor = get_processor(
                detector_type=detector_type,
                model_id=model_id,
                api_key=api_key,
                plate_model=plate_model,
                output_dir=output_directory if save_result else None,
                weights_path=weights_path,
            )

            # Process input
            if os.path.isfile(input_path):
                progress.add_task(
                    description=f"Processing image: {input_path}", total=None
                )
                results = processor.process_image(
                    input_path, show_result=show_result, save_result=save_result
                )
                console.print(f"\nResults for {input_path}:")
                console.print(results)

            elif os.path.isdir(input_path):
                progress.add_task(description="Processing directory...", total=None)
                results = process_directory(
                    processor=processor, directory=input_path, recursive=recursive
                )
                console.print("\nResults:")
                for img_path, img_results in results.items():
                    if img_results:
                        console.print(f"\n{img_path}:")
                        console.print(img_results)
                    else:
                        console.print(
                            f"\n[yellow]No plates detected in {img_path}[/yellow]"
                        )

            else:
                console.print("[red]Error: Input path does not exist[/red]")
                raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from . import __version__

    console.print(f"License Plate Detection and Recognition v{__version__}")


@app.command()
def eval(
    dataset_dir: Annotated[
        str, typer.Argument(help="Path to RodoSol-ALPR dataset directory")
    ],
    detector_type: Annotated[
        DetectorType, typer.Option(help="Type of detector to use")
    ] = DetectorType.LOCAL,
    n_samples: Annotated[
        int, typer.Option(help="Number of samples to evaluate per category")
    ] = 100,
    weights_path: Annotated[
        Optional[str],
        typer.Option(help="Path to YOLO weights file (required for YOLO detector)"),
    ] = None,
):
    """Evaluate detector performance on RodoSol-ALPR dataset."""
    try:
        # Initialize logging
        logger = init_logging()

        # Import dotenv only when needed
        from dotenv import load_dotenv

        load_dotenv()

        # Get configuration from environment
        api_key = os.getenv("ROBOFLOW_API_KEY")
        model_id = os.getenv("MODEL_ID", "license-plates-xfeyr/1")
        plate_model = os.getenv("PLATE_MODEL", "global-plates-mobile-vit-v2-model")
        default_weights_path = os.getenv("WEIGHTS_PATH", "weights")

        # Use provided weights_path or default
        weights_path = weights_path or default_weights_path

        # Import dependencies
        from rich.table import Table
        from .evaluation import evaluate_processor

        console.print("\nInitializing detector...")

        processor = get_processor(
            detector_type=detector_type,
            model_id=model_id,
            api_key=api_key,
            plate_model=plate_model,
            weights_path=weights_path,
        )

        # Run evaluation
        results = evaluate_processor(
            processor=processor,
            dataset_dir=dataset_dir,
            n_samples=n_samples,
        )

        # Create results table
        table = Table(title="\nEvaluation Results")
        table.add_column("Category")
        table.add_column("Detection Rate")
        table.add_column("Accuracy")
        table.add_column("Samples")
        table.add_column("Detected")
        table.add_column("Correct")

        # Add results to table
        for category, metrics in results.items():
            table.add_row(
                category,
                f"{metrics['detection_rate']:.2%}",
                f"{metrics['accuracy']:.2%}",
                str(metrics["n_samples"]),
                str(metrics["n_detected"]),
                str(metrics["n_correct"]),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def eval_prepare(
    dataset_zip: Annotated[
        str, typer.Argument(help="Path to RodoSol-ALPR dataset zip file")
    ],
):
    """Prepare RodoSol-ALPR dataset by extracting and creating annotations."""
    try:
        # Initialize logging
        logger = init_logging()

        # Import dependencies
        from .evaluation import prepare_annotations

        # Determine output directory
        output_dir = Path(dataset_zip).parent / "RodoSol-ALPR"

        # Extract dataset if needed
        if not output_dir.exists() or not (output_dir / "images").exists():
            console.print(f"\nExtracting dataset to {output_dir}...")

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract zip file
            with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
                zip_ref.extractall(output_dir.parent)

            console.print("Dataset extracted successfully")
        else:
            console.print("\nDataset already extracted")

        # Prepare annotations
        console.print("\nPreparing annotations...")
        prepare_annotations(str(output_dir))
        console.print("Annotations prepared successfully")

        console.print("\n[green]Dataset is ready for evaluation![/green]")

    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)
