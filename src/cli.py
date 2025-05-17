import os
from typing import Dict, List, Optional, Tuple
import typer
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from enum import Enum
from typing_extensions import Annotated
from dotenv import load_dotenv
import pathlib
from .detectors.local_detector import LocalLicensePlateDetector
from .detectors.sdk_detector import SDKLicensePlateDetector
from .processor import LicensePlateProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
console = Console()

# Create CLI app
app = typer.Typer(
    help="License Plate Detection and Recognition CLI", add_completion=False
)


class DetectorType(str, Enum):
    """Enum for detector types."""

    LOCAL = "local"
    SDK = "sdk"


def process_directory(
    processor: LicensePlateProcessor,
    directory: str,
    recursive: bool = False,
    extensions: List[str] = [".jpg", ".jpeg", ".png"],
) -> Dict[str, Dict[str, Tuple[int, int, int, int]]]:
    """Process all images in a directory.

    Args:
        processor: LicensePlateProcessor instance
        directory: Directory path
        recursive: Whether to process subdirectories
        extensions: List of valid file extensions

    Returns:
        Dict mapping image paths to their results
    """
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
):
    """Detect and recognize license plates in images."""
    try:
        # Load configuration
        load_dotenv()

        # Get configuration from environment
        api_key = os.getenv("ROBOFLOW_API_KEY")
        model_id = os.getenv("MODEL_ID", "license-plates-xfeyr/1")
        plate_model = os.getenv("PLATE_MODEL", "global-plates-mobile-vit-v2-model")
        default_output = os.getenv("OUTPUT_DIR", "output")

        # Use provided output_dir or default
        output_directory = output_dir or default_output

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize detector
            progress.add_task(description="Initializing detector...", total=None)

            if detector_type == DetectorType.LOCAL:
                detector = LocalLicensePlateDetector(model_id=model_id)
            else:
                if not api_key:
                    console.print("[red]Error: API key required for SDK detector[/red]")
                    raise typer.Exit(1)
                detector = SDKLicensePlateDetector(api_key=api_key, model_id=model_id)

            # Initialize processor
            processor = LicensePlateProcessor(
                detector=detector,
                plate_model=plate_model,
                output_dir=output_directory if save_result else None,
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
    console.print("License Plate Detection and Recognition v1.0.0")
