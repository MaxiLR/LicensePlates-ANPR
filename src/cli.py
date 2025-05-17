import os
from enum import Enum
import typer
from typing_extensions import Annotated
from typing import Optional
from rich.console import Console

# Create CLI app
app = typer.Typer(
    help="License Plate Detection and Recognition CLI", add_completion=False
)

console = Console()


class DetectorType(str, Enum):
    """Enum for detector types."""

    LOCAL = "local"
    SDK = "sdk"


def init_logging():
    """Initialize logging configuration."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def get_processor(
    detector_type: DetectorType,
    model_id: str,
    api_key: str = None,
    plate_model: str = None,
    output_dir: Optional[str] = None,
):
    """Initialize detector and processor with lazy imports."""
    # Import heavy dependencies only when needed
    from .detectors.local_detector import LocalLicensePlateDetector
    from .detectors.sdk_detector import SDKLicensePlateDetector
    from .processor import LicensePlateProcessor

    if detector_type == DetectorType.LOCAL:
        detector = LocalLicensePlateDetector(model_id=model_id)
    else:
        if not api_key:
            console.print("[red]Error: API key required for SDK detector[/red]")
            raise typer.Exit(1)
        detector = SDKLicensePlateDetector(api_key=api_key, model_id=model_id)

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

        # Use provided output_dir or default
        output_directory = output_dir or default_output

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
