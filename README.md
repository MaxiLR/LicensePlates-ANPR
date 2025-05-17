# License Plate Detection and Recognition

This project provides a flexible implementation for detecting and recognizing license plates in images using either local processing or the Roboflow API.

## Features

- Support for both local and Roboflow SDK-based detection
- Batch processing capabilities
- Configurable output options (save/show results)
- Comprehensive error handling and logging
- Easy to extend with new detection implementations
- Command Line Interface (CLI) for easy usage
- Support for processing single images or entire directories
- Rich console output with progress indicators

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/LicensePlates-ANPR.git
cd LicensePlates-ANPR
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your configuration:

```bash
ROBOFLOW_API_KEY=your_api_key_here [required]
ROBOFLOW_API_URL=https://serverless.roboflow.com [optional]
MODEL_ID=license-plates-xfeyr/1 [optional]
PLATE_MODEL=global-plates-mobile-vit-v2-model [optional]
OUTPUT_DIR=output [optional]
```

## Usage

### Command Line Interface (CLI)

The easiest way to use the tool is through its CLI:

1. Process a single image:

```bash
# Using local detector (default)
python main.py detect image.jpg

# Using SDK detector
python main.py detect image.jpg --detector-type sdk

# Show results in window
python main.py detect image.jpg --show-result

# Specify output directory
python main.py detect image.jpg --output-dir results
```

2. Process a directory of images:

```bash
# Process all images in a directory
python main.py detect ./images/

# Process directory recursively
python main.py detect ./images/ --recursive

# Process with SDK and custom output
python main.py detect ./images/ --detector-type sdk --output-dir results
```

Available options:

```bash
python main.py detect --help
```

```
Options:
  --detector-type [local|sdk]  Type of detector to use
  --recursive                  Process subdirectories if input is a directory
  --show-result               Show results in window
  --save-result               Save annotated images
  --output-dir TEXT           Output directory for annotated images
  --help                      Show this message and exit.
```

### Programmatic Usage

You can also use the library programmatically:

#### Basic Usage

```python
from main import LocalLicensePlateDetector, LicensePlateProcessor

# Initialize detector and processor
detector = LocalLicensePlateDetector(model_id="license-plates-xfeyr/1")
processor = LicensePlateProcessor(
    detector=detector,
    plate_model="global-plates-mobile-vit-v2-model",
    output_dir="output"
)

# Process a single image
results = processor.process_image(
    "test.jpg",
    show_result=True,
    save_result=True
)
```

#### Using Roboflow SDK

```python
from main import SDKLicensePlateDetector, LicensePlateProcessor

# Initialize SDK detector
detector = SDKLicensePlateDetector(
    api_key="your_api_key",
    model_id="license-plates-xfeyr/1"
)
processor = LicensePlateProcessor(
    detector=detector,
    plate_model="global-plates-mobile-vit-v2-model",
    output_dir="output"
)

# Process an image
results = processor.process_image("test.jpg")
```

#### Batch Processing

```python
# Process multiple images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = processor.process_batch(
    image_paths,
    show_results=False,
    save_results=True
)
```

## Project Structure

- `main.py`: Main implementation with detector classes and processor
- `.env`: Configuration file (create from .env.example)
- `requirements.txt`: Project dependencies

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the creators of Fast Plate OCR for the plate recognition model
- Built with Typer for the CLI interface
- Uses Rich for beautiful console output

## Support

For support, please open an issue in the GitHub repository.
