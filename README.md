# License Plate Detection and Recognition

## 🚀 Quick Start - Real-time Webcam Detection

For **instant real-time license plate detection** using your webcam:

```bash
# Run the streaming web application
streamlit run app_stream.py
```

Then open your browser at `http://localhost:8502` and click **"Start Live Detection"** to begin real-time license plate recognition with your webcam!

**Features:**

- 🎥 Live webcam streaming with real-time detection
- ⚡ YOLOv8 model with ~200ms inference speed
- 🎯 Automatic license plate annotations
- ⚙️ Adjustable confidence and performance settings
- 📊 Live FPS counter and detection statistics

---

This project provides a flexible implementation for detecting and recognizing license plates in images using either local processing or the Roboflow API.

## Features

- Support for YOLOv8, local and Roboflow SDK-based detection
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
git clone https://github.com/MaxiLR/LicensePlates-ANPR.git
cd LicensePlates-ANPR
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your configuration:

```bash
ROBOFLOW_API_KEY=your_api_key_here [required for SDK/LOCAL]
ROBOFLOW_API_URL=https://serverless.roboflow.com [optional]
MODEL_ID=license-plates-xfeyr/1 [optional]
PLATE_MODEL=global-plates-mobile-vit-v2-model [optional]
OUTPUT_DIR=output [optional]
WEIGHTS_PATH=models/license-plate-recognition-rxg4e-wyhgr/1/weights.onnx [required for YOLO]
```

## Usage

### Command Line Interface (CLI)

The easiest way to use the tool is through its CLI:

1. Process a single image:

```bash
# Using YOLO detector
python main.py detect image.jpg --detector-type yolo --weights-path models/license-plate-recognition-rxg4e-wyhgr/1/weights.onnx

# Using local detector
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
# Process with YOLO detector
python main.py detect ./images/ --detector-type yolo --weights-path models/license-plate-recognition-rxg4e-wyhgr/1/weights.onnx

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
  --detector-type [yolo|local|sdk]  Type of detector to use
  --weights-path TEXT               Path to YOLO weights file (required for YOLO detector)
  --recursive                       Process subdirectories if input is a directory
  --show-result                     Show results in window
  --save-result                     Save annotated images
  --output-dir TEXT                 Output directory for annotated images
  --help                           Show this message and exit.
```

### Programmatic Usage

You can also use the library programmatically:

#### Basic Usage

```python
from main import YOLOv8LicensePlateDetector, LicensePlateProcessor

# Initialize detector and processor
detector = YOLOv8LicensePlateDetector(weights_path="models/license-plate-recognition-rxg4e-wyhgr/1/weights.onnx")
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

### Model Evaluation

The project includes tools for evaluating license plate detection models using the RodoSol-ALPR dataset. The evaluation process consists of two steps:

1. Prepare the dataset:

```bash
python main.py eval-prepare RodoSol-ALPR-dataset.zip
```

This command will:

- Extract the RodoSol-ALPR dataset from the zip file
- Create the necessary annotation files
- Set up the directory structure for evaluation

2. Run the evaluation:

```bash
python main.py eval RodoSol-ALPR/ --detector-type yolo --weights-path models/license-plate-recognition-rxg4e-wyhgr/1/weights.onnx --n-samples 200
```

Options for evaluation:

- `--detector-type`: Type of detector to use (`local`, `sdk`, or `yolo`)
- `--weights-path`: Path to YOLO weights file (required for YOLO detector)
- `--n-samples`: Number of samples to evaluate per category (default: 100)

The evaluation will:

- Sample images from each category (cars-br, cars-me, motorcycles-br, motorcycles-me)
- Run detection and recognition on the samples
- Calculate and display metrics including:
  - Detection Rate: Percentage of images where a plate was detected
  - Accuracy: Percentage of correctly read plate numbers
  - Number of samples processed
  - Number of plates detected
  - Number of correct readings

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
