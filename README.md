# License Plate Recognition System

A Python-based system for detecting and recognizing license plates from images using deep learning and computer vision techniques.

## Features

- Detects license plates in images (using Roboflow inference)
- Recognizes plate numbers using OCR
- Annotates images with detected plates and their numbers
- Uses CPU execution for inference
- Modular architecture for easy model switching

## Inference Implementation

The system currently uses Roboflow inference for license plate detection, but this can be easily modified by changing the implementation of the `detect_plates` function. The current implementation uses a Roboflow model with ID `license-plates-xfeyr/1`, but you can replace it with any other detection model by modifying the `model = get_model()` line in `main.py`.

To switch to a different detection model:
1. Update the `model_id` parameter in the `get_model()` call
2. Ensure the new model outputs detections in a compatible format
3. The rest of the pipeline (OCR and annotation) remains unchanged

This modular design allows you to experiment with different detection models without affecting the OCR or annotation components.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MaxiLR/LicensePlates-ANPR.git
cd LicensePlates-ANPR
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your test image in the project directory as `test.jpg`
2. Run the script:
```bash
python main.py
```

The script will:
1. Detect license plates in the image
2. Recognize the plate numbers
3. Annotate the image with bounding boxes and plate numbers
4. Display the annotated image

## Requirements

- Python 3.8 or higher
- OpenCV for image processing
- Supervision for detection and annotation
- Fast Plate OCR for license plate recognition
- ONNX Runtime for model inference

## Environment Variables

The project uses environment variables for configuration. Create a `.env` file based on `.env.example`:

```
# .env
# Add any required environment variables here
```

## Dependencies

The project uses the following major dependencies:
- fast-plate-ocr
- supervision
- numpy
- python-dotenv
- opencv-python
- onnxruntime

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the creators of Fast Plate OCR for the plate recognition model

## Support

For support, please open an issue in the GitHub repository.