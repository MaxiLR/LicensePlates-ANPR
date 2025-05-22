import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from src.cli import DetectorType, get_processor
from pathlib import Path

# Load environment variables for API key only
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

# Available models configuration
MODELS = {
    "license-plate-recognition-rxg4e-wyhgr": ["1", "3"],
    "license-plates-v2-m5o4y": ["1"],
}

st.title("License Plate Detection and Recognition")
st.write(
    "An automatic license plate recognition (ALPR) system using custom implementation."
)

# Sidebar for selecting detector type
detector_type = st.sidebar.selectbox(
    "Choose Detector Type",
    [DetectorType.LOCAL, DetectorType.SDK, DetectorType.YOLO],
    format_func=lambda x: x.value.upper(),
)

# Model selection in sidebar
model_name = st.sidebar.selectbox("Choose Model", list(MODELS.keys()))

version = st.sidebar.selectbox("Choose Version", MODELS[model_name])

# Show API key input if needed
if detector_type in [DetectorType.LOCAL, DetectorType.SDK]:
    input_api_key = st.sidebar.text_input(
        "Roboflow API Key",
        value=api_key if api_key else "",
        type="password",
    )
    if not input_api_key:
        st.warning("Please enter your Roboflow API Key in the sidebar.")
        st.stop()
    api_key = input_api_key

# Configure model paths and IDs based on selection
if detector_type == DetectorType.YOLO:
    weights_path = f"models/{model_name}/{version}/weights.onnx"
    if not Path(weights_path).exists():
        st.error(f"Weights file not found at {weights_path}")
        st.stop()
else:
    weights_path = None

model_id = f"{model_name}/{version}"

# Initialize processor with selected detector
processor = None
with st.spinner(
    f"Loading {detector_type.value.upper()} detector and models... This might take a few seconds."
):
    try:
        processor = get_processor(
            detector_type=detector_type,
            model_id=model_id,
            api_key=api_key,
            plate_model="global-plates-mobile-vit-v2-model",  # Using default OCR model
            weights_path=weights_path,
        )
    except Exception as e:
        st.error(f"Error initializing processor: {str(e)}")
        st.stop()

if not processor:
    st.stop()

# Load image
uploaded_file = st.file_uploader(
    "Upload an image of a vehicle with a license plate", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Create a temporary file to save the uploaded image
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / uploaded_file.name

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Display original image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Process image
    st.write("Processing...")
    try:
        results = processor.process_image(
            str(temp_path), show_result=False, save_result=False
        )

        # Display results
        if results:
            # Read and annotate image
            image = cv2.imread(str(temp_path))
            annotated_image = processor._annotate(image.copy(), results)
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            # Show annotated image
            st.image(
                annotated_image,
                caption="Annotated Image with Detection Results",
                use_container_width=True,
            )

            # Display detection results
            st.write("**Detection Results:**")
            for plate_text, coords in results.items():
                st.write(f"- Detected Plate: `{plate_text}`")
        else:
            st.write("No license plates detected 😔")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

    # Clean up temporary file
    temp_path.unlink()
else:
    st.write("Please upload an image to continue.")

# Add footer with information
st.markdown("---")
st.markdown(
    f"""
    This application uses a custom ALPR implementation with multiple detector options:
    - **LOCAL**: Uses Roboflow's local inference
    - **SDK**: Uses Roboflow's cloud API
    - **YOLO**: Uses local YOLOv8 model
    
    Currently using:
    - Model: `{model_name}`
    - Version: `{version}`
    - Detector: `{detector_type.value.upper()}`
    """
)
