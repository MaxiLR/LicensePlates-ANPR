import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from src.cli import DetectorType, get_processor
from pathlib import Path
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Fixed configuration - only YOLO detector with specific model
MODEL_NAME = "license-plate-recognition-rxg4e-wyhgr"
VERSION = "3"
DETECTOR_TYPE = DetectorType.YOLO

st.set_page_config(
    page_title="Real-time License Plate Detection", page_icon="🚗", layout="wide"
)

st.title("🚗 Real-time License Plate Detection Stream")
st.write("Live webcam feed with automatic license plate recognition using YOLOv8")

# Configure model path
weights_path = f"models/{MODEL_NAME}/{VERSION}/weights.onnx"
if not Path(weights_path).exists():
    st.error(f"Weights file not found at {weights_path}")
    st.info(f"Expected path: {weights_path}")
    st.stop()

model_id = f"{MODEL_NAME}/{VERSION}"


# Initialize processor with YOLO detector
@st.cache_resource
def load_processor():
    """Cache the processor to avoid reloading on each interaction"""
    try:
        processor = get_processor(
            detector_type=DETECTOR_TYPE,
            model_id=model_id,
            api_key=None,  # Not needed for YOLO
            plate_model="global-plates-mobile-vit-v2-model",
            weights_path=weights_path,
        )
        return processor
    except Exception as e:
        st.error(f"Error initializing processor: {str(e)}")
        return None


# Load processor
with st.spinner("Loading YOLO detector and models... This might take a few seconds."):
    processor = load_processor()

if not processor:
    st.stop()

# Sidebar controls
with st.sidebar:
    st.title("📹 Stream Controls")
    st.info(
        f"""
    **Model Configuration:**
    - Model: `{MODEL_NAME}`
    - Version: `{VERSION}`
    - Detector: `YOLO`
    """
    )

    # Basic controls (always visible)
    st.subheader("🎛️ Basic Controls")

    # Detection confidence threshold
    confidence_threshold = st.slider(
        "Detection Confidence", min_value=0.1, max_value=1.0, value=0.5, step=0.05
    )

    # Camera device selection
    camera_device = st.number_input(
        "Camera Device ID",
        min_value=0,
        max_value=5,
        value=0,
        help="Try different values if camera doesn't work (usually 0)",
    )

    camera_resolution = st.selectbox(
        "Camera Resolution", ["640x480", "800x600", "1280x720"], index=0
    )

    # Advanced controls (collapsible)
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.write("**Performance Settings**")

        process_every_n_frames = st.slider(
            "Process every N frames",
            min_value=1,
            max_value=15,
            value=5,
            help="Higher values = better performance but fewer detections",
        )

        show_fps = st.checkbox("Show FPS", value=True)

        # Performance options
        use_threading = st.checkbox("Use Threading (Recommended)", value=True)

        buffer_size = st.slider(
            "Camera Buffer Size",
            min_value=1,
            max_value=3,
            value=1,
            help="Lower = less latency but may drop frames",
        )

        # Annotation persistence control
        annotation_persistence = st.slider(
            "Annotation Persistence (seconds)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="How long to keep annotations visible after detection",
        )

# Main content area with two columns
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🎥 Live Camera Feed")

    # Streaming controls
    start_stream = st.button("🚀 Start Live Detection", type="primary")
    stop_stream = st.button("⏹️ Stop Stream", type="secondary")

    # Video placeholder
    video_placeholder = st.empty()

with col2:
    st.subheader("📋 Live Results")
    results_placeholder = st.empty()
    fps_placeholder = st.empty()

# Initialize session state for stream control
if "streaming" not in st.session_state:
    st.session_state.streaming = False

if "last_results" not in st.session_state:
    st.session_state.last_results = {}

if "last_detection_time" not in st.session_state:
    st.session_state.last_detection_time = 0

if "processing_queue" not in st.session_state:
    st.session_state.processing_queue = queue.Queue(maxsize=2)

if start_stream:
    st.session_state.streaming = True

if stop_stream:
    st.session_state.streaming = False


# Optimized processing function
def process_frame_async(frame, frame_id):
    """Process frame asynchronously directly from memory without saving to disk"""
    try:
        # Process directly using the YOLO detector
        detections = processor.detector.detect_plates(frame)

        # Filter by confidence threshold
        if hasattr(detections, "confidence") and len(detections.confidence) > 0:
            mask = detections.confidence >= confidence_threshold
            detections = detections[mask]

        if len(detections) == 0:
            return {}

        # Read plates directly from memory using the plate reader
        labels = {}
        for i in range(len(detections)):
            try:
                x1, y1, x2, y2 = map(int, detections.xyxy[i])

                # Validate bounding box
                if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
                    continue

                plate_image = frame[y1:y2, x1:x2]

                # Skip if plate image is too small
                if plate_image.shape[0] < 10 or plate_image.shape[1] < 10:
                    continue

                # Convert to grayscale for OCR
                plate_image_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

                # Use the plate reader directly
                plate_text = processor.plate_reader.run(plate_image_gray)

                # Clean and validate plate text
                if plate_text and len(plate_text) > 0:
                    cleaned_text = plate_text[0].replace("_", "").strip()
                    if len(cleaned_text) > 0:
                        labels[cleaned_text] = (x1, y1, x2, y2)
            except Exception as e:
                continue  # Skip failed plates

        return labels
    except Exception as e:
        return {}


# Main streaming function
if st.session_state.streaming:

    # Parse resolution
    width, height = map(int, camera_resolution.split("x"))

    try:
        cap = cv2.VideoCapture(int(camera_device))

        if not cap.isOpened():
            st.error(
                "❌ Cannot access webcam. Please check your camera permissions and device ID."
            )
            st.session_state.streaming = False
        else:
            # Optimize camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 60)  # Request higher FPS
            cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)  # Reduce buffer size
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

            # Additional optimizations for better performance
            cap.set(
                cv2.CAP_PROP_AUTO_EXPOSURE, 0.25
            )  # Reduce auto exposure for faster capture
            cap.set(
                cv2.CAP_PROP_AUTOFOCUS, 0
            )  # Disable autofocus for consistent performance

            st.success("✅ Camera connected successfully!")

            frame_count = 0
            start_time = time.time()
            fps = 0

            # Threading setup
            executor = ThreadPoolExecutor(max_workers=2) if use_threading else None
            processing_future = None

            # Stream loop
            while st.session_state.streaming:
                ret, frame = cap.read()

                if not ret:
                    st.error("❌ Failed to read from webcam")
                    break

                # Calculate FPS more efficiently (every 30 frames)
                if frame_count % 30 == 0 and frame_count > 0:
                    current_time = time.time()
                    fps = 30 / (current_time - start_time)
                    start_time = current_time

                # Process every N frames for detection
                detect_this_frame = frame_count % process_every_n_frames == 0

                if detect_this_frame:
                    if use_threading and executor:
                        # Check if previous processing is done
                        if processing_future is None or processing_future.done():
                            if processing_future and processing_future.done():
                                try:
                                    results = processing_future.result()
                                    if results:
                                        st.session_state.last_results = results
                                        st.session_state.last_detection_time = (
                                            time.time()
                                        )
                                except Exception as e:
                                    pass  # Ignore processing errors

                            # Start new processing
                            processing_future = executor.submit(
                                process_frame_async, frame.copy(), frame_count
                            )
                    else:
                        # Synchronous processing (fallback)
                        results = process_frame_async(frame.copy(), frame_count)
                        if results:
                            st.session_state.last_results = results
                            st.session_state.last_detection_time = time.time()

                # Clear annotations if no detections for the specified time
                current_time = time.time()
                if (
                    current_time - st.session_state.last_detection_time
                ) > annotation_persistence:
                    st.session_state.last_results = {}

                # Annotate frame (show annotations only if recent detections exist)
                if st.session_state.last_results:
                    try:
                        annotated_frame = processor._annotate(
                            frame.copy(), st.session_state.last_results
                        )
                    except:
                        annotated_frame = frame.copy()
                else:
                    annotated_frame = frame.copy()

                # Add FPS counter to frame if enabled
                if show_fps:
                    cv2.putText(
                        annotated_frame,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # Convert BGR to RGB for streamlit (only once)
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Display frame
                video_placeholder.image(
                    annotated_frame_rgb,
                    channels="RGB",
                    use_container_width=True,
                    caption=(
                        f"Frame {frame_count} | FPS: {fps:.1f}"
                        if show_fps
                        else f"Frame {frame_count}"
                    ),
                )

                # Update results display less frequently (every 15 frames for better performance)
                if frame_count % 15 == 0:
                    current_time = time.time()
                    time_since_detection = (
                        current_time - st.session_state.last_detection_time
                    )

                    if st.session_state.last_results:
                        with results_placeholder.container():
                            st.write("🎯 **Detected Plates:**")
                            for i, (plate_text, coords) in enumerate(
                                st.session_state.last_results.items(), 1
                            ):
                                st.success(f"**{i}.** `{plate_text}`")

                            st.write(f"📊 **Stats:**")
                            st.write(
                                f"- Total plates: {len(st.session_state.last_results)}"
                            )
                            st.write(f"- Frame: {frame_count}")
                            if show_fps:
                                st.write(f"- FPS: {fps:.1f}")

                            # Show time since last detection
                            if time_since_detection < annotation_persistence:
                                remaining_time = (
                                    annotation_persistence - time_since_detection
                                )
                                st.write(
                                    f"- ⏱️ Annotation expires in: {remaining_time:.1f}s"
                                )
                            else:
                                st.write("- ⏱️ Annotations expired")
                    else:
                        with results_placeholder.container():
                            st.info("🔍 Scanning for license plates...")
                            st.write(f"📊 **Stats:**")
                            st.write(f"- Frame: {frame_count}")
                            if show_fps:
                                st.write(f"- FPS: {fps:.1f}")
                            if st.session_state.last_detection_time > 0:
                                st.write(
                                    f"- ⏱️ Last detection: {time_since_detection:.1f}s ago"
                                )

                frame_count += 1

                # Minimal delay only if frame processing was very fast
                # This prevents overwhelming the system while maximizing FPS
                if frame_count % 3 == 0:  # Check every 3 frames to reduce overhead
                    time.sleep(0.001)  # Minimal delay for system stability

                # Safety check to prevent infinite loop
                if frame_count > 50000:
                    st.warning("Stream stopped after 50000 frames for safety")
                    break

            # Cleanup
            if executor:
                executor.shutdown(wait=False)
            cap.release()
            st.info("📹 Camera released")

    except Exception as e:
        st.error(f"❌ Camera error: {str(e)}")
        st.session_state.streaming = False

elif not st.session_state.streaming:
    with video_placeholder.container():
        st.info("📷 Click 'Start Live Detection' to begin streaming")
        st.image(
            "https://via.placeholder.com/640x480/f0f0f0/666666?text=Camera+Feed+Will+Appear+Here",
            caption="Waiting for stream to start...",
        )

# Footer with enhanced information
st.markdown("---")

# Authors section
st.markdown(
    """
### 👥 **Authors**
This project was developed by:
- **Giovanni Borgogno**
- **Maximo Lucero Ruiz**
- **Santiago Quesada**
- **Felipe Cañas**
- **Paolo Cetti**
"""
)
