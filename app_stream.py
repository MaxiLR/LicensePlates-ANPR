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

    # Detection confidence threshold
    confidence_threshold = st.slider(
        "Detection Confidence", min_value=0.1, max_value=1.0, value=0.5, step=0.05
    )

    # Processing settings
    st.subheader("⚙️ Performance Settings")

    process_every_n_frames = st.slider(
        "Process every N frames",
        min_value=1,
        max_value=10,
        value=3,
        help="Lower values = more detections but slower performance",
    )

    camera_resolution = st.selectbox(
        "Camera Resolution", ["640x480", "800x600", "1280x720"], index=0
    )

    show_fps = st.checkbox("Show FPS", value=True)

    # Camera device selection
    camera_device = st.number_input(
        "Camera Device ID",
        min_value=0,
        max_value=5,
        value=0,
        help="Try different values if camera doesn't work (usually 0)",
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

if start_stream:
    st.session_state.streaming = True

if stop_stream:
    st.session_state.streaming = False

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
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)

            st.success("✅ Camera connected successfully!")

            frame_count = 0
            last_time = time.time()
            fps = 0
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)

            # Stream loop
            while st.session_state.streaming:
                ret, frame = cap.read()

                if not ret:
                    st.error("❌ Failed to read from webcam")
                    break

                current_time = time.time()

                # Calculate FPS
                if show_fps and frame_count % 10 == 0:
                    if frame_count > 0:
                        fps = 10 / (current_time - last_time)
                    last_time = current_time

                # Process every N frames for detection
                detect_this_frame = frame_count % process_every_n_frames == 0
                results = {}

                if detect_this_frame:
                    # Save frame temporarily for processing
                    temp_path = temp_dir / f"stream_frame_{frame_count}.jpg"
                    cv2.imwrite(str(temp_path), frame)

                    try:
                        # Process frame for license plate detection
                        results = processor.process_image(
                            str(temp_path), show_result=False, save_result=False
                        )

                        # Clean up temp file immediately
                        if temp_path.exists():
                            temp_path.unlink()

                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
                        if temp_path.exists():
                            temp_path.unlink()

                # Annotate frame (always show annotations from last detection)
                if hasattr(st.session_state, "last_results") or results:
                    if results:
                        st.session_state.last_results = results

                    if (
                        hasattr(st.session_state, "last_results")
                        and st.session_state.last_results
                    ):
                        annotated_frame = processor._annotate(
                            frame.copy(), st.session_state.last_results
                        )
                    else:
                        annotated_frame = frame.copy()
                else:
                    annotated_frame = frame.copy()

                # Add FPS counter to frame
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

                # Convert BGR to RGB for streamlit
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Display frame
                video_placeholder.image(
                    annotated_frame_rgb,
                    channels="RGB",
                    use_container_width=True,
                    caption=f"Live Feed - Frame {frame_count}",
                )

                # Update results display
                if (
                    hasattr(st.session_state, "last_results")
                    and st.session_state.last_results
                ):
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
                else:
                    with results_placeholder.container():
                        st.info("🔍 Scanning for license plates...")
                        st.write(f"📊 **Stats:**")
                        st.write(f"- Frame: {frame_count}")
                        if show_fps:
                            st.write(f"- FPS: {fps:.1f}")

                frame_count += 1

                # Small delay to control frame rate
                time.sleep(0.03)  # ~30 FPS max

                # Safety check to prevent infinite loop
                if frame_count > 10000:
                    st.warning("Stream stopped after 10000 frames for safety")
                    break

            # Release camera
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

st.markdown(
    """
### 🔧 System Info
- **Model**: license-plate-recognition-rxg4e-wyhgr v3
- **Detector**: YOLOv8 (Local ONNX)
- **OCR**: global-plates-mobile-vit-v2-model
"""
)
