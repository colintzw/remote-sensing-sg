import glob
import time

import numpy as np
import rasterio
import streamlit as st
from PIL import Image
from streamlit_extras.stylable_container import stylable_container


# Function to read and preprocess geotiff files
def read_geotiff(file_path):
    with rasterio.open(file_path) as src:
        # Read all bands
        image = src.read()
        # Transpose to get (height, width, channels)
        image = np.transpose(image, (1, 2, 0))
        # Normalize to 0-255 range
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(
            np.uint8
        )
    return Image.fromarray(image)


# Get list of geotiff files
geotiff_files = sorted(glob.glob("../downloaded_data/punggol_slices/*.tif"))

# Streamlit app
st.title("Geotiff Slideshow")


# Slideshow controls
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    speed = st.slider(
        "Speed (seconds per slide)", min_value=0.1, max_value=5.0, value=1.0, step=0.1
    )

# Initialize session state
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.index = 0

# Display image with navigation buttons on sides
col_prev, col_image, col_next = st.columns([1, 10, 1])

with col_prev:
    with stylable_container(
        key="prev_button",
        css_styles="""
            button {
                position: relative;
                top: 50%;
                transform: translateY(-50%);
                width: 100%;
                height: auto;
                aspect-ratio: 1;
                padding: 0;
            }
            button:hover {
                background-color: #d3d3d3;
            }
            button > div {
                display: flex;
                justify-content: center;
                align-items: center;
            }
        """,
    ):
        prev_slide = st.button("⬅️")

with col_image:
    image_placeholder = st.empty()

with col_next:
    with stylable_container(
        key="next_button",
        css_styles="""
            button {
                position: relative;
                top: 50%;
                transform: translateY(-50%);
                width: 100%;
                height: 500%;
                aspect-ratio: 5;
                padding: 0;
            }
            button:hover {
                background-color: #d3d3d3;
            }
            button > div {
                display: flex;
                justify-content: center;
                align-items: center;
            }
        """,
    ):
        next_slide = st.button("➡️")

# Playback controls
col_start, col_pause = st.columns(2)

with col_start:
    start = st.button("▶️")

with col_pause:
    pause = st.button("⏸️")

# Handle start/pause
if start:
    st.session_state.running = True
if pause:
    st.session_state.running = False

# Handle prev/next
if prev_slide and st.session_state.index > 0:
    st.session_state.index -= 1
if next_slide and st.session_state.index < len(geotiff_files) - 1:
    st.session_state.index += 1

while True:
    # Read and display current image
    current_image = read_geotiff(geotiff_files[st.session_state.index])
    image_placeholder.image(
        current_image,
        caption=f"Image {st.session_state.index + 1}/{len(geotiff_files)}",
        use_column_width=True,
    )

    # If slideshow is running, wait and move to next image
    if st.session_state.running:
        time.sleep(speed)
        st.session_state.index = (st.session_state.index + 1) % len(geotiff_files)
    else:
        time.sleep(0.1)  # Small delay to prevent excessive CPU usage

    # Rerun the app to update the display
    st.rerun()
