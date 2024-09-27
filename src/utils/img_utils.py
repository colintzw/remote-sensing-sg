import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.exposure import match_histograms


def resize_img(pil_image, factor):
    new_size = (int(pil_image.width * factor), int(pil_image.height * factor))
    return pil_image.resize(new_size)


def add_annotation(
    image,
    text,
    position=(10, 10),
    font_path=None,
    font_size=40,
    font_color=(255, 255, 255),
):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()  # Use default font
    draw.text(position, text, font=font, fill=font_color)  # Change fill color as needed
    return image


def adjust_contrast_brightness(image, contrast=1.0, brightness=0.0):
    # Adjust contrast and brightness
    adjusted_image = np.clip(contrast * image + brightness, 0, 1)
    return adjusted_image


def match_images(ref_img, src_img):
    matched_image = match_histograms(src_img, ref_img, channel_axis=-1)
    return matched_image


def normalize_image(img):
    """Normalize a 3D RGB image to the [0, 1] range based on the min and max of each channel."""
    # Normalize each channel independently
    min_vals = img.min(axis=(0, 1))
    max_vals = img.max(axis=(0, 1))
    img_normalized = (img - min_vals) / (max_vals - min_vals)
    return img_normalized


def convert_to_uint8(img):
    # needed to save to png.
    return (img * 255).astype(np.uint8)


def apply_histogram_equalization(image):
    # Apply histogram equalization on each channel separately
    equalized_image = np.empty_like(image)
    for i in range(3):
        equalized_image[..., i] = (
            cv2.equalizeHist((image[..., i] * 255).astype(np.uint8)) / 255.0
        )
    return equalized_image


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Apply CLAHE on each channel separately
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = np.empty_like(image)
    for i in range(3):
        clahe_image[..., i] = (
            clahe.apply((image[..., i] * 255).astype(np.uint8)) / 255.0
        )
    return clahe_image
