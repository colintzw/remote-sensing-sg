import cv2
import numpy as np
from skimage.exposure import match_histograms
from PIL import ImageDraw, ImageFont


def add_annotation(image, text, position=(10, 10), font_path = None):
    draw = ImageDraw.Draw(image)
    if font_path:
        font = ImageFont.truetype(font_path, 40)
    else:
        font = ImageFont.load_default()  # Use default font
    draw.text(position, text, font=font, fill='white')  # Change fill color as needed
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
    img_normalized = np.empty_like(img, dtype=np.float32)
    for i in range(img.shape[-1]):
        channel_min = np.min(img[:, :, i])
        channel_max = np.max(img[:, :, i])
        img_normalized[:, :, i] = (img[:, :, i] - channel_min) / (
            channel_max - channel_min
        )
    return img_normalized

def normalize_to_uint8(img, max_value=0, min_value = 0, saturate_img = False):
    #normalization needed to save to png.
    #saturate if needed
    if saturate_img:
        saturate = np.clip(img, min_value,max_value) - min_value
    else:
        saturate = img

    dynamic_range = max_value - min_value
    return ((saturate/dynamic_range) * 255).astype(np.uint8)


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
