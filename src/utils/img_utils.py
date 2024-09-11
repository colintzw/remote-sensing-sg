import cv2
import numpy as np


def adjust_contrast_brightness(image, contrast=1.0, brightness=0.0):
    # Adjust contrast and brightness
    adjusted_image = np.clip(contrast * image + brightness, 0, 1)
    return adjusted_image


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
