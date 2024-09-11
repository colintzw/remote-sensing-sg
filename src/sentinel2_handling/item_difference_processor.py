from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.ndimage import binary_dilation, binary_opening, label
from skimage.measure import regionprops

from src.sentinel2_handling.base_classes.raster import Raster
from src.sentinel2_handling.base_classes.spectral_indices import (
    Sentinel2SpectralIndices,
)
from src.sentinel2_handling.img_utils import normalize_image


@dataclass
class IndexDiffs:
    ndvi: Raster
    savi: Raster
    bsi: Raster
    ndmi: Raster


@dataclass
class ArrayBox:
    min_row: int
    max_row: int
    min_col: int
    max_col: int

    def slice_array(self, input_arr: np.array) -> np.array:
        ndims = len(input_arr.shape)
        if (ndims < 2) or (ndims > 3):
            raise ValueError("bad shape")
        return input_arr[self.min_row : self.max_row, self.min_col : self.max_col]

    def box_size_in_pixels(self) -> int:
        dy = self.max_row - self.min_row
        dx = self.max_col - self.min_col
        return dx * dy


def compute_diffs(
    indices_before: Sentinel2SpectralIndices, indices_after: Sentinel2SpectralIndices
):
    diffs = IndexDiffs(
        ndvi=Raster.subtract_rasters(
            raster_left=indices_after.ndvi,
            raster_right=indices_before.ndvi,
            new_band_names=["Delta NDVI"],
        ),
        savi=Raster.subtract_rasters(
            raster_left=indices_after.savi,
            raster_right=indices_before.savi,
            new_band_names=["Delta SAVI"],
        ),
        bsi=Raster.subtract_rasters(
            raster_left=indices_after.bsi,
            raster_right=indices_before.bsi,
            new_band_names=["Delta BSI"],
        ),
        ndmi=Raster.subtract_rasters(
            raster_left=indices_after.ndmi,
            raster_right=indices_before.ndmi,
            new_band_names=["Delta NDMI"],
        ),
    )

    combined_cloud_mask = (indices_after.cloud_mask.img == 1) | (
        indices_before.cloud_mask.img == 1
    )
    return diffs, combined_cloud_mask


def output_boxes_from_mask(
    binary_mask, num_pixels_to_buffer=2, noise_removal=False, min_region_size=2
):
    boxes = []

    if noise_removal:
        noise_kernel = np.ones((3, 3))
        binary_mask = binary_opening(binary_mask, structure=noise_kernel)

    buffer_kernel_h = num_pixels_to_buffer * 2 + 1
    buffer_kernel = np.ones((buffer_kernel_h, buffer_kernel_h))

    dilated_mask = binary_dilation(binary_mask, structure=buffer_kernel)
    labeled_mask, _ = label(dilated_mask)
    for region in regionprops(labeled_mask):
        if region.area < min_region_size:
            continue

        # Get bounding box (min_row, min_col, max_row, max_col)
        min_row, min_col, max_row, max_col = region.bbox

        # save with a different structure thanks.
        boxes.append(
            ArrayBox(min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col)
        )

    return boxes


def output_before_after_rgb_from_boxes(
    boxes: List[ArrayBox],
    indices_before: Sentinel2SpectralIndices,
    indices_after: Sentinel2SpectralIndices,
):
    rgb_before = normalize_image(indices_before.rgb_image.img)
    rgb_after = normalize_image(indices_after.rgb_image.img)
    sliced_pairs = []
    for box in boxes:
        sliced_pairs.append((box.slice_array(rgb_before), box.slice_array(rgb_after)))

    return sliced_pairs
