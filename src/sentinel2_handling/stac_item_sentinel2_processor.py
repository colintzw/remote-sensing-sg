from typing import Dict, List

import geopandas as gpd
import numpy as np
import pystac
import rasterio
import rasterio.mask
from shapely.geometry import box

from src.sentinel2_handling.base_classes.raster import Raster
from src.sentinel2_handling.base_classes.sentinel2_bands import Sentinel2L2ABands
from src.sentinel2_handling.base_classes.spectral_indices import (
    Sentinel2SpectralIndices,
)


class StacItemSentinel2Processor:
    _item: pystac.item.Item
    _bbox: List
    _clip_gdf: gpd.GeoDataFrame

    # sentinel 2 meta-data
    partial_meta_10m: Dict
    partial_meta_20m: Dict

    # sentinel 2 bands
    S2_ASSET_NAMES = [
        Sentinel2L2ABands.Blue,
        Sentinel2L2ABands.Green,
        Sentinel2L2ABands.Red,
        Sentinel2L2ABands.NIR,
        Sentinel2L2ABands.SWIR1,
        Sentinel2L2ABands.SCL,
    ]
    s2_bands: Dict[str, Raster]
    _bands_loaded: bool = False

    # computed
    spectral_indices: Sentinel2SpectralIndices

    def __init__(self, item, bbox):
        self._item = item
        if len(bbox) != 4:
            raise ValueError("Nope. the bbox should be a 4 tuple.")

        # create the bbox.
        self._bbox = bbox
        self._clip_gdf = gpd.GeoDataFrame({"geometry": [box(*bbox)]}, crs="epsg:4326")

    def __load_and_clip_asset(self, asset, asset_name) -> Raster:
        with rasterio.open(asset.href) as src:
            # Reproject if necessary
            if self._clip_gdf.crs != src.crs:
                self._clip_gdf = self._clip_gdf.to_crs(src.crs)

            # Mask the raster with the bbox
            out_image, out_transform = rasterio.mask.mask(
                src, self._clip_gdf.geometry, crop=True
            )

            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "height": out_image.shape[0],
                    "width": out_image.shape[1],
                    "transform": out_transform,
                }
            )
            # convert img to HxW
            return Raster(img=out_image[0], meta=out_meta, band_names=[asset_name])

    def _load_and_clip_required_assets(self) -> None:
        self.s2_bands = {
            asset_name: self.__load_and_clip_asset(
                asset=self._item.assets[asset_name.value], asset_name=asset_name.value
            )
            for asset_name in self.S2_ASSET_NAMES
        }
        self._bands_loaded = True

    def _compute_spectral_indices(self) -> None:
        # indices are computed on initialization
        self.spectral_indices = Sentinel2SpectralIndices(self.s2_bands)

    def load_and_compute_spectral_indices(self) -> None:
        self._load_and_clip_required_assets()
        self._compute_spectral_indices()

    def compute_usable_pixels(self) -> float:
        if self._bands_loaded:
            cloud_mask = self.spectral_indices.cloud_mask.img
        else:
            scl_raster = self.__load_and_clip_asset(
                asset=self._item.assets[Sentinel2L2ABands.SCL.value],
                asset_name=Sentinel2L2ABands.SCL.value,
            )
            cloud_mask = Sentinel2SpectralIndices.compute_cloud_mask(
                scl_raster=scl_raster, resample_to_ref=False
            ).img

        usable_pixels = np.sum(cloud_mask == 0)
        total_pixels = cloud_mask.size
        return usable_pixels / total_pixels * 100
