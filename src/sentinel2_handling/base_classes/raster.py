from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import geopandas as gpd
import numpy as np
import rasterio
from shapely import box


@dataclass
class Raster:
    img: np.array  # assume that img.shape == h x w x num_bands
    meta: Dict
    band_names: Optional[List[str]] = field(default_factory=list)
    num_bands: int = field(init=False)

    def __post_init__(self):
        # check img shape
        if len(self.img.shape) == 3:
            self.num_bands = self.img.shape[-1]
        elif len(self.img.shape) == 2:
            self.num_bands = 1
        else:
            raise ValueError("Bad img shape")

        # check num band_names match..
        if len(self.band_names) == 0:
            self.band_names = ["ThisIsABand"] * self.num_bands

        if len(self.band_names) != self.num_bands:
            raise ValueError("Bad number band names.")

    @classmethod
    def load_from_tif(cls, path_to_geotiff) -> "Raster":
        with rasterio.open(path_to_geotiff, "r") as src:
            # Read all bands
            image = src.read()
            # Transpose to get (height, width, channels)
            image = np.transpose(image, (1, 2, 0))
            band_names = src.descriptions
            meta = src.meta

        return Raster(img=image, meta=meta, band_names=band_names)

    def clip_to_bbox(self, bbox: List, bbox_crs="EPSG:4326") -> "Raster":
        clip_gdf = gpd.GeoDataFrame({"geometry": [box(*bbox)]}, crs=bbox_crs)
        if clip_gdf.crs != self.meta["crs"]:
            clip_gdf = clip_gdf.to_crs(self.meta["crs"])

        out_image, out_transform = rasterio.mask.mask(
            self.img, clip_gdf.geometry, crop=True
        )
        out_meta = self.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[0],
                "width": out_image.shape[1],
                "transform": out_transform,
            }
        )
        return Raster(img=out_image, band_names=self.band_names, meta=out_meta)

    def to_file(self, filename) -> None:
        # fix band count and datatype
        self.meta["count"] = self.num_bands
        self.meta["dtype"] = str(self.img.dtype)

        with rasterio.open(filename, "w", **self.meta) as dst:
            for b in range(self.num_bands):
                band_num = b + 1
                dst.write(self.img[:, :, b], band_num)
                dst.set_band_description(band_num, self.band_names[b])

    def binarize(self):
        out_img = self.img.copy()
        out_img[out_img <= 0] = 0
        out_img[out_img > 0] = 1
        self.img = out_img

    def resample(
        self, target_shape, target_affine_transform, band_names=None
    ) -> "Raster":
        crs = self.meta["crs"]
        out_image, _ = rasterio.warp.reproject(
            self.img,
            np.zeros(target_shape, dtype=self.img.dtype),
            src_transform=self.meta["transform"],
            src_crs=crs,
            dst_transform=target_affine_transform,
            dst_crs=crs,
            resampling=rasterio.enums.Resampling.bilinear,
        )
        if band_names is None:
            band_names = self.band_names

        out_meta = self.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[0],
                "width": out_image.shape[1],
                "transform": target_affine_transform,
            }
        )
        return Raster(img=out_image, band_names=band_names, meta=out_meta)

    @classmethod
    def subtract_rasters(
        cls,
        raster_left: "Raster",
        raster_right: "Raster",
        new_dtype: Optional[Type[np.number]] = None,
        new_band_names: Optional[List[str]] = None,
    ) -> "Raster":
        if raster_left.num_bands != raster_right.num_bands:
            raise ValueError("Both rasters must have the same number of bands.")

        meta = raster_left.meta.copy()
        if new_dtype is None:
            subtracted_img = raster_left.img - raster_right.img
        else:
            subtracted_img = raster_left.img.astype(new_dtype) - raster_right.img
            # Cast to the desired datatype
            subtracted_img = subtracted_img.astype(new_dtype)
            meta["dtype"] = str(new_dtype)

        if new_band_names is None:
            new_band_names = []
            for n in range(raster_left.num_bands):
                n1 = raster_left.band_names[n]
                n2 = raster_right.band_names[n]
                new_band_names.append(f"{n1} minus {n2}]")

        # Create the output raster
        return cls(subtracted_img, meta, new_band_names)

    @classmethod
    def add_rasters(
        cls,
        raster_left: "Raster",
        raster_right: "Raster",
        new_dtype: Optional[Type[np.number]] = None,
        new_band_names: Optional[List[str]] = None,
    ) -> "Raster":
        if raster_left.num_bands != raster_right.num_bands:
            raise ValueError("Both rasters must have the same number of bands.")

        meta = raster_left.meta.copy()
        if new_dtype is None:
            added_img = raster_left.img + raster_right.img
        else:
            added_img = raster_left.img.astype(new_dtype) + raster_right.img
            # Cast to the desired datatype
            added_img = added_img.astype(new_dtype)
            meta["dtype"] = str(new_dtype)

        if new_band_names is None:
            new_band_names = []
            for n in range(raster_left.num_bands):
                n1 = raster_left.band_names[n]
                n2 = raster_right.band_names[n]
                new_band_names.append(f"{n1} plus {n2}")

        # Create the output raster
        return cls(added_img, meta, new_band_names)

    @classmethod
    def divide_rasters(
        cls,
        raster_numerator: "Raster",
        raster_divisor: "Raster",
        new_dtype: Optional[Type[np.number]] = np.float64,
        new_band_names: Optional[List[str]] = None,
    ) -> "Raster":
        if raster_numerator.num_bands != raster_divisor.num_bands:
            raise ValueError("Both rasters must have the same number of bands.")

        meta = raster_numerator.meta.copy()

        divided_img = np.true_divide(
            raster_numerator.img.astype(new_dtype), raster_divisor.img
        )
        divided_img = divided_img.astype(new_dtype)
        meta["dtype"] = str(new_dtype)

        if new_band_names is None:
            new_band_names = []
            for n in range(raster_numerator.num_bands):
                n1 = raster_numerator.band_names[n]
                n2 = raster_divisor.band_names[n]
                new_band_names.append(f"Division of {n1} by {n2}")

        # Create the output raster
        return cls(divided_img, meta, new_band_names)
