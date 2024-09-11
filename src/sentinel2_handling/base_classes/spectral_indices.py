from typing import Dict

import numpy as np

from src.sentinel2_handling.base_classes.raster import Raster
from src.sentinel2_handling.base_classes.sentinel2_bands import Sentinel2L2ABands


# fmt: off
class BaseSpectralIndices:
    rgb_image: Raster   # 3-band image (Red, Green, Blue) typically used for visual interpretation of the scene
    cloud_mask: Raster  # Mask identifying pixels covered by clouds (helps in filtering out cloud-covered areas)
    ndvi: Raster        # Normalized Difference Vegetation Index (High values indicate healthy, dense vegetation; Low values indicate sparse or stressed vegetation)
    bsi: Raster         # Bare Soil Index (High values indicate bare soil; Low values indicate vegetated or non-soil surfaces)
    savi: Raster        # Soil Adjusted Vegetation Index (Adjusts NDVI for soil brightness, useful in areas with sparse vegetation)
    ndmi: Raster        # Normalized Difference Moisture Index (High values indicate moist vegetation; Low values indicate dry conditions or bare soil)
#fmt : on

class Sentinel2SpectralIndices(BaseSpectralIndices):
    # upsampled BSI and NDMI to 10m.
    loaded_bands: Dict[str, Raster]

    def __init__(self, loaded_bands: Dict[str, Raster], only_rgb = False):
        self.loaded_bands = loaded_bands
        self.rgb_image = self.compute_rgb_image(
            red_raster=loaded_bands.get(Sentinel2L2ABands.Red),
            green_raster=loaded_bands.get(Sentinel2L2ABands.Green),
            blue_raster=loaded_bands.get(Sentinel2L2ABands.Blue),
        )
        if not only_rgb:    
            self.cloud_mask = self.compute_cloud_mask(
                scl_raster=loaded_bands.get(Sentinel2L2ABands.SCL),
                ref_raster=loaded_bands.get(Sentinel2L2ABands.Red),
                resample_to_ref=True,
            )
            self.ndvi = self.compute_ndvi(
                nir_raster=loaded_bands.get(Sentinel2L2ABands.NIR),
                red_raster=loaded_bands.get(Sentinel2L2ABands.Red),
            )
            self.bsi = self.compute_bsi(
                nir_raster=loaded_bands.get(Sentinel2L2ABands.NIR),
                swir_raster=loaded_bands.get(Sentinel2L2ABands.SWIR1),
                red_raster=loaded_bands.get(Sentinel2L2ABands.Red),
                blue_raster=loaded_bands.get(Sentinel2L2ABands.Blue),
            )
            self.ndmi = self.compute_ndmi(
                nir_raster=loaded_bands.get(Sentinel2L2ABands.NIR),
                swir_raster=loaded_bands.get(Sentinel2L2ABands.SWIR1),
            )
            self.savi = self.compute_savi(
                nir_raster=loaded_bands.get(Sentinel2L2ABands.NIR),
                red_raster=loaded_bands.get(Sentinel2L2ABands.Red)
            )

    @staticmethod
    def compute_cloud_mask(scl_raster: Raster, ref_raster:Raster = None, resample_to_ref: bool = False) -> Raster:
        mask_dtype = "uint8"
        cloud_mask = np.isin(scl_raster.img, [0, 3, 8, 9]).astype(mask_dtype)
        meta = scl_raster.meta.copy()
        meta["dtype"] = mask_dtype
        meta["nodata"] = None
        cloud_raster = Raster(img=cloud_mask, meta=meta, band_names=["CloudMask (SCL)"])
        if not resample_to_ref:
            return cloud_raster

        if ref_raster is None:
            raise ValueError("ref raster should be filled if resample is True")

        cloud_upsampled = cloud_raster.resample(
            target_shape=ref_raster.img.shape,
            target_affine_transform=ref_raster.meta["transform"],
            band_names=None,
        )
        #rebinarize
        cloud_upsampled.binarize()
        return cloud_upsampled


    @staticmethod
    def compute_ndvi(nir_raster: Raster, red_raster: Raster) -> Raster:
        numerator = Raster.subtract_rasters(raster_left=nir_raster, raster_right=red_raster, new_dtype=np.float64)
        divisor = Raster.add_rasters(raster_left=nir_raster, raster_right=red_raster, new_dtype=np.float64)
        ndvi = Raster.divide_rasters(raster_numerator=numerator, raster_divisor=divisor, new_dtype=np.float64,new_band_names=["NDVI"])
        return ndvi

    @staticmethod
    def compute_savi(nir_raster: Raster, red_raster: Raster) -> Raster:
        nir = nir_raster.img.astype("float")  # NIR 10m
        red = red_raster.img.astype("float")  # Red 10m
        L_factor = 0.5
        numerator = (nir - red) * (1 + L_factor)
        divisor = nir + red + L_factor
        savi = numerator/divisor
        meta = nir_raster.meta.copy()
        meta['dtype'] = str(np.float64)
        return Raster(img = savi, meta = meta, band_names=["SAVI"])

    @staticmethod
    def compute_rgb_image(
        red_raster: Raster, green_raster: Raster, blue_raster: Raster
    ) -> Raster:
        # Combine the Red, Green, and Blue into an RGB image
        red = red_raster.img
        green = green_raster.img
        blue = blue_raster.img
        rgb_image = np.stack([red, green, blue], axis=-1)
        meta = red_raster.meta.copy()
        meta["count"] = 3
        return Raster(img=rgb_image, meta=meta, band_names=["Red", "Green", "Blue"])

    @staticmethod
    def compute_bsi(
        nir_raster: Raster, red_raster: Raster, blue_raster: Raster, swir_raster: Raster
    ) -> Raster:
        swir_upsample = swir_raster.resample(
            target_shape=nir_raster.img.shape,
            target_affine_transform=nir_raster.meta["transform"],
            band_names=None,
        )

        soil_refl = swir_upsample.img.astype(float) + red_raster.img.astype(float)
        veg_refl = nir_raster.img.astype(float) + blue_raster.img.astype(float)

        bsi = (soil_refl - veg_refl) / (soil_refl + veg_refl)
        meta = nir_raster.meta.copy()
        meta["dtype"] = str(bsi.dtype)
        return Raster(img=bsi, meta=meta, band_names=["Bare Soil Index"])

    @staticmethod
    def compute_ndmi(
        nir_raster: Raster, swir_raster: Raster
    ) -> Raster:
        swir_upsample = swir_raster.resample(
            target_shape=nir_raster.img.shape,
            target_affine_transform=nir_raster.meta["transform"],
            band_names=None,
        )

        numerator = Raster.subtract_rasters(nir_raster, swir_upsample, new_dtype=np.float64)
        divisor = Raster.add_rasters(nir_raster, swir_upsample, new_dtype=np.float64)
        ndmi = Raster.divide_rasters(raster_numerator=numerator, raster_divisor=divisor, new_dtype=np.float64, new_band_names=['NDMI'])
        return ndmi
