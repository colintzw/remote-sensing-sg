# fmt: off
# because black is too OCD..
from enum import Enum


class Sentinel2L2ABands(Enum):
    Blue = "B02"        # 10m resolution, 490 nm central wavelength
    Green = "B03"       # 10m resolution, 560 nm central wavelength
    Red = "B04"         # 10m resolution, 665 nm central wavelength
    NIR = "B08"         # 10m resolution, 842 nm central wavelength

    RedEdge1 = "B05"    # 20m resolution, 705 nm central wavelength
    RedEdge2 = "B06"    # 20m resolution, 740 nm central wavelength
    RedEdge3 = "B07"    # 20m resolution, 783 nm central wavelength
    SWIR1 = "B11"       # 20m resolution, 1610 nm central wavelength
    SWIR2 = "B12"       # 20m resolution, 2190 nm central wavelength

    Coastal = "B01"     # 60m resolution, 443 nm central wavelength
    WaterVapor = "B09"  # 60m resolution, 945 nm central wavelength
    Cirrus = "B10"      # 60m resolution, 1375 nm central wavelength

    #derived product
    SCL = "SCL"         # 20m resolution, Scene Classification Layer
# fmt: on
