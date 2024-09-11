from datetime import datetime, timedelta
from typing import List, Tuple

import planetary_computer
import pystac_client

from src.sentinel2_handling.stac_item_sentinel2_processor import (
    StacItemSentinel2Processor,
)


def query_sentinel2(bbox, max_cloud_cover=80, end_date=None, num_days_before_end=30):
    """
    Query Sentinel-2 images for a given bounding box and cloud cover threshold.
    """
    # Set up the STAC API client
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Create a search query
    if end_date is None:
        end_date = datetime.now()

    start_date = end_date - timedelta(days=num_days_before_end)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )

    # reverse the list because MPC returns newest results first.
    return list(search.get_items())[::-1]


def filter_item_list(
    item_list, bbox, min_usable_pct=85
) -> Tuple[List[str], List[StacItemSentinel2Processor]]:
    good_item_metadata = []
    good_item_processors = []
    for item in item_list:
        item_date = item.properties["datetime"][:10]
        prev_cloud_cover = item.properties["eo:cloud_cover"]
        item_proc = StacItemSentinel2Processor(item=item, bbox=bbox)
        usable_pixel_percentage = item_proc.compute_usable_pixels()
        if usable_pixel_percentage >= min_usable_pct:
            good_item_metadata.append(
                (item_date, prev_cloud_cover, usable_pixel_percentage)
            )
            good_item_processors.append(item_proc)

    print(
        f"List filtered as {len(good_item_metadata)} out of {len(item_list)} items orginally"
    )
    return (good_item_metadata, good_item_processors)


def attack_script(bbox, min_usable_pct, end_date, num_days_before_end):
    item_list = query_sentinel2(
        bbox=bbox,
        max_cloud_cover=80,
        end_date=end_date,
        num_days_before_end=num_days_before_end,
    )
    good_item_dates, good_item_processors = filter_item_list(
        item_list=item_list, bbox=bbox, min_usable_pct=min_usable_pct
    )

    for item_proc in good_item_processors:
        item_proc.load_and_compute_spectral_indices()

    # Steps that must be done:
    # 1. Compute delta_ndvi
    # 2. Set a threshold on what is a change we want to see.
    # 3. Remove pixels that are not impt.
    # 4. Return polygons?
