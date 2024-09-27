from datetime import datetime, timedelta
from typing import List, Tuple

import planetary_computer
import pystac_client
from joblib import Parallel, delayed

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
    item_list, bbox, min_usable_pct=85, njobs=-1
) -> Tuple[List[str], List[StacItemSentinel2Processor]]:
    good_item_metadata = []
    good_item_processors = []

    results = Parallel(n_jobs=njobs)(
        delayed(get_processor_and_metadata)(
            item=item, bbox=bbox, min_usable_pct=min_usable_pct
        )
        for item in item_list
    )

    for res in results:
        if res[0] is None:
            continue
        good_item_metadata.append(res[0])
        good_item_processors.append(res[1])

    print(
        f"List filtered as {len(good_item_metadata)} out of {len(item_list)} items orginally"
    )
    return (good_item_metadata, good_item_processors)


def get_processor_and_metadata(item, bbox, min_usable_pct):
    item_date = item.properties["datetime"][:10]
    prev_cloud_cover = item.properties["eo:cloud_cover"]
    item_proc = StacItemSentinel2Processor(item=item, bbox=bbox)
    usable_pixel_percentage = item_proc.compute_usable_pixels()
    if usable_pixel_percentage < min_usable_pct:
        return (None, None)
    metadata = (item_date, prev_cloud_cover, usable_pixel_percentage)
    return (metadata, item_proc)
