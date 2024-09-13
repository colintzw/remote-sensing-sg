import pandas as pd
import requests
from bs4 import BeautifulSoup


def parse_xml_table(xml_string):
    soup = BeautifulSoup(xml_string, "html.parser")
    table = soup.find("table")

    data = []
    for row in table.find_all("tr"):
        cols = row.find_all(["th", "td"])
        cols = [col.text.strip() for col in cols]
        data.append(cols)

    return data


def expand_description_column(desc):
    data = parse_xml_table(desc)
    return dict(data[1:-2])


def add_desc_col(df):
    desc_table = df.Description.apply(expand_description_column).apply(pd.Series)
    return pd.concat([df, desc_table], axis=1).drop(columns=["Description"])


# querying one map.
def results_from_search(search_val):
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={search_val}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
    response = requests.request("GET", url)
    return response.json()["results"]


def get_lat_long_from_search(search_str):
    results = results_from_search(search_str)
    if not results:
        return None
    # just return the first
    first_result = results[0]
    return (first_result["LATITUDE"], first_result["LONGITUDE"])


def safe_get_lat_long(search_str):
    res = get_lat_long_from_search(search_str)
    if res is not None:
        return (search_str, res)
    return (search_str, None)


# reverse geocode
def reverse_geocode(lat, long, key):
    baseurl = "https://www.onemap.gov.sg/api/public/revgeocode"

    latlong_str = f"{lat},{long}"
    url = (
        baseurl + f"?location={latlong_str}&buffer=100&addressType=All&otherFeatures=N"
    )
    headers = {"Authorization": key}
    res = requests.request("GET", url, headers=headers)
    return res.json()
