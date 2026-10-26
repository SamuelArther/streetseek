"""Geocoding (Nominatim) and map data (Overpass) helpers."""

import math
import re
import time

import requests

USER_AGENT = "streetseek/0.1 (personal street view search tool)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

EARTH_RADIUS_M = 6_371_000

# Description keywords -> Overpass tag filters. Each filter is applied to nodes, ways and relations.
FEATURE_TAGS = {
    r"\bchurch(es)?\b|\bchapel\b|\bcathedral\b": ['["amenity"="place_of_worship"]["religion"="christian"]'],
    r"\bmosque\b": ['["amenity"="place_of_worship"]["religion"="muslim"]'],
    r"\bsynagogue\b": ['["amenity"="place_of_worship"]["religion"="jewish"]'],
    r"\btemple\b": ['["amenity"="place_of_worship"]'],
    r"\bschool\b": ['["amenity"="school"]'],
    r"\bgas station\b|\bpetrol\b|\bfuel\b": ['["amenity"="fuel"]'],
    r"\bwater tower\b": ['["man_made"="water_tower"]'],
    r"\blighthouse\b": ['["man_made"="lighthouse"]'],
    r"\bpark\b": ['["leisure"="park"]'],
    r"\bplayground\b": ['["leisure"="playground"]'],
    r"\bbridge\b": ['["bridge"="yes"]["highway"]'],
    r"\bfire station\b": ['["amenity"="fire_station"]'],
    r"\bpolice station\b": ['["amenity"="police"]'],
    r"\bhospital\b": ['["amenity"="hospital"]'],
    r"\blibrary\b": ['["amenity"="library"]'],
    r"\bpost office\b": ['["amenity"="post_office"]'],
    r"\bbank\b": ['["amenity"="bank"]'],
    r"\bdiner\b|\brestaurant\b": ['["amenity"="restaurant"]'],
    r"\bfast food\b|\bburger\b": ['["amenity"="fast_food"]'],
    r"\bcaf[eé]\b|\bcoffee\b": ['["amenity"="cafe"]'],
    r"\bbar\b|\bpub\b": ['["amenity"~"^(bar|pub)$"]'],
    r"\bmotel\b|\bhotel\b": ['["tourism"~"^(motel|hotel)$"]'],
    r"\bsupermarket\b|\bgrocery\b": ['["shop"="supermarket"]'],
    r"\bconvenience store\b": ['["shop"="convenience"]'],
    r"\bcar wash\b": ['["amenity"="car_wash"]'],
    r"\bstadium\b": ['["leisure"="stadium"]'],
    r"\bcemetery\b|\bgraveyard\b": ['["landuse"="cemetery"]', '["amenity"="grave_yard"]'],
    r"\bfarm\b|\bbarn\b|\bsilo\b": ['["building"~"^(barn|farm|farm_auxiliary|silo)$"]', '["man_made"="silo"]'],
    r"\bwindmill\b": ['["man_made"="windmill"]'],
    r"\bstatue\b|\bmonument\b": ['["historic"~"^(memorial|monument)$"]', '["memorial"="statue"]'],
    r"\bfountain\b": ['["amenity"="fountain"]'],
    r"\btrain station\b|\brailway station\b": ['["railway"="station"]'],
    r"\bmall\b": ['["shop"="mall"]'],
    r"\bapartment(s)?\b": ['["building"="apartments"]'],
}

class GeoError(Exception):
    pass


def _session():
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def geocode(area):
    """Resolve a place name to (lat, lon, display_name)."""
    resp = _session().get(NOMINATIM_URL, params={"q": area, "format": "json", "limit": 1}, timeout=30)
    resp.raise_for_status()
    hits = resp.json()
    if not hits:
        raise GeoError(f"Couldn't find an area called {area!r}")
    hit = hits[0]
    return float(hit["lat"]), float(hit["lon"]), hit["display_name"]


def distance_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Compass bearing from point 1 to point 2, in degrees."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(x, y)) % 360


def feature_filters(description):
    text = description.lower()
    filters = []
    for pattern, tags in FEATURE_TAGS.items():
        if re.search(pattern, text):
            filters.extend(tags)
    return filters


def _overpass(query):
    """Run an Overpass query, falling back to mirrors when a server is busy or down."""
    session = _session()
    errors = []
    for url in OVERPASS_URLS:
        try:
            resp = session.post(url, data={"data": query}, timeout=25)
        except requests.RequestException as e:
            errors.append(f"{url}: {e.__class__.__name__}")
            continue
        if resp.ok:
            return resp.json()["elements"]
        errors.append(f"{url}: HTTP {resp.status_code}")
    raise GeoError("Map data servers are busy (" + "; ".join(errors) + ")")


def find_features(description, lat, lon, radius_m):
    """Map features whose type is named in the description, as (lat, lon, name) tuples."""
    filters = feature_filters(description)
    if not filters:
        return []
    around = f"(around:{int(radius_m)},{lat},{lon})"
    body = "".join(f"nwr{f}{around};" for f in filters)
    try:
        elements = _overpass(f"[out:json][timeout:20];({body});out center tags;")
    except GeoError:
        features = _nominatim_features(description, lat, lon, radius_m)
        if not features:
            raise
        return features
    features = []
    for el in elements:
        point = el.get("center") or el
        if "lat" not in point:
            continue
        name = el.get("tags", {}).get("name", "")
        features.append((point["lat"], point["lon"], name))
    return features


def _nominatim_features(description, lat, lon, radius_m):
    """Fallback for find_features when Overpass is down: Nominatim category search inside the area's box."""
    text = description.lower()
    phrases = [m.group(0) for pattern in FEATURE_TAGS if (m := re.search(pattern, text))]
    dlat = radius_m / 111_320
    dlon = radius_m / (111_320 * max(math.cos(math.radians(lat)), 0.01))
    viewbox = f"{lon - dlon},{lat + dlat},{lon + dlon},{lat - dlat}"
    session = _session()
    features = []
    for i, phrase in enumerate(phrases):
        if i:
            time.sleep(1)  # Nominatim usage policy: at most one request per second
        try:
            resp = session.get(NOMINATIM_URL, timeout=30, params={
                "q": phrase, "format": "json", "limit": 50, "viewbox": viewbox, "bounded": 1,
            })
            resp.raise_for_status()
        except requests.RequestException:
            continue
        for hit in resp.json():
            plat, plon = float(hit["lat"]), float(hit["lon"])
            if distance_m(lat, lon, plat, plon) <= radius_m:
                features.append((plat, plon, hit.get("name") or hit["display_name"].split(",")[0]))
    return features
