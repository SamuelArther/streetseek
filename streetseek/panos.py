"""Street View panorama lookup, download and perspective rendering."""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image
from streetlevel import streetview

_local = threading.local()

COVERAGE_ZOOM = 17


def _session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def coverage(lat, lon, radius_m, workers=4):
    """Every official panorama within radius_m, read from Google's coverage tiles (one request per ~300 m tile).

    Returned panoramas only carry id, position and heading; call full_pano() before downloading imagery.
    """
    dlat = radius_m / 111_320
    dlon = radius_m / (111_320 * max(math.cos(math.radians(lat)), 0.01))
    x0, y0 = streetview.wgs84_to_tile_coord(lat + dlat, lon - dlon, COVERAGE_ZOOM)
    x1, y1 = streetview.wgs84_to_tile_coord(lat - dlat, lon + dlon, COVERAGE_ZOOM)
    tiles = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]

    def fetch(tile):
        try:
            return streetview.get_coverage_tile(*tile, session=_session())
        except Exception:
            return []

    with ThreadPoolExecutor(workers) as pool:
        found = [p for batch in pool.map(fetch, tiles) for p in batch]
    seen, result = set(), []
    for p in found:
        if p.id in seen or streetview.is_third_party_panoid(p.id):
            continue
        seen.add(p.id)
        if _distance_m(lat, lon, p.lat, p.lon) <= radius_m:
            result.append(p)
    return result


def full_pano(pano_id):
    """Complete metadata (image sizes, date, address) for a panorama id, or None."""
    try:
        return streetview.find_panorama_by_id(pano_id, session=_session())
    except Exception:
        return None


def full_panos(pano_ids, workers=8):
    with ThreadPoolExecutor(workers) as pool:
        return [p for p in pool.map(full_pano, pano_ids) if p is not None]


def _distance_m(lat1, lon1, lat2, lon2):
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1)
    return 6_371_000 * math.hypot(x, y)


def spaced(panos, min_spacing_m):
    """Greedy subset where no two panoramas are closer than min_spacing_m."""
    if not panos:
        return []
    cell = min_spacing_m / 111_320
    lon_scale = math.cos(math.radians(panos[0].lat))
    grid, kept = {}, []
    for p in panos:
        gx, gy = math.floor(p.lon * lon_scale / cell), math.floor(p.lat / cell)
        near = (q for dx in (-1, 0, 1) for dy in (-1, 0, 1) for q in grid.get((gx + dx, gy + dy), ()))
        if any(_distance_m(p.lat, p.lon, q.lat, q.lon) < min_spacing_m for q in near):
            continue
        grid.setdefault((gx, gy), []).append(p)
        kept.append(p)
    return kept


def fetch_equirect(pano, zoom=2, attempts=3):
    """The full 360 image as a numpy array (H, W, 3), uint8. Retries with backoff on network errors."""
    for attempt in range(attempts):
        try:
            image = streetview.get_panorama(pano, zoom=zoom)
            return np.asarray(image.convert("RGB"))
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * 3**attempt)


def render_view(equirect, pano_heading_rad, heading_deg, fov_deg=90.0, pitch_deg=0.0, size=(640, 480)):
    """Project part of an equirectangular panorama into a normal flat photo.

    heading_deg is a compass bearing; the middle column of a Google panorama faces pano_heading_rad.
    """
    h_src, w_src = equirect.shape[:2]
    w_out, h_out = size
    focal = (w_out / 2) / math.tan(math.radians(fov_deg) / 2)

    xs = np.arange(w_out) - (w_out - 1) / 2
    ys = np.arange(h_out) - (h_out - 1) / 2
    x, y = np.meshgrid(xs, -ys)
    z = np.full_like(x, focal)

    pitch = math.radians(pitch_deg)
    y_r = y * math.cos(pitch) + z * math.sin(pitch)
    z_r = -y * math.sin(pitch) + z * math.cos(pitch)

    yaw = np.arctan2(x, z_r) + math.radians(heading_deg) - pano_heading_rad
    elev = np.arctan2(y_r, np.hypot(x, z_r))

    u = ((yaw / (2 * math.pi) + 0.5) % 1.0) * w_src - 0.5
    v = (0.5 - elev / math.pi) * h_src - 0.5
    return Image.fromarray(_bilinear(equirect, u, v))


def _bilinear(img, u, v):
    h, w = img.shape[:2]
    v = np.clip(v, 0, h - 1)
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    du = (u - u0)[..., None]
    dv = (v - v0)[..., None]
    u0 %= w
    u1 = (u0 + 1) % w
    v1 = np.minimum(v0 + 1, h - 1)
    src = img.astype(np.float32)
    top = src[v0, u0] * (1 - du) + src[v0, u1] * du
    bottom = src[v1, u0] * (1 - du) + src[v1, u1] * du
    return (top * (1 - dv) + bottom * dv).astype(np.uint8)


def link(pano, heading_deg, fov_deg=75.0, pitch_deg=0.0):
    """Google Maps Street View URL; Google's pitch is 90 at the horizon."""
    return pano.permalink(heading=heading_deg % 360, pitch=90.0 + pitch_deg, fov=fov_deg)

