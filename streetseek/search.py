"""The search pipeline: area -> candidate panoramas -> rendered views -> CLIP ranking -> blur check."""

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import geo, panos, scoring

SCAN_VIEW_SIZE = (336, 336)
SCAN_FOV = 75.0
SCAN_OFFSETS = (0, 60, 120, 180, 240, 300)
RESULT_VIEW_SIZE = (800, 600)
RESULT_FOV = 80.0
RESULT_PITCH = 10.0
BLUR_THRESHOLD = 0.03
BLUR_CHECK_FOV = 90.0
BLUR_CHECK_SIZE = (640, 480)
MIN_RESULT_SPACING_M = 50
FEATURE_MAX_DISTANCE_M = 80
REFINE_TOP = 8
REFINE_RADIUS_M = 70
REFINE_SPACING_M = 15
REFINE_MIN = 20


@dataclass
class Candidate:
    pano: object
    headings: list
    feature_name: str = ""


@dataclass
class Match:
    rank: int
    score: float
    lat: float
    lon: float
    heading: float
    pano_id: str
    date: str
    address: str
    link: str
    blurred: bool
    blur_score: float
    feature_name: str = ""
    image: object = field(default=None, repr=False)


def _address(pano):
    parts = [getattr(a, "value", str(a)) for a in (pano.address or [])]
    return ", ".join(parts)


def _feature_candidates(features_future, coverage, limit, log):
    """Panoramas aimed at map features (churches, gas stations, ...) named in the description."""
    try:
        features = features_future.result()
    except geo.GeoError as e:
        log(f"Skipping map feature lookup: {e}")
        return {}
    picks = {}
    for flat, flon, name in features:
        if len(picks) >= limit:
            break
        nearest = min(coverage, key=lambda p: geo.distance_m(p.lat, p.lon, flat, flon))
        if geo.distance_m(nearest.lat, nearest.lon, flat, flon) > FEATURE_MAX_DISTANCE_M or nearest.id in picks:
            continue
        heading = geo.bearing_deg(nearest.lat, nearest.lon, flat, flon)
        picks[nearest.id] = ([heading, heading - 30, heading + 30], name)
    if picks:
        log(f"Found {len(picks)} map features matching the description with Street View nearby")
    return picks


def gather_candidates(description, lat, lon, radius_m, max_panos, log):
    with ThreadPoolExecutor(1) as pool:
        features_future = pool.submit(geo.find_features, description, lat, lon, radius_m)
        coverage = panos.coverage(lat, lon, radius_m)
        log(f"{len(coverage)} Street View panoramas in range")
        if not coverage:
            return [], []
        picks = _feature_candidates(features_future, coverage, max_panos // 2, log)

    room = max_panos - len(picks)
    rest = sorted((p for p in coverage if p.id not in picks), key=lambda p: geo.distance_m(lat, lon, p.lat, p.lon))
    spacing = 20.0
    street = panos.spaced(rest, spacing)
    while len(street) > room:
        spacing *= 1.2
        street = panos.spaced(rest, spacing)
    log(f"Scanning {len(street)} street spots about {spacing:.0f} m apart")
    for p in street:
        picks[p.id] = (None, "")

    return _to_candidates(picks), coverage


def _to_candidates(picks):
    """Fetch full metadata for {pano_id: (headings or None, feature_name)} and build Candidates."""
    candidates = []
    for pano in panos.full_panos(list(picks)):
        headings, name = picks.get(pano.id, (None, ""))
        if headings is None:
            base = math.degrees(pano.heading)
            headings = [base + d for d in SCAN_OFFSETS]
        candidates.append(Candidate(pano, headings, name))
    return candidates


def _refine_picks(best, coverage, limit):
    """Extra panoramas packed tightly around the best first-pass spots, so nothing between spots is missed."""
    top = sorted(best.values(), key=lambda b: -b[0])[:REFINE_TOP]
    def gap(p):
        return min(geo.distance_m(p.lat, p.lon, c.pano.lat, c.pano.lon) for _, c, _ in top)

    nearby = sorted((p for p in coverage if p.id not in best and gap(p) <= REFINE_RADIUS_M), key=gap)
    return {p.id: (None, "") for p in panos.spaced(nearby, REFINE_SPACING_M)[:limit]}


def _score_into(best, candidates, description, scorer):
    """Render and score every view of the candidates, keeping each panorama's best view in `best`."""
    with ThreadPoolExecutor(8) as pool:
        views = [v for batch in pool.map(_render_candidate, candidates) for v in batch]
    if not views:
        return 0
    feats = scorer.encode_images([v[2] for v in views])
    scores = scorer.similarity(feats, scorer.description_embedding(description))
    for (cand, heading, _), score in zip(views, scores):
        pid = cand.pano.id
        if pid not in best or score > best[pid][0]:
            best[pid] = (float(score), cand, heading)
    return len(views)


def _render_candidate(cand):
    try:
        eq = panos.fetch_equirect(cand.pano, zoom=2)
    except Exception:
        return []
    return [
        (cand, h % 360, panos.render_view(eq, cand.pano.heading, h, SCAN_FOV, size=SCAN_VIEW_SIZE))
        for h in cand.headings
    ]


def search(description, area, radius_m=500, max_panos=120, top_k=5, scorer=None, log=print):
    lat, lon, area_name = geo.geocode(area)
    log(f"Area: {area_name} ({lat:.5f}, {lon:.5f}), radius {radius_m} m")

    candidates, coverage = gather_candidates(description, lat, lon, radius_m, max_panos, log)
    if not candidates:
        log("No Street View coverage found in that area")
        return []

    scorer = scorer or scoring.Scorer()
    best = {}
    log(f"Downloading and comparing {len(candidates)} panoramas")
    _score_into(best, candidates, description, scorer)

    extra = _refine_picks(best, coverage, max(REFINE_MIN, max_panos // 3))
    if extra:
        log(f"Zooming in: {len(extra)} more spots around the best matches")
        _score_into(best, _to_candidates(extra), description, scorer)

    picked = []
    for score, cand, heading in sorted(best.values(), key=lambda b: -b[0]):
        p = cand.pano
        if any(geo.distance_m(p.lat, p.lon, q.pano.lat, q.pano.lon) < MIN_RESULT_SPACING_M for _, q, _ in picked):
            continue
        picked.append((score, cand, heading))
        if len(picked) == top_k * 2:
            break

    log("Checking the top matches for privacy blur")
    matches = []
    for score, cand, heading in picked:
        if len(matches) == top_k:
            break
        p = cand.pano
        try:
            eq = panos.fetch_equirect(p, zoom=3)
        except Exception:
            log(f"Couldn't download {p.id}, using the next best match instead")
            continue
        image = panos.render_view(eq, p.heading, heading, RESULT_FOV, RESULT_PITCH, size=RESULT_VIEW_SIZE)
        level = panos.render_view(eq, p.heading, heading, BLUR_CHECK_FOV, 0.0, size=BLUR_CHECK_SIZE)
        blur = scoring.blur_fraction(level)
        matches.append(Match(
            rank=len(matches) + 1,
            score=score,
            lat=p.lat,
            lon=p.lon,
            heading=heading,
            pano_id=p.id,
            date=str(p.date or "unknown"),
            address=_address(p),
            link=panos.link(p, heading, RESULT_FOV, RESULT_PITCH),
            blurred=blur >= BLUR_THRESHOLD,
            blur_score=blur,
            feature_name=cand.feature_name,
            image=image,
        ))
    return matches
