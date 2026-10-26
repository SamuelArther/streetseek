"""Writes search results to a folder with a self-contained HTML page and the match photos."""

import datetime
import html
import json
import re
from pathlib import Path

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Streetseek results</title>
<style>
  :root {{ --bg:#101418; --card:#1a2027; --text:#e8edf2; --muted:#94a3b3; --accent:#4da3ff; --ok:#3ecf8e; --warn:#ffb547; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text); font:16px/1.5 system-ui, sans-serif; }}
  main {{ max-width:960px; margin:0 auto; padding:24px 16px 48px; }}
  h1 {{ margin:0 0 4px; font-size:1.6rem; }}
  .query {{ color:var(--muted); margin-bottom:24px; }}
  .card {{ background:var(--card); border-radius:12px; overflow:hidden; margin-bottom:20px; }}
  .card img {{ width:100%; display:block; }}
  .info {{ padding:14px 16px; }}
  .top {{ display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }}
  .rank {{ font-weight:700; font-size:1.2rem; }}
  .badge {{ padding:3px 10px; border-radius:999px; font-size:.85rem; font-weight:600; color:#101418; }}
  .clear {{ background:var(--ok); }} .blurred {{ background:var(--warn); }}
  .meta {{ color:var(--muted); font-size:.92rem; margin:6px 0 10px; }}
  a.go {{ display:inline-block; background:var(--accent); color:#06121f; font-weight:600; text-decoration:none; padding:8px 14px; border-radius:8px; }}
  .home {{ color:var(--accent); text-decoration:none; display:inline-block; margin-bottom:12px; }}
  .note {{ color:var(--muted); font-size:.85rem; }}
</style>
</head>
<body>
<main>
{home}<h1>Top {count} matches</h1>
<div class="query">&ldquo;{description}&rdquo; near {area}</div>
{cards}
<p class="note">Match score is how closely CLIP thinks the photo fits your description (higher is better; scores are only comparable within one search).
Blur check looks for a large smeared area like Google's privacy blur; small face and plate blurs are ignored.</p>
</main>
</body>
</html>
"""

CARD = """<div class="card">
  <img src="{image}" alt="Street View match {rank}">
  <div class="info">
    <div class="top"><span class="rank">#{rank} &middot; match {score}</span><span class="badge {badge_class}">{badge}</span></div>
    <div class="meta">{place}<br>{lat:.6f}, {lon:.6f} &middot; facing {heading:.0f}&deg; &middot; imagery {date}</div>
    <a class="go" href="{link}" target="_blank" rel="noopener">Open in Street View</a>
  </div>
</div>"""


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "search"


def write(matches, description, area, root="results", home_link=False):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = Path(root) / f"{stamp}-{_slug(description)}"
    folder.mkdir(parents=True, exist_ok=True)

    cards = []
    for m in matches:
        name = f"match{m.rank}.jpg"
        m.image.save(folder / name, quality=90)
        place = " &middot; ".join(html.escape(p) for p in (m.feature_name, m.address) if p) or "Unnamed spot"
        cards.append(CARD.format(
            image=name, rank=m.rank, score=f"{m.score * 100:.1f}",
            badge="Blurred" if m.blurred else "Not blurred",
            badge_class="blurred" if m.blurred else "clear",
            place=place, lat=m.lat, lon=m.lon, heading=m.heading,
            date=html.escape(m.date), link=html.escape(m.link),
        ))
    page = PAGE.format(
        home='<a class="home" href="/">&larr; New search</a>\n' if home_link else "",
        count=len(matches), description=html.escape(description), area=html.escape(area),
        cards="\n".join(cards) or "<p>No matches found.</p>",
    )
    (folder / "index.html").write_text(page, encoding="utf-8")

    data = [{k: v for k, v in vars(m).items() if k != "image"} for m in matches]
    (folder / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return folder
