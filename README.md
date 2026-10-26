# Streetseek

Describe a place, say roughly where it is, and Streetseek scans Google Street View in that area and returns the
**top 5 matching spots**, each with a photo, a **blurred / not blurred** verdict, and a link that opens Street View
at that exact spot and angle.

## Setup

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The first search downloads the CLIP model (~1.7 GB). An NVIDIA GPU makes it fast; CPU works but is slow.

## Use it

**Web UI:** double-click `streetseek.bat` (or run `python -m streetseek serve`) and open http://127.0.0.1:8765.

**Command line:**

```
python -m streetseek "a red brick church with a tall pointed steeple" --area "German Village, Columbus, Ohio"
```

| Option | Default | What it does |
| --- | --- | --- |
| `--area` | (required) | Town, neighborhood, street, intersection or ZIP. The search is a circle around it. |
| `--radius` | 500 | Circle radius in meters (50–2000). |
| `--max-panos` | 120 | How many Street View spots to check (5–300). More = slower but fewer misses. |
| `--out` | `results` | Where reports are saved. |

Each search saves `results/<time>-<description>/` with `index.html`, the match photos and `results.json`.

## How it works

1. **Area** – the place name is geocoded with OpenStreetMap Nominatim.
2. **Coverage** – Google's Street View coverage tiles list every panorama in the circle (one request per ~300 m tile).
3. **Map features** – if the description names a kind of place (church, gas station, school, water tower, park…),
   OpenStreetMap is searched for those and the camera is aimed straight at each one.
4. **Street scan** – the rest of the budget goes to panoramas spread evenly over the area, each looked at in 6 directions.
5. **Ranking** – [CLIP](https://github.com/mlfoundations/open_clip) (ViT-L/14) scores how well each view matches the
   description. A second pass rescans densely around the 8 best spots so nothing between scan points is missed.
6. **Blur check** – the top matches are re-downloaded in higher resolution and checked for a large region that has
   coarse shape but no fine detail, which is what Google's privacy blur on houses looks like. Small face and
   license-plate blurs are ignored.

## Good to know

- **Be specific about what's visible from the street:** colors, materials, roof shape, porches, signs, trees, fences.
- **Keep the area tight.** A neighborhood works much better than a whole city; spots are spread thinner as the
  radius grows.
- CLIP is good at the overall look but fuzzy on details like *which* thing is red, so the right place is often in
  the top 5 rather than always #1.
- The blur check was tuned on real Street View photos with a privacy-style blur added (1 miss and 0 false alarms
  out of 80 test images). It only looks at the matched view, so a blurred house just out of frame isn't counted.
- Street View imagery is fetched with the [streetlevel](https://github.com/sk-zk/streetlevel) library, which uses the
  same endpoints as the Google Maps website instead of the paid Street View API. It needs no API key but isn't
  officially supported by Google and could break or be rate limited, so keep searches reasonably small.
- Map data © OpenStreetMap contributors. Street View imagery © Google.

## Tests

```
python -m unittest discover -s tests
```
