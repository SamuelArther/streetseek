"""Command line entry point: `python -m streetseek "description" --area "place"` or `python -m streetseek serve`."""

import argparse
import logging
import sys

from . import geo, report, search


def run_search(args):
    try:
        matches = search.search(args.description, args.area, radius_m=args.radius, max_panos=args.max_panos)
    except geo.GeoError as e:
        sys.exit(f"Error: {e}")
    if not matches:
        sys.exit("No matches found")

    print()
    for m in matches:
        where = " | ".join(p for p in (m.feature_name, m.address) if p) or f"{m.lat:.6f}, {m.lon:.6f}"
        status = "BLURRED" if m.blurred else "not blurred"
        print(f"#{m.rank}  match {m.score * 100:.1f}  [{status}]  {where}  ({m.date})")
        print(f"    {m.link}")
    folder = report.write(matches, args.description, args.area, root=args.out)
    print(f"\nPhotos and report: {(folder / 'index.html').resolve()}")


def main():
    # Library chatter: Hugging Face asks for an optional login token, and streetlevel's aiohttp
    # sessions log "unclosed session" after a retried download. Neither affects results.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        from . import server

        parser = argparse.ArgumentParser(prog="streetseek serve")
        parser.add_argument("serve")
        parser.add_argument("--port", type=int, default=8765)
        parser.add_argument("--out", default="results")
        args = parser.parse_args()
        server.serve(args.port, args.out)
        return

    parser = argparse.ArgumentParser(
        prog="streetseek",
        description="Describe a place and find the best matching Street View spots in an area.",
    )
    parser.add_argument("description", help='what it looks like, e.g. "yellow house with a red door and a big oak tree"')
    parser.add_argument("--area", required=True, help='general area, e.g. "Westerville, Ohio" or a street/ZIP')
    parser.add_argument("--radius", type=int, default=500, help="search radius in meters (default 500, max 2000)")
    parser.add_argument("--max-panos", type=int, default=120, help="max panoramas to download (default 120, max 300)")
    parser.add_argument("--out", default="results", help="folder for result reports")
    args = parser.parse_args()
    args.radius = max(50, min(args.radius, 2000))
    args.max_panos = max(5, min(args.max_panos, 300))
    run_search(args)


if __name__ == "__main__":
    main()
