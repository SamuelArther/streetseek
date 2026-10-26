"""Small local web UI: a search form that runs searches in the background and shows the report."""

import json
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from . import geo, report, search

FORM = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Streetseek</title>
<style>
  :root { --bg:#101418; --card:#1a2027; --text:#e8edf2; --muted:#94a3b3; --accent:#4da3ff; --bad:#ff6b6b; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:16px/1.5 system-ui, sans-serif; }
  main { max-width:640px; margin:0 auto; padding:32px 16px; }
  h1 { margin:0 0 4px; }
  p.sub { color:var(--muted); margin:0 0 24px; }
  label { display:block; font-weight:600; margin:16px 0 6px; }
  textarea, input { width:100%; padding:10px 12px; border-radius:8px; border:1px solid #2c3540; background:var(--card); color:var(--text); font:inherit; }
  textarea { min-height:110px; resize:vertical; }
  .row { display:flex; gap:12px; } .row > div { flex:1; }
  .hint { color:var(--muted); font-size:.85rem; margin-top:4px; }
  button { margin-top:22px; width:100%; padding:12px; border:0; border-radius:8px; background:var(--accent); color:#06121f; font-weight:700; font-size:1rem; cursor:pointer; }
  button:disabled { opacity:.5; cursor:wait; }
  #log { margin-top:24px; background:var(--card); border-radius:8px; padding:12px 14px; font:14px/1.6 ui-monospace, monospace; white-space:pre-wrap; display:none; }
  .err { color:var(--bad); }
</style>
</head>
<body>
<main>
  <h1>Streetseek</h1>
  <p class="sub">Describe a place, find it on Street View.</p>
  <form id="f">
    <label for="d">What does it look like?</label>
    <textarea id="d" name="description" required placeholder="a yellow two story house with a red door, white picket fence and a big oak tree in the front yard"></textarea>
    <label for="a">General area</label>
    <input id="a" name="area" required placeholder="Westerville, Ohio">
    <div class="hint">A town, neighborhood, street or ZIP code. The search covers a circle around it.</div>
    <div class="row">
      <div><label for="r">Radius (m)</label><input id="r" name="radius" type="number" value="500" min="50" max="2000"></div>
      <div><label for="m">Spots to check</label><input id="m" name="max_panos" type="number" value="120" min="5" max="300"></div>
    </div>
    <button id="go">Search</button>
  </form>
  <div id="log"></div>
</main>
<script>
const form = document.getElementById("f"), log = document.getElementById("log"), go = document.getElementById("go");
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  go.disabled = true; log.style.display = "block"; log.textContent = "Starting...";
  const res = await fetch("/search", { method: "POST", body: new URLSearchParams(new FormData(form)) });
  const { id } = await res.json();
  const poll = async () => {
    const job = await (await fetch("/status/" + id)).json();
    log.textContent = job.log.join("\\n");
    if (job.error) { log.innerHTML += '\\n<span class="err">' + job.error.replace(/</g, "&lt;") + "</span>"; go.disabled = false; return; }
    if (job.report) { location.href = job.report; return; }
    setTimeout(poll, 1000);
  };
  poll();
});
</script>
</body>
</html>
"""


class App:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir).resolve()
        self.jobs = {}
        self.scorer = None
        self.search_lock = threading.Lock()

    def start(self, description, area, radius, max_panos):
        job_id = uuid.uuid4().hex[:12]
        job = {"log": [], "error": None, "report": None}
        self.jobs[job_id] = job
        threading.Thread(target=self._run, args=(job, description, area, radius, max_panos), daemon=True).start()
        return job_id

    def _run(self, job, description, area, radius, max_panos):
        log = job["log"].append
        try:
            with self.search_lock:
                if self.scorer is None:
                    log("Loading the image matching model (first search only)...")
                    from .scoring import Scorer

                    self.scorer = Scorer()
                matches = search.search(description, area, radius, max_panos, scorer=self.scorer, log=log)
            if not matches:
                job["error"] = "No Street View coverage found there. Try a bigger radius or a different area."
                return
            folder = report.write(matches, description, area, root=self.out_dir, home_link=True)
            job["report"] = f"/{self.out_dir.name}/{folder.name}/index.html"
        except geo.GeoError as e:
            job["error"] = str(e)
        except Exception as e:
            job["error"] = f"Search failed: {e.__class__.__name__}: {e}"


def _handler(app):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(app.out_dir.parent), **kwargs)

        def log_message(self, *args):
            pass

        def _json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = FORM.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/status/"):
                job = app.jobs.get(self.path.rsplit("/", 1)[-1])
                self._json(job) if job else self._json({"error": "unknown job"}, 404)
            elif self.path.startswith(f"/{app.out_dir.name}/"):
                super().do_GET()
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/search":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode()).items()}
            description, area = form.get("description", "").strip(), form.get("area", "").strip()
            if not description or not area:
                self._json({"error": "description and area are required"}, 400)
                return
            radius = max(50, min(int(form.get("radius") or 500), 2000))
            max_panos = max(5, min(int(form.get("max_panos") or 120), 300))
            self._json({"id": app.start(description, area, radius, max_panos)})

    return Handler


def serve(port=8765, out_dir="results"):
    app = App(out_dir)
    app.out_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(app))
    print(f"Streetseek running at http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
