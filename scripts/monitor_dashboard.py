"""
Mini load dashboard (a tiny local Grafana-style monitor).

Runs a lightweight HTTP server on http://localhost:8765 that shows live
system load (CPU, RAM, swap) and the training process while experiments run.

Usage:
    venv/bin/python scripts/monitor_dashboard.py

Then open http://localhost:8765 in a browser. No internet needed.
Stop with Ctrl+C. Designed to be light on an 8 GB machine.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = PROJECT_ROOT / "results" / "final_experiment_registry.csv"

PORT = 8765
SAMPLE_SECONDS = 2.0
HISTORY = 180  # ~6 minutes of 2s samples

# Shared state ------------------------------------------------------------
_lock = threading.Lock()
_samples: deque[dict] = deque(maxlen=HISTORY)
_stop = threading.Event()


def _find_training_proc() -> psutil.Process | None:
    """Return the running train_icbhi_baseline python process, if any."""
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            cmdline = proc.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any("train_icbhi_baseline" in str(part) for part in cmdline):
            return proc
    return None


def _experiment_label(proc: psutil.Process | None) -> str:
    if proc is None:
        return "—"
    try:
        cmdline = proc.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "—"
    model = "?"
    ckpt = ""
    for i, part in enumerate(cmdline):
        if part == "--model-name" and i + 1 < len(cmdline):
            model = cmdline[i + 1]
        if part == "--checkpoint-path" and i + 1 < len(cmdline):
            ckpt = Path(cmdline[i + 1]).stem
    return ckpt or model


def _registry_count() -> int:
    if not REGISTRY.exists():
        return 0
    try:
        lines = REGISTRY.read_text(encoding="utf-8").strip().splitlines()
        return max(0, len(lines) - 1)  # minus header
    except OSError:
        return 0


def _sampler() -> None:
    psutil.cpu_percent(interval=None)  # prime
    train_proc: psutil.Process | None = None
    while not _stop.is_set():
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()

        proc = _find_training_proc()
        proc_cpu = 0.0
        proc_rss_mb = 0.0
        elapsed = 0.0
        if proc is not None:
            try:
                if train_proc is None or train_proc.pid != proc.pid:
                    train_proc = proc
                    train_proc.cpu_percent(interval=None)  # prime per-proc
                proc_cpu = train_proc.cpu_percent(interval=None)
                proc_rss_mb = train_proc.memory_info().rss / (1024 * 1024)
                elapsed = time.time() - train_proc.create_time()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                train_proc = None
        else:
            train_proc = None

        sample = {
            "t": time.strftime("%H:%M:%S"),
            "cpu": round(cpu, 1),
            "ram_pct": round(vm.percent, 1),
            "ram_used_gb": round(vm.used / (1024**3), 2),
            "ram_total_gb": round(vm.total / (1024**3), 2),
            "swap_pct": round(sw.percent, 1),
            "swap_used_gb": round(sw.used / (1024**3), 2),
            "proc_cpu": round(proc_cpu, 1),
            "proc_rss_mb": round(proc_rss_mb, 1),
            "experiment": _experiment_label(proc),
            "elapsed_s": int(elapsed),
            "running": proc is not None,
            "done_count": _registry_count(),
        }
        with _lock:
            _samples.append(sample)
        _stop.wait(SAMPLE_SECONDS)


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Thesis Load Monitor</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0f1115; color:#e6e6e6;
         font-family:-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:14px 20px; border-bottom:1px solid #222733;
           display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  h1 { font-size:16px; margin:0; font-weight:600; }
  .pill { padding:3px 10px; border-radius:999px; font-size:12px;
          background:#1b2230; border:1px solid #2a3346; }
  .pill.run { background:#10331f; border-color:#1d6b3a; color:#5be08a; }
  .pill.idle { color:#9aa4b2; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
          gap:14px; padding:18px; }
  .card { background:#151922; border:1px solid #222733; border-radius:12px;
          padding:14px 16px; }
  .label { font-size:12px; color:#9aa4b2; text-transform:uppercase;
           letter-spacing:.05em; }
  .val { font-size:28px; font-weight:700; margin-top:2px; }
  .sub { font-size:12px; color:#7d8696; margin-top:2px; }
  canvas { width:100%; height:90px; margin-top:8px; display:block; }
  .warn { color:#ffb454; }
  .bad  { color:#ff6b6b; }
</style>
</head>
<body>
<header>
  <h1>🩺 Thesis Load Monitor</h1>
  <span id="status" class="pill idle">idle</span>
  <span id="exp" class="pill">experiment: —</span>
  <span id="progress" class="pill">final runs: 0 / 5</span>
  <span id="clock" class="pill">--:--:--</span>
</header>
<div class="grid">
  <div class="card"><div class="label">CPU total</div>
    <div class="val" id="cpu">0%</div>
    <canvas id="c_cpu"></canvas></div>
  <div class="card"><div class="label">RAM</div>
    <div class="val" id="ram">0%</div>
    <div class="sub" id="ram_sub"></div>
    <canvas id="c_ram"></canvas></div>
  <div class="card"><div class="label">Swap (disk paging = slowdown)</div>
    <div class="val" id="swap">0%</div>
    <div class="sub" id="swap_sub"></div>
    <canvas id="c_swap"></canvas></div>
  <div class="card"><div class="label">Training process</div>
    <div class="val" id="pcpu">0%</div>
    <div class="sub" id="prss"></div>
    <canvas id="c_prss"></canvas></div>
</div>
<script>
function draw(id, data, color, max) {
  const cv = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w*dpr; cv.height = h*dpr;
  const ctx = cv.getContext('2d'); ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle = '#222733'; ctx.lineWidth = 1;
  for (let i=1;i<4;i++){const y=h*i/4; ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  if (!data.length) return;
  const m = max || Math.max(1, ...data);
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
  data.forEach((v,i)=>{const x=w*i/(data.length-1||1); const y=h-(v/m)*h*0.95-2;
    i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
  ctx.stroke();
  ctx.lineTo(w,h); ctx.lineTo(0,h); ctx.closePath();
  ctx.fillStyle = color+'22'; ctx.fill();
}
async function tick() {
  try {
    const r = await fetch('/metrics'); const s = await r.json();
    if (!s.length) return;
    const last = s[s.length-1];
    document.getElementById('clock').textContent = last.t;
    document.getElementById('cpu').textContent = last.cpu + '%';
    document.getElementById('ram').textContent = last.ram_pct + '%';
    document.getElementById('ram_sub').textContent =
      last.ram_used_gb + ' / ' + last.ram_total_gb + ' GB';
    document.getElementById('swap').textContent = last.swap_pct + '%';
    document.getElementById('swap_sub').textContent = last.swap_used_gb + ' GB paged';
    document.getElementById('pcpu').textContent = last.proc_cpu + '%';
    document.getElementById('prss').textContent = last.proc_rss_mb + ' MB RSS';
    document.getElementById('progress').textContent = 'final runs: ' + last.done_count + ' / 5';
    const exp = document.getElementById('exp');
    exp.textContent = 'experiment: ' + last.experiment;
    const st = document.getElementById('status');
    if (last.running) { st.textContent='running ('+Math.floor(last.elapsed_s/60)+'m'+(last.elapsed_s%60)+'s)'; st.className='pill run'; }
    else { st.textContent='idle'; st.className='pill idle'; }
    const ramEl=document.getElementById('ram');
    ramEl.className='val'+(last.ram_pct>92?' bad':last.ram_pct>80?' warn':'');
    const swEl=document.getElementById('swap');
    swEl.className='val'+(last.swap_pct>60?' bad':last.swap_pct>20?' warn':'');
    draw('c_cpu', s.map(x=>x.cpu), '#4aa3ff', 100);
    draw('c_ram', s.map(x=>x.ram_pct), '#5be08a', 100);
    draw('c_swap', s.map(x=>x.swap_pct), '#ffb454', 100);
    draw('c_prss', s.map(x=>x.proc_rss_mb), '#c77dff', null);
  } catch(e) {}
}
setInterval(tick, 2000); tick();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence request logging
        pass

    def do_GET(self):
        if self.path.startswith("/metrics"):
            with _lock:
                payload = json.dumps(list(_samples)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main() -> None:
    t = threading.Thread(target=_sampler, daemon=True)
    t.start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Load monitor running at http://localhost:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping monitor.")
    finally:
        _stop.set()
        server.shutdown()


if __name__ == "__main__":
    main()
