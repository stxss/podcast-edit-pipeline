# Everything after transcription in one command, in dependency order, independent steps in parallel, each step timed.
#   python tq/run_post.py episodes/EPxx.json [--from build] [--no-align] [--no-render-check]
#   align (GPU)  ->  build_edl  ->  { detect -> verify (GPU) } || { check_render (GPU) } || { make_plan -> fades }
#   then tokens.py (for the editorial agent / panel) runs alongside.
# GPU steps use the GPU Python (torch, torchaudio, faster-whisper); the rest use this Python.
import json, os, subprocess, sys, time, argparse
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); import tqconfig

def gpu_python():
    """TQ_GPU_PYTHON, else gpu_python from local.json (tqconfig), else this Python."""
    return os.environ.get("TQ_GPU_PYTHON") or tqconfig.gpu_python()

def run(name, py, script, cfg, log):
    t0 = time.time()
    p = subprocess.run([py, os.path.join(HERE, script), cfg], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    tail = [l for l in (p.stdout + p.stderr).splitlines() if l.strip() and "Warning" not in l and "warn(" not in l][-3:]
    log.append((name, round(dt, 1), p.returncode, tail))
    print(f"  {name:13s} {dt:6.1f} s  {'ok' if p.returncode == 0 else 'FAILED'}  {tail[-1][:110] if tail else ''}", flush=True)
    if p.returncode: raise SystemExit(f"{name} failed:\n" + "\n".join((p.stdout + p.stderr).splitlines()[-25:]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--no-render-check", action="store_true")
    a = ap.parse_args(); cfg = a.config; gpy, py = gpu_python(), sys.executable; log = []; t0 = time.time()
    if not a.no_align: run("align", gpy, "align.py", cfg, log)
    run("build_edl", py, "build_edl.py", cfg, log)
    def chain_detect(): run("detect", py, "detect.py", cfg, log); run("verify_cuts", gpy, "verify_cuts.py", cfg, log)
    def chain_plan(): run("make_plan", py, "make_plan.py", cfg, log); run("fades", py, "fades.py", cfg, log)
    jobs = [chain_detect, chain_plan, lambda: run("tokens", py, "tokens.py", cfg, log)]
    if not a.no_render_check: jobs.append(lambda: run("check_render", gpy, "check_render.py", cfg, log))
    with ThreadPoolExecutor(len(jobs)) as ex:
        for f in [ex.submit(j) for j in jobs]: f.result()
    total = time.time() - t0; serial = sum(x[1] for x in log)
    print(f"done in {total:.0f} s (steps add up to {serial:.0f} s; parallel saved {serial - total:.0f} s)")
    W = tqconfig.load(cfg)["work"]
    json.dump({"total_s": round(total, 1), "steps": [{"step": n, "s": s} for n, s, _, _ in log]}, open(os.path.join(W, "run_post.json"), "w"), indent=1)

if __name__ == "__main__": main()
