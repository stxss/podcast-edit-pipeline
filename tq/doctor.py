#!/usr/bin/env python3
"""Checks that this machine is ready to run the pipeline and says how to fix what's missing.

    python tq/doctor.py            # everything
    python tq/doctor.py EP01       # also check that episode's files

Each line is OK, WARN (works without it, but slower or with less) or MISSING (needed). Nothing is changed.
"""
import importlib.util, json, os, platform, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
import tqconfig

MAC, WIN = sys.platform == "darwin", os.name == "nt"
PY = "python3" if MAC else "python"
rows = []


def ok(what, detail=""): rows.append(("OK", what, detail))
def warn(what, fix): rows.append(("WARN", what, fix))
def bad(what, fix): rows.append(("MISSING", what, fix))


def run(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:
        return 1, str(e)


def py_has(py, mods):
    code = "import importlib.util,sys;print(','.join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))"
    rc, out = run([py, "-c", code, *mods])
    return None if rc else [m for m in out.split(",") if m]


def main():
    L = tqconfig.local()
    # ---- python + packages
    if sys.version_info >= (3, 10): ok("Python", sys.version.split()[0])
    else: bad("Python 3.10+", f"you have {sys.version.split()[0]}; install a newer Python")
    if importlib.util.find_spec("numpy"): ok("numpy")
    else: bad("numpy", f"{PY} -m pip install -r requirements.txt")
    # ---- ffmpeg
    ff = tqconfig.ffmpeg()
    if shutil.which(ff) or os.path.isfile(ff): ok("ffmpeg", shutil.which(ff) or ff)
    else: bad("ffmpeg", "brew install ffmpeg" if MAC else "winget install Gyan.FFmpeg (then open a new terminal)")
    # ---- local.json + show profile
    if L.get("_file"): ok("local.json", L["_file"])
    else: bad("local.json", "copy local.example.json to local.json and fill it in")
    sd = tqconfig.show_dir()
    show = None
    if not sd: bad("show_dir in local.json", "copy examples/show somewhere outside the repo and point show_dir at it")
    elif not os.path.isfile(os.path.join(sd, "show.json")): bad("show.json", f"not found in {sd}")
    else:
        try:
            show = tqconfig.load_show(os.path.join(sd, "show.json")); ok("show profile", f"{show.get('name')} - speakers {[s['id'] for s in show['speakers']]}")
        except SystemExit as e: bad("show.json", str(e))
        except Exception as e: bad("show.json", f"not valid JSON: {e}")
        for f in ("SHOW.md", "EDITORIAL.md"):
            if not os.path.isfile(os.path.join(sd, f)): warn(f, f"no {f} in the show profile (Claude has no show rules)")
    mr = tqconfig.media_root(show.get("id") if show else None)
    if mr and os.path.isdir(mr): ok("media_root", mr)
    else: bad("media_root", "set media_root in local.json to the folder with the raw episode folders")
    # ---- premiere-mcp
    pm = L.get("premiere_mcp") or os.environ.get("PREMIERE_MCP_DIR")
    if pm and os.path.isfile(os.path.join(pm, "server", "src", "index.js")):
        ok("premiere-mcp", pm)
        if not os.path.isdir(os.path.join(pm, "server", "node_modules")): bad("premiere-mcp npm install", f"cd \"{pm}/server\" && npm install")
    else: bad("premiere-mcp", "clone premiere-mcp and set premiere_mcp in local.json")
    if os.environ.get("PREMIERE_MCP_DIR"): ok("PREMIERE_MCP_DIR", os.environ["PREMIERE_MCP_DIR"])
    else: bad("PREMIERE_MCP_DIR env var (.mcp.json uses it)",
              'add  export PREMIERE_MCP_DIR="/path/to/premiere-mcp"  to ~/.zshrc' if MAC else 'setx PREMIERE_MCP_DIR "C:\\path\\to\\premiere-mcp"  (then open a new terminal)')
    rc, out = run(["node", "--version"])
    if rc == 0 and int(out.lstrip("v").split(".")[0]) >= 22: ok("Node.js", out)
    else: bad("Node.js 22+", "brew install node" if MAC else "winget install OpenJS.NodeJS.LTS")
    # ---- CEP helper (audio transitions)
    ext = os.path.expanduser("~/Library/Application Support/Adobe/CEP/extensions") if MAC else os.path.join(os.environ.get("APPDATA", ""), "Adobe", "CEP", "extensions")
    if os.path.exists(os.path.join(ext, "dev.premieremcp.qehelper")): ok("CEP helper linked", ext)
    else: warn("CEP helper (audio transitions)", "link premiere-mcp/cep into " + ext + " (README step 2)")
    if MAC: rc, out = run(["defaults", "read", "com.adobe.CSXS.12", "PlayerDebugMode"])
    elif WIN: rc, out = run(["reg", "query", r"HKCU\Software\Adobe\CSXS.12", "/v", "PlayerDebugMode"])
    else: rc, out = 1, ""
    if rc == 0 and "1" in out: ok("CEP debug mode (CSXS.12)")
    else: warn("CEP debug mode", "defaults write com.adobe.CSXS.12 PlayerDebugMode 1" if MAC else r"reg add HKCU\Software\Adobe\CSXS.12 /v PlayerDebugMode /t REG_SZ /d 1 /f")
    # ---- alignment / transcription
    gp = tqconfig.gpu_python()
    miss = py_has(gp, ["torch", "torchaudio"])
    if miss is None: bad("gpu_python", f"{gp} does not run")
    elif miss: bad(f"torch/torchaudio in {gp}", f"{gp} -m pip install -r requirements-gpu.txt (word alignment needs them)")
    else:
        rc, out = run([gp, "-c", "import torch;print('cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else 'cpu'))"])
        (ok if out.strip() in ("cuda", "mps") else warn)("alignment device", out.strip() if out.strip() in ("cuda", "mps") else "CPU only: tq/align.py works but is slower")
    nem = L.get("nemo_speech") or any(os.path.isdir(os.path.join(REPO, "tools", "nemo-speech", d)) for d in (os.listdir(os.path.join(REPO, "tools", "nemo-speech")) if os.path.isdir(os.path.join(REPO, "tools", "nemo-speech")) else []))
    asr = []
    if nem: asr.append("nemotron (tq/asr_nemotron.py)")
    wp = tqconfig.whisper_python(); m = py_has(wp, ["faster_whisper"])
    if m == []: asr.append("faster-whisper (tq/asr_whisper.py)")
    if MAC:
        m2 = py_has(wp, ["mlx_whisper"])
        if m2 == []: asr.append("mlx-whisper (tq/asr_whisper.py)")
    if asr: ok("transcription", ", ".join(asr))
    else: bad("a transcription engine", ("python3 -m pip install -r requirements-mac.txt" if MAC else "pip install faster-whisper (tq/asr_whisper.py), or NeMo-Speech.cpp for Nemotron (README)"))
    if MAC and platform.machine() != "arm64": warn("Intel Mac", "transcription will be very slow; an Apple Silicon Mac is recommended")
    # ---- an episode
    if len(sys.argv) > 1:
        try:
            c = tqconfig.load(sys.argv[1]); ok("episode config", c["_path"])
            for s in tqconfig.speakers(c):
                r = (c.get("raw") or {}).get(s)
                if r and os.path.isfile(r): ok(f"raw {s}", r)
                else: bad(f"raw {s}", f"not found: {r}")
                w = os.path.join(c["work"], s + ".wav")
                (ok if os.path.isfile(w) else warn)(f"{s}.wav", w if os.path.isfile(w) else "not extracted yet (step 3)")
        except SystemExit as e: bad("episode", str(e))
    w = max(len(r[1]) for r in rows)
    for st, what, d in rows: print(f"{st:8}{what:<{w}}  {d}")
    n = sum(r[0] == "MISSING" for r in rows)
    print(f"\n{'ready' if n == 0 else str(n) + ' thing(s) missing'} ({platform.system()} {platform.machine()})")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
