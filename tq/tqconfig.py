# Per-machine settings, show profiles and episode config loading.
#
# local.json (gitignored, copy local.example.json) holds what differs between machines:
#   show_dir        default show profile folder (holds show.json and episodes/), so scripts accept "EP01" for an episode
#   media_root      folder with the raw episode folders; per show: "shows": {"<show id>": {"media_root": ...}}
#   premiere_mcp    clone of the premiere-mcp repo (fill_holes.py runs its server/src/transcribe.py)
#   gpu_python      Python with torch + torchaudio (align.py, verify_cuts.py); "" = the one running the script
#   whisper_python  Python with faster-whisper (fill_holes.py); "" = the one running the script
#   ffmpeg          ffmpeg executable ("ffmpeg" = from PATH)
# It is looked up in the repo root and its parents (so git worktrees nested in the checkout find it),
# or at the path in the TQ_LOCAL environment variable.
#
# A show profile is a folder with show.json (speakers, layout, Premiere template names, pacing) and
# episodes/<EP>.json. It lives outside this repo (see examples/show for the format). An episode config finds its
# show.json in its own folder or a parent folder.
#
# Episode configs may hold "work" and "raw" paths relative to media_root; absolute paths keep working.
# load() resolves them and attaches the show; save() writes the file back exactly as it was apart from your edits.
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
_ADDED = ("_unresolved", "_show", "_show_dir", "_shift", "_path")
_local = None


def local_path():
    env = os.environ.get("TQ_LOCAL")
    if env: return env
    d = REPO
    while True:
        p = os.path.join(d, "local.json")
        if os.path.isfile(p): return p
        up = os.path.dirname(d)
        if up == d: return None
        d = up


def local():
    global _local
    if _local is None:
        p = local_path()
        _local = json.load(open(p, encoding="utf-8")) if p else {}
        _local["_file"] = p
    return _local


def media_root(show_id=None):
    L = local()
    r = (L.get("shows", {}).get(show_id, {}) if show_id else {}).get("media_root") or L.get("media_root")
    return os.path.expanduser(r) if r else None


def resolve(p, show_id=None):
    """Absolute (or ~) paths are returned as given; relative ones are joined to the show's media_root."""
    if p is None: return p
    q = os.path.expanduser(p)
    if os.path.isabs(q): return q if q != p else p
    root = media_root(show_id)
    if not root:
        sys.exit(f"tqconfig: '{p}' is relative but no media_root is set - create local.json from local.example.json")
    return os.path.join(root, q).replace("\\", "/")


def show_dir():
    d = local().get("show_dir")
    return os.path.expanduser(d) if d else None


def episode_path(arg):
    """A path to an episode config, or a bare episode name ("EP01") looked up in show_dir/episodes/."""
    if os.path.isfile(arg) or arg.endswith(".json"): return arg
    d = show_dir()
    if not d: sys.exit(f"tqconfig: '{arg}' is not a file and no show_dir is set in local.json")
    return os.path.join(d, "episodes", arg + ".json")


def find_show(start):
    d = os.path.abspath(start)
    while True:
        p = os.path.join(d, "show.json")
        if os.path.isfile(p): return p
        up = os.path.dirname(d)
        if up == d: return None
        d = up


def load_show(path):
    s = json.load(open(path, encoding="utf-8"))
    if not s.get("speakers"): sys.exit(f"tqconfig: {path} has no speakers")
    return s


def load(path):
    path = episode_path(path)
    cfg = json.load(open(path, encoding="utf-8"))
    sp = find_show(os.path.dirname(path))
    if not sp: sys.exit(f"tqconfig: no show.json in {os.path.dirname(os.path.abspath(path))} or above")
    show = load_show(sp)
    sid = show.get("id")
    orig = {"work": cfg.get("work"), "raw": dict(cfg["raw"]) if isinstance(cfg.get("raw"), dict) else cfg.get("raw")}
    if cfg.get("work") is not None: cfg["work"] = resolve(cfg["work"], sid)
    if isinstance(cfg.get("raw"), dict): cfg["raw"] = {k: resolve(v, sid) for k, v in cfg["raw"].items()}
    cfg["_unresolved"] = orig
    cfg["_show"] = show; cfg["_show_dir"] = os.path.dirname(os.path.abspath(sp)); cfg["_path"] = os.path.abspath(path)
    # sync.shift_s = {speaker: seconds to add to that speaker's clock to get the first speaker's clock}.
    # Older configs store one key per speaker instead ("<id>_shift_s").
    sync = cfg.get("sync") or {}
    sh = dict(sync.get("shift_s") or {})
    for s in speakers(cfg):
        if s not in sh: sh[s] = sync.get(f"{s}_shift_s")
        sh[s] = float(sh[s] or 0.0)
    cfg["_shift"] = sh
    return cfg


def save(cfg, path=None):
    """Write the config back with the work/raw strings as they were in the file and without the loaded show."""
    path = path or cfg["_path"]
    out = {k: v for k, v in cfg.items() if k not in _ADDED}
    orig = cfg.get("_unresolved")
    if orig:
        if orig["work"] is not None: out["work"] = orig["work"]
        if orig["raw"] is not None: out["raw"] = orig["raw"]
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)


# ---- speakers ----
def show(cfg): return cfg["_show"]
def speakers(cfg): return [s["id"] for s in cfg["_show"]["speakers"]]
def ref(cfg): return speakers(cfg)[0]                      # the time base: everything is in this speaker's audio clock
def others(cfg, s): return [x for x in speakers(cfg) if x != s]
def speaker(cfg, s): return next(x for x in cfg["_show"]["speakers"] if x["id"] == s)
def shift_s(cfg, s): return cfg["_shift"][s]
def setting(cfg, section, key, default):
    """A tuning value from show.json (section.key), else the engine default."""
    return cfg["_show"].get(section, {}).get(key, default)


def ffmpeg(): return local().get("ffmpeg") or "ffmpeg"
def gpu_python(): return local().get("gpu_python") or sys.executable
def whisper_python(): return local().get("whisper_python") or sys.executable


def premiere_mcp(*parts):
    root = local().get("premiere_mcp")
    if not root:
        sys.exit("tqconfig: set premiere_mcp (path to the premiere-mcp clone) in local.json")
    return os.path.join(os.path.expanduser(root), *parts)


if __name__ == "__main__":   # python tq/tqconfig.py [EPxx | path/to/EPxx.json] -> show what this machine resolves
    L = local()
    print("local.json:", L.get("_file") or "(none)")
    for k in ("show_dir", "media_root", "premiere_mcp", "gpu_python", "whisper_python", "ffmpeg"):
        print(f"  {k}: {L.get(k, '')!r}")
    if len(sys.argv) > 1:
        c = load(sys.argv[1]); print("show:", c["_show"].get("name"), "speakers:", speakers(c), "shift_s:", c["_shift"])
        print("work:", c["work"]); print("raw:", c.get("raw"))
