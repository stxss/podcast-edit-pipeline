# CrisperWhisper sometimes drops whole stretches of speech (seen: 4 holes of 5-10 s on one mic).
# Finds mic-active stretches with no transcript words, transcribes just those windows with large-v3
# (premiere-mcp's transcribe.py) and merges the words into the verbatim transcript (flag "fill": true).
# Usage: python tq/fill_holes.py episodes/EPxx.json [--min 3.0] [--dry]
import json, os, sys, subprocess, argparse, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from vadlib import load_db, runs, close
import tqconfig

def holes(work, spk, tr, min_s):
    db = load_db(os.path.join(work, spk))
    act = close(db > -55, 30)
    cov = np.zeros(len(db), bool)
    for g in tr["segments"]:
        for w in g["words"]: cov[int(w["start"] * 100):int(w["end"] * 100) + 1] = True
    cov = close(cov, 150)
    return [(a / 100, b / 100) for a, b in runs(act & ~cov) if (b - a) / 100 >= min_s]

FILLER_PROMPT = "Um, uh, so, like, you know, I- I mean, uh, we- we were, um, yeah."   # keeps fillers / stutters in large-v3

def transcribe_window(W, spk, x, y, tdir, own_off=0.0):
    """large-v3 words for [x,y] (speaker's own clock), returned in that clock."""
    s0 = max(0.0, x - 1.0); dur = (y - x) + 2.0
    wav = os.path.join(tdir, f"{spk}_{s0:.1f}_{dur:.1f}.wav"); js = wav[:-4] + ".json"
    if not os.path.exists(js):
        subprocess.run([tqconfig.ffmpeg(), "-v", "error", "-y", "-ss", f"{s0:.3f}", "-t", f"{dur:.3f}", "-i", os.path.join(W, spk + ".wav"), wav], check=True)
        subprocess.run([tqconfig.whisper_python(), tqconfig.premiere_mcp("server", "src", "transcribe.py"), "--input", wav, "--output", js, "--model", "large-v3", "--language", "en", "--ffmpeg", tqconfig.ffmpeg(),
                        "--initial-prompt", FILLER_PROMPT], check=True, stderr=subprocess.DEVNULL)
    d = json.load(open(js, encoding="utf-8"))
    return [{"start": round(w["start"] + s0, 3), "end": round(w["end"] + s0, 3), "word": " " + w["word"].strip(), "prob": w.get("prob", 0)}
            for g in d["segments"] for w in g["words"] if x - 0.05 <= (w["start"] + w["end"]) / 2 + s0 <= y + 0.05]

def replace_ranges(cfg, ranges, workers=4):
    """ranges: [(spk, a, b)] in reference-audio seconds. Words of spk with their midpoint in [a,b] are replaced by large-v3 words.
    For CrisperWhisper's invented repeats ('and and and' over a real sentence) and holes it skipped."""
    from concurrent.futures import ThreadPoolExecutor
    W = cfg["work"]; tdir = os.path.join(W, "holes"); os.makedirs(tdir, exist_ok=True)
    jobs = [(spk, a - tqconfig.shift_s(cfg, spk), b - tqconfig.shift_s(cfg, spk)) for spk, a, b in ranges]   # -> own clock
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda j: transcribe_window(W, j[0], j[1], j[2], tdir), jobs))
    for spk in tqconfig.speakers(cfg):
        mine = [(j, r) for j, r in zip(jobs, res) if j[0] == spk]
        if not mine: continue
        f = os.path.join(W, cfg["transcripts_verbatim"][spk]); tr = json.load(open(f, encoding="utf-8"))
        if not os.path.exists(f + ".orig"): shutil.copy(f, f + ".orig")
        removed = added = 0
        for (_, x, y), words in mine:
            for g in tr["segments"]:
                keep = [w for w in g["words"] if not (x <= (w["start"] + w["end"]) / 2 <= y)]
                removed += len(g["words"]) - len(keep); g["words"] = keep
            if words:
                tr["segments"].append({"start": words[0]["start"], "end": words[-1]["end"], "text": " ".join(w["word"].strip() for w in words),
                                       "words": words, "fill": "large-v3-replace"})
                added += len(words)
        tr["segments"] = sorted([g for g in tr["segments"] if g["words"]], key=lambda g: g["start"])
        json.dump(tr, open(f, "w", encoding="utf-8"), ensure_ascii=False)
        print(spk, f"replaced {len(mine)} ranges: -{removed} words, +{added} words")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--min", type=float, default=3.0); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--replace", nargs="*", default=[], help="spk:a-b in reference-audio seconds, e.g. host:1688-1720")
    a = ap.parse_args(); cfg = tqconfig.load(a.config); W = cfg["work"]
    if a.replace:
        rr = []
        for r in a.replace:
            spk, span = r.split(":"); x, y = map(float, span.split("-")); rr.append((spk, x, y))
        replace_ranges(cfg, rr); return
    tdir = os.path.join(W, "holes"); os.makedirs(tdir, exist_ok=True)
    for spk in tqconfig.speakers(cfg):
        f = os.path.join(W, cfg["transcripts_verbatim"][spk]); tr = json.load(open(f, encoding="utf-8"))
        if any(g.get("fill") for g in tr["segments"]): print(spk, "already filled"); continue
        hs = holes(W, spk, tr, a.min); print(spk, "holes:", [(round(x, 1), round(y - x, 1)) for x, y in hs])
        if a.dry or not hs: continue
        existing = [(w["start"], w["end"]) for g in tr["segments"] for w in g["words"]]
        added = 0
        for x, y in hs:
            s0 = max(0.0, x - 1.0); dur = (y - x) + 2.0
            wav = os.path.join(tdir, f"{spk}_{s0:.1f}.wav"); js = wav[:-4] + ".json"
            subprocess.run([tqconfig.ffmpeg(), "-v", "error", "-y", "-ss", f"{s0:.3f}", "-t", f"{dur:.3f}", "-i", os.path.join(W, spk + ".wav"), wav], check=True)
            subprocess.run([tqconfig.whisper_python(), tqconfig.premiere_mcp("server", "src", "transcribe.py"), "--input", wav, "--output", js, "--model", "large-v3", "--language", "en", "--ffmpeg", tqconfig.ffmpeg()],
                           check=True, stderr=subprocess.DEVNULL)
            d = json.load(open(js, encoding="utf-8")); words = []
            for g in d["segments"]:
                for w in g["words"]:
                    ws, we = w["start"] + s0, w["end"] + s0
                    if ws < x - 0.3 or we > y + 0.3: continue                       # only inside the hole
                    if any(ws < e and we > s for s, e in existing): continue        # never duplicate a kept word
                    words.append({"start": round(ws, 3), "end": round(we, 3), "word": " " + w["word"].strip(), "prob": w.get("prob", 0)})
            if words:
                tr["segments"].append({"start": words[0]["start"], "end": words[-1]["end"], "text": " ".join(w["word"].strip() for w in words),
                                       "words": words, "fill": "large-v3"})
                added += len(words)
        tr["segments"].sort(key=lambda g: g["start"])
        shutil.copy(f, f + ".orig")
        json.dump(tr, open(f, "w", encoding="utf-8"), ensure_ascii=False)
        print(spk, f"added {added} words from {len(hs)} holes (original kept as {os.path.basename(f)}.orig)")

if __name__ == "__main__": main()
