# Whole-program cut check: render the edited audio exactly as the EDL places it (picture segments + each mic's gated
# audio clips), transcribe the WHOLE render in one pass, and diff it against the words that should survive.
# (Per-cut 4 s clips made Whisper hallucinate; a full-length render reads like the original, so the ASR behaves.)
# Every difference is mapped back to the nearest join; differences right at a join are the cut problems:
#   missing  a kept word next to a join is not heard -> the cut clipped it
#   extra    a word next to a join is heard that should be gone -> leaked syllable / surviving filler
# Run with the GPU Python:
#   <gpu_python from local.json> tq/check_render.py episodes/EPxx.json [--model NAME]
# Writes <work>/check_render.json {joins: [...], summary} and <work>/check/render.wav (16 kHz, the edit as heard).
import json, os, re, sys, time, wave, difflib, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import verbatim as VB
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tqconfig

SR = 16000; FPS = 30; NEAR = 0.6     # a difference within NEAR s (output time) of a join belongs to that join
FILL = {"um", "uh", "umm", "uhm", "uhh", "er", "erm", "hmm", "mm", "ah"}
PROMPT = "Um, uh, so, like, you know, I- I mean, uh, we- we were, um, yeah."

def norm(t):   # filler tags as words, numbers as spoken ("2026" = "twenty twenty six": transcripts write them differently)
    import align as AL
    t = re.sub(r"\[(um|uh|hmm|mm|ah|er)\]", r" \1 ", t.lower())
    if re.search(r"\d,\d", t): t = t.replace(",", "")
    t = re.sub(r"(\d+)(st|nd|rd|th|s)?", AL.say_number, t)
    return [x for x in re.sub(r"[^a-z' ]", " ", t).split() if x]

def load(path, shift_s=0.0):
    w = wave.open(path); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    n = int(round(shift_s * SR))
    return np.concatenate([np.zeros(n, np.float32), x]) if n > 0 else (x[-n:] if n < 0 else x)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--model", default="mobiuslabsgmbh/faster-whisper-large-v3-turbo")
    ap.add_argument("--no-vad", action="store_true", default=True)
    a = ap.parse_args(); t0 = time.time()
    cfg = tqconfig.load(a.config); W = cfg["work"]
    E = json.load(open(os.path.join(W, "edl.json")))
    pcm = {s: load(os.path.join(W, s + ".wav"), tqconfig.shift_s(cfg, s)) for s in tqconfig.speakers(cfg)}
    gain = {k: 0.12 / max(1e-4, float(np.percentile(np.abs(v[::50]), 99))) for k, v in pcm.items()}
    # output timeline: segments back to back; each mic plays only inside its audio clips (the gate)
    segs = E["segments"]; out_len = sum(b - a for a, b in segs)
    render = np.zeros(int(out_len / FPS * SR) + SR, np.float32)
    seg_out = []; pos = 0
    for fa, fb in segs: seg_out.append((fa, fb, pos)); pos += fb - fa
    def to_out(f):     # source frame -> output frame (None if cut)
        for fa, fb, p in seg_out:
            if fa <= f < fb: return p + (f - fa)
        return None
    mic_on = {}
    for spk in tqconfig.speakers(cfg):
        on = np.zeros(int(segs[-1][1] / FPS * SR) + SR, bool)
        for xa, ya in E["audio"][spk]: on[int(xa / FPS * SR):int(ya / FPS * SR)] = True
        mic_on[spk] = on
    f = int(0.01 * SR); ramp = np.linspace(0, 1, f, dtype=np.float32)
    for fa, fb, p in seg_out:
        i, j, o = int(fa / FPS * SR), int(fb / FPS * SR), int(p / FPS * SR)
        piece = sum(pcm[s][i:j] * mic_on[s][i:j] * gain[s] for s in pcm)
        if len(piece) > 2 * f: piece = piece.copy(); piece[:f] *= ramp; piece[-f:] *= ramp[::-1]
        render[o:o + len(piece)] += piece
    render = np.clip(render[:int(out_len / FPS * SR)], -1, 1)
    os.makedirs(os.path.join(W, "check"), exist_ok=True)
    wp = os.path.join(W, "check", "render.wav"); ww = wave.open(wp, "wb"); ww.setnchannels(1); ww.setsampwidth(2); ww.setframerate(SR)
    ww.writeframes((render * 32767).astype(np.int16).tobytes()); ww.close()
    # intended words: source words whose middle is kept AND whose mic is open there, in output order
    intended = []
    for w in sum((VB.load(cfg, s) for s in tqconfig.speakers(cfg)), []):
        m = (w["s"] + w["e"]) / 2; o = to_out(int(m * FPS))
        if o is None or not mic_on[w["spk"]][min(len(mic_on[w["spk"]]) - 1, int(m * SR))]: continue
        for t in norm(w["raw"]): intended.append({"t": t, "o": o / FPS + (m * FPS - int(m * FPS)) / FPS, "src": round(m, 2), "spk": w["spk"]})
    intended.sort(key=lambda x: x["o"])
    # transcribe the render
    import asr_whisper as AW     # faster-whisper (NVIDIA/CPU) or mlx-whisper (Apple Silicon)
    eng = AW.pick_engine("auto"); AW.PROMPT = PROMPT
    sg, _ = AW.transcribe(eng, a.model if eng == "faster" else AW.MODELS["mlx"]["turbo"], render, "none")
    heard = [{"t": t, "o": (x["start"] + x["end"]) / 2} for s in sg for x in s["words"] for t in norm(x["word"])]
    t_asr = time.time() - t0
    # joins in output seconds, with the source gap they skip
    joins = [{"o": p / FPS, "src_gap": [round(seg_out[k - 1][1] / FPS, 3), round(fa / FPS, 3)]} for k, (fa, fb, p) in enumerate(seg_out) if k]
    jo = np.array([j["o"] for j in joins])
    for j in joins: j.update(missing=[], extra=[], filler_left=[])
    A = [x for x in intended if x["t"] not in FILL]; B = [x for x in heard if x["t"] not in FILL]
    sm = difflib.SequenceMatcher(None, [x["t"] for x in A], [x["t"] for x in B], autojunk=False)
    far = 0
    def near(o):
        k = int(np.argmin(np.abs(jo - o))) if len(jo) else -1
        return k if k >= 0 and abs(jo[k] - o) <= NEAR else None
    def same(xs, ys):   # spelling / contraction differences between two transcripts are not cut problems
        p, q = "".join(x["t"] for x in xs).replace("'", ""), "".join(y["t"] for y in ys).replace("'", "")
        p, q = p.replace("are", "re").replace("is", "s").replace("not", "nt").replace("will", "ll"), q.replace("are", "re").replace("is", "s").replace("not", "nt").replace("will", "ll")
        return bool(p) and bool(q) and difflib.SequenceMatcher(None, p, q).ratio() >= 0.7
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal": continue
        if op == "replace" and same(A[i1:i2], B[j1:j2]): continue
        for x in A[i1:i2]:
            k = near(x["o"]); (joins[k]["missing"].append(x["t"]) if k is not None else None); far += k is None
        for x in B[j1:j2]:
            k = near(x["o"]); (joins[k]["extra"].append(x["t"]) if k is not None else None); far += k is None
    # fillers heard right at a join where a filler was meant to be cut
    for x in heard:
        if x["t"] in FILL:
            k = near(x["o"])
            if k is not None and abs(jo[k] - x["o"]) < 0.35: joins[k]["filler_left"].append(x["t"])
    # a join only counts when the words differ (a missing word that is also heard as extra = same word, timing noise)
    bad = []
    for j in joins:
        miss, ext = list(j["missing"]), list(j["extra"])
        for t in list(miss):
            if t in ext: miss.remove(t); ext.remove(t)
        j["missing"], j["extra"] = miss, ext
        if miss or ext or j["filler_left"]: bad.append(j)
    res = {"model": a.model, "joins": len(joins), "flagged": len(bad), "diffs_away_from_joins": far,
           "match_ratio": round(sm.ratio(), 3), "seconds": round(time.time() - t0, 1), "asr_seconds": round(t_asr, 1), "flagged_joins": bad}
    json.dump(res, open(os.path.join(W, "check_render.json"), "w", encoding="utf-8"), indent=1)
    print(f"render {out_len / FPS / 60:.1f} min, {len(joins)} joins, {len(bad)} flagged, {far} differences away from joins, "
          f"word match {sm.ratio():.3f}, {time.time() - t0:.0f} s")
    for j in bad[:25]:
        s = j["src_gap"]; print(f"  join at {int(j['o'] // 60)}:{j['o'] % 60:05.2f} (source {s[0]:.2f}-{s[1]:.2f}) missing {j['missing']} extra {j['extra']} filler {j['filler_left']}")

if __name__ == "__main__": main()
