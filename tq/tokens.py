# Indexed token stream of the whole conversation for the editorial agent (idea from github.com/crxssrazr93/verbatim-cut,
# written fresh for one mic per speaker + aligned words). The agent answers with TOKEN INDEX RANGES, never times or quoted text,
# so a cut can't land inside a word and a phrase said twice is never ambiguous.
#   python tq/tokens.py <episode>                 -> <work>/tokens.json + tokens.txt
#   python tq/tokens.py <episode> --cuts cuts.json  -> prints manual_cuts [[a, b, why]] in reference-audio seconds
#     cuts.json: [[first_token, last_token, "why"], ...]  (inclusive)
# Token types: word, filler (um/uh...), repeat (first take of "I I" / "we we should we should"), silence, noise
# (sound on a mic with no word: breath, laugh, dropped filler - deliberately not labelled further).
import json, os, sys, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import verbatim as VB
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tqconfig

FILLERS = {"um", "uh", "umm", "uhm", "uhh", "er", "erm", "hmm", "mm", "ah"}
NUMBER_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million oh".split())
MIN_GAP = 0.15; RUN = 0.10   # gaps shorter than MIN_GAP are ignored; silence/noise runs shorter than RUN are absorbed

def norm(t): return re.sub(r"[^a-z0-9']", "", t.lower())

def db10(path):
    import wave
    w = wave.open(path); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    k = len(x) // 160; return 20 * np.log10(np.sqrt((x[:k * 160].reshape(k, 160) ** 2).mean(1)) + 1e-3) - 20 * np.log10(32768)

def build(cfg):
    W = cfg["work"]; thr = cfg["mic_threshold_db"]; S = tqconfig.speakers(cfg)
    def onref(s):   # this mic's level on the reference clock
        x = db10(os.path.join(W, s + ".wav")); sh = int(round(tqconfig.shift_s(cfg, s) * 100))
        return np.concatenate([np.full(sh, -120.0), x]) if sh > 0 else x[-sh:]
    lv = {s: onref(s) for s in S}
    n = min(len(v) for v in lv.values()); loud = np.max([lv[s][:n] - thr[s] for s in S], axis=0)   # > 0: a mic hears something
    ws = sorted(sum((VB.load(cfg, s) for s in S), []), key=lambda w: w["s"])
    s0, s1 = cfg["show"]["start"], cfg["show"]["end"]
    ws = [w for w in ws if w["e"] > s0 and w["s"] < s1]
    toks = []
    def gap(a, b):
        if b - a < MIN_GAP: return
        i, j = int(a * 100), min(n, int(b * 100))
        runs = []
        for k in range(i, j):
            kind = "noise" if loud[k] > 6 else "silence"
            if runs and runs[-1][0] == kind: runs[-1][2] = k + 1
            else: runs.append([kind, k, k + 1])
        merged = []
        for r in runs:
            if merged and (r[2] - r[1] < RUN * 100 or merged[-1][0] == r[0]): merged[-1][2] = r[2]
            else: merged.append(r)
        for kind, x, y in merged:
            t = {"type": kind, "start": round(x / 100, 2), "end": round(y / 100, 2)}
            if kind == "noise": t["db"] = round(float(loud[x:y].max()), 0)
            toks.append(t)
    cur = s0
    for w in ws:
        if w["s"] > cur: gap(cur, w["s"])
        typ = "filler" if (w["fillers"] and not w["w"]) or norm(w["raw"]) in FILLERS else "word"
        toks.append({"type": typ, "spk": w["spk"], "text": w["raw"], "start": round(w["s"], 3), "end": round(w["e"], 3)})
        cur = max(cur, w["e"])
    gap(cur, s1)
    # repeats: first copy of an immediately repeated 1-3 word phrase by the same speaker within 2 s
    wi = [k for k, t in enumerate(toks) if t["type"] in ("word", "filler")]
    for m in (3, 2, 1):
        for p in range(len(wi) - 2 * m + 1):
            A, B = [toks[k] for k in wi[p:p + m]], [toks[k] for k in wi[p + m:p + 2 * m]]
            if any(t["type"] == "repeat" for t in A + B) or len({t["spk"] for t in A + B}) > 1: continue
            if all(norm(t["text"]) in NUMBER_WORDS for t in A): continue   # "twenty twenty four" is a year, not a stutter
            if [norm(t["text"]) for t in A] == [norm(t["text"]) for t in B] and B[-1]["end"] - A[0]["start"] <= 2.0 and all(norm(t["text"]) for t in A):
                for t in A: t["type"] = "repeat"
    for i, t in enumerate(toks): t["i"] = i
    return toks

def text_view(toks):
    out = []; spk = None
    for t in toks:
        if t.get("spk") and t["spk"] != spk:
            spk = t["spk"]; out.append(f"\n{spk.upper()} @{int(t['start'] // 60)}:{t['start'] % 60:04.1f}:")
        d = t["end"] - t["start"]
        if t["type"] == "word": out.append(f"[{t['i']}]{t['text']}")
        elif t["type"] in ("filler", "repeat"): out.append(f"[{t['i']}]<{t['type']}:{t['text']}>")
        elif t["type"] == "noise": out.append(f"[{t['i']}]<noise {d:.2f}s>")
        elif d >= 0.3: out.append(f"[{t['i']}]<sil {d:.1f}s>")
    return " ".join(out).strip() + "\n"

def main():
    cfg = tqconfig.load(sys.argv[1]); W = cfg["work"]
    if "--cuts" in sys.argv:
        toks = json.load(open(os.path.join(W, "tokens.json"), encoding="utf-8"))
        cuts = json.load(open(sys.argv[sys.argv.index("--cuts") + 1], encoding="utf-8"))
        out = []
        for a, b, why in cuts:
            A, B = toks[a], toks[b]
            # edges sit in the gaps next to the range (half way to the neighbouring token), never inside a kept word
            prev_e = toks[a - 1]["end"] if a else A["start"]; next_s = toks[b + 1]["start"] if b + 1 < len(toks) else B["end"]
            s = A["start"] if A["type"] in ("silence", "noise") else max((prev_e + A["start"]) / 2, A["start"] - 0.05)
            e = B["end"] if B["type"] in ("silence", "noise") else min((B["end"] + next_s) / 2, B["end"] + 0.05)
            out.append([round(s, 2), round(e, 2), why])
        print(json.dumps(out, indent=1)); return
    toks = build(cfg)
    json.dump(toks, open(os.path.join(W, "tokens.json"), "w", encoding="utf-8"), ensure_ascii=False)
    open(os.path.join(W, "tokens.txt"), "w", encoding="utf-8").write(text_view(toks))
    from collections import Counter
    print(len(toks), "tokens", dict(Counter(t["type"] for t in toks)), "->", os.path.join(W, "tokens.txt"))

if __name__ == "__main__": main()
