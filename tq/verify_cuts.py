# Checks every cut against the audio itself.
# For each cut: take the words just before and after it (aligned transcript), and force-align them (MMS_FA, GPU)
# to the original audio and to the edited audio (cut removed, 10 ms crossfade, both mics mixed).
# A kept word that the cut clipped or removed loses most of its alignment score; untouched words keep it.
# (Re-transcribing with Whisper was tried first: on 4 s clips it loops/hallucinates and flagged good cuts.)
# Run with the GPU Python (gpu_python in local.json): <gpu_python> tq/verify_cuts.py episodes/EPxx.json
# Writes <work>/verify.json: {item_id: {status: pass|fail|unclear, reason, words:[[word, score_before, score_after]]}}
import json, os, re, sys, wave, time
import numpy as np
import torch
from torchaudio.pipelines import MMS_FA as bundle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tqconfig
import verbatim as VB
import truth as TR

SR = 16000; FPS = 30; CTX = 1.2          # seconds of kept speech checked on each side of a cut
DROP = 0.35                               # kept word fails if score_after < DROP * score_before
TAG = re.compile(r"\[(UM|UH|HMM|MM|AH|ER)\]", re.I)

def cut_id(kind, spk, a, b): return f"{kind}|{spk}|{a:.2f}|{b:.2f}"   # same as review.py

def load(path, shift_s=0.0):
    w = wave.open(path); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    n = int(round(shift_s * SR))
    return np.concatenate([np.zeros(n, np.float32), x]) if n > 0 else (x[-n:] if n < 0 else x)

def letters(raw):
    if TAG.fullmatch(raw.strip()): return "um" if "UM" in raw.upper() else "uh"
    t = re.sub(r"\d", lambda m: " " + "zero one two three four five six seven eight nine".split()[int(m.group(0))] + " ", raw.lower())
    return re.sub(r"[^a-z]", "", t)

def main():
    cfg = tqconfig.load(sys.argv[1]); W = cfg["work"]; t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = bundle.get_model(with_star=False).to(dev); tok = bundle.get_tokenizer(); aligner = bundle.get_aligner()
    pcm = {s: load(os.path.join(W, s + ".wav"), tqconfig.shift_s(cfg, s)) for s in tqconfig.speakers(cfg)}
    n = min(len(v) for v in pcm.values())
    mix = sum(v[:n] * (0.12 / max(1e-4, float(np.percentile(np.abs(v[::50]), 99)))) for v in pcm.values())
    words = sorted(sum((VB.load(cfg, s) for s in tqconfig.speakers(cfg)), []), key=lambda w: w["s"])
    E = json.load(open(os.path.join(W, "edl.json"))); kept = TR.union([[a / FPS, b / FPS] for a, b in E["segments"]])
    D = json.load(open(os.path.join(W, "detections.json")))
    manual = [(a, b) for a, b, _, why in E["cuts"] if why.startswith("manual:")]
    items = []
    for a, b, spk, why in E["cuts"]:
        if TR.overlap(a, b, kept) > 0.5 * (b - a): continue
        if not why.startswith("manual:") and any(x - 0.05 <= a and b <= y + 0.05 for x, y in manual): continue
        items.append((cut_id("edl:" + why.split(":")[0], spk, a, b), a, b))
    for a, b, spk, kind, txt, conf in D["cuts"]:
        items.append((cut_id("det:" + kind, spk, a, b), a, b))

    def scores(audio, ws):
        """mean alignment score per word (list of letter strings) on this audio."""
        x = torch.from_numpy(np.clip(audio, -1, 1)).unsqueeze(0).to(dev)
        with torch.inference_mode():
            em, _ = model(x)
        spans = aligner(em[0], tok(ws))
        return [float(np.mean([s.score for s in sp])) for sp in spans]

    out = {}
    INTENDED = ("edl:manual", "edl:repeat", "edl:restart", "det:fragment")   # these cuts are meant to remove real words
    FILL = {"um", "uh"}
    for iid, a, b in items:
        # hard rule first: never cut through a word, never remove a real word unless the cut is meant to
        part = [w for w in words if w["e"] - w["s"] >= 0.05 and w["s"] < b - 0.03 and w["e"] > a + 0.03 and not (w["s"] >= a - 0.02 and w["e"] <= b + 0.02)]
        # words squeezed to < 50 ms by the aligner count as a point at their start
        full = [w for w in words if w["s"] >= a - 0.02 and (w["e"] <= b + 0.02 if w["e"] - w["s"] >= 0.05 else w["s"] < b - 0.02)
                and letters(w["raw"]) not in FILL and letters(w["raw"])]
        if part:
            out[iid] = {"status": "fail", "reason": "cuts through: " + " ".join(w["raw"] for w in part[:4]), "words": []}; continue
        if full and not iid.startswith(INTENDED):
            out[iid] = {"status": "fail", "reason": "removes real word(s): " + " ".join(w["raw"] for w in full[:4]), "words": []}; continue
        left = [w for w in words if a - CTX <= w["s"] and w["e"] <= a + 0.03 and letters(w["raw"])]
        right = [w for w in words if b - 0.03 <= w["s"] and w["e"] <= b + CTX and letters(w["raw"])]
        inside = [w for w in words if w["s"] > a - 0.03 and w["e"] < b + 0.03 and letters(w["raw"])]
        if not left and not right:
            out[iid] = {"status": "unclear", "reason": "no words next to the cut", "words": []}; continue
        lo = min([w["s"] for w in left] + [a]) - 0.3; hi = max([w["e"] for w in right] + [b]) + 0.3
        i0, i1 = max(0, int(lo * SR)), min(n, int(hi * SR)); xa, xb = int(a * SR) - i0, int(b * SR) - i0
        before = mix[i0:i1].copy()
        if b - a > 20:   # long editorial cut: align only the edges of the removed part
            inside = [w for w in inside if w["s"] < a + 1.5 or w["e"] > b - 1.5]
            before = np.concatenate([before[:xa + int(1.5 * SR)], before[xb - int(1.5 * SR):]]); xb = xa + int(3.0 * SR)
        f = int(0.01 * SR); L, R = before[:xa].copy(), before[xb:].copy()
        if len(L) > f and len(R) > f:
            ramp = np.linspace(0, 1, f, dtype=np.float32); L[-f:] = L[-f:] * (1 - ramp) + R[:f] * ramp; R = R[f:]
        after = np.concatenate([L, R])
        kw = left + right
        try:
            sb = scores(before, [letters(w["raw"]) for w in left + inside + right])
            sa = scores(after, [letters(w["raw"]) for w in kw])
        except Exception as ex:
            out[iid] = {"status": "unclear", "reason": f"aligner: {ex}"[:80], "words": []}; continue
        sb_kept = sb[:len(left)] + sb[len(left) + len(inside):]
        rows = [[w["raw"], round(x, 3), round(y, 3)] for w, x, y in zip(kw, sb_kept, sa)]
        lost = [r[0] for r in rows if r[1] >= 0.15 and r[2] < DROP * r[1]]
        edge_ids = ({id(left[-1])} if left else set()) | ({id(right[0])} if right else set())
        weak = [r[0] for r, w in zip(rows, kw) if id(w) in edge_ids and r[1] >= 0.15 and r[2] < 0.6 * r[1] and r[0] not in lost]
        if lost: st, why = "fail", "kept word(s) damaged: " + " ".join(lost[:5])
        elif weak: st, why = "fail", "word at the cut sounds clipped: " + " ".join(weak)
        else: st, why = "pass", ""
        out[iid] = {"status": st, "reason": why, "words": rows}
    json.dump(out, open(os.path.join(W, "verify.json"), "w", encoding="utf-8"), indent=0, ensure_ascii=False)
    from collections import Counter
    print(f"verified {len(out)} cuts in {time.time() - t0:.0f} s:", dict(Counter(v["status"] for v in out.values())))

if __name__ == "__main__": main()
