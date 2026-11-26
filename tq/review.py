# Review panel: the editor approves / undoes risky cuts, picks detector candidates, edits cold open / chapters / shorts.
# Usage: python tq/review.py episodes/EPxx.json [--port 8790]   -> open http://127.0.0.1:8790
# Reads  <work>/edl.json (cuts the EDL applied), <work>/detections.json (tq/detect.py candidates), the config.
# Writes <work>/review.json (cut/mute decisions, edited edges a2/b2, user_cuts = your own cut/mute sections; build_edl.py
# applies them on the next run)
#        and chapters / shorts / cold_open back into the episode config (it is in git).
# Only listens on 127.0.0.1.
import json, os, sys, io, wave, argparse, threading, subprocess
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import verbatim as VB
import tqconfig
import truth as TR

FPS = 30; SR = 16000; HSR = 48000; PAD = 2.5   # SR: analysis copies; HSR: what you listen to
RISKY = {"manual": "high", "restart": "high", "repeat": "medium", "um(acoustic)": "medium", "filler": "low"}

def load_pcm(path, shift_s=0.0):
    w = wave.open(path); assert w.getframerate() == SR and w.getnchannels() == 1, path
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    n = int(round(shift_s * SR))
    if n > 0: x = np.concatenate([np.zeros(n, np.float32), x])
    elif n < 0: x = x[-n:]
    return x

def hq_track(W, raw, spk):
    """48 kHz mono 16-bit copy of the raw recording (the 16 kHz <speaker>.wav copies are for analysis only).
    Made once with ffmpeg, then memory-mapped."""
    p = os.path.join(W, f"{spk}_48k.wav")
    if not os.path.exists(p):
        print(f"extracting {spk} audio at 48 kHz (once)...", flush=True)
        subprocess.run([tqconfig.ffmpeg(), "-v", "error", "-y", "-i", raw, "-vn", "-ac", "1", "-ar", str(HSR), "-c:a", "pcm_s16le", p + ".part.wav"], check=True)
        os.replace(p + ".part.wav", p)
    w = wave.open(p); assert w.getframerate() == HSR and w.getnchannels() == 1 and w.getsampwidth() == 2, p
    n = w.getnframes(); w.close()
    off = os.path.getsize(p) - 2 * n          # the data chunk is the last chunk in ffmpeg's wav
    return np.memmap(p, np.int16, "r", off, (n,))

def cut_id(kind, spk, a, b): return f"{kind}|{spk}|{a:.2f}|{b:.2f}"

class Episode:
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path; self.lock = threading.Lock()
        self.cfg = tqconfig.load(cfg_path); W = self.cfg["work"]; self.W = W
        self.E = json.load(open(os.path.join(W, "edl.json")))
        dp = os.path.join(W, "detections.json")
        self.D = json.load(open(dp)) if os.path.exists(dp) else {"cuts": [], "mutes": []}
        rp = os.path.join(W, "review.json")
        self.R = json.load(open(rp)) if os.path.exists(rp) else {"decisions": {}}
        vp = os.path.join(W, "verify.json")   # tq/verify_cuts.py: each cut rendered + re-transcribed
        self.V = json.load(open(vp, encoding="utf-8")) if os.path.exists(vp) else {}
        self.S = tqconfig.speakers(self.cfg); sh = {s: tqconfig.shift_s(self.cfg, s) for s in self.S}
        self.pcm = {s: load_pcm(os.path.join(W, s + ".wav"), sh[s]) for s in self.S}
        n = min(len(v) for v in self.pcm.values())
        self.pcm = {k: v[:n] for k, v in self.pcm.items()}
        # per-speaker gain so both mics play at a similar level
        self.gain = {k: 0.12 / max(1e-4, float(np.percentile(np.abs(v[::50]), 99))) for k, v in self.pcm.items()}
        self.words = sorted(sum((VB.load(self.cfg, s) for s in self.S), []), key=lambda w: w["s"])
        self.kept = TR.union([[a / FPS, b / FPS] for a, b in self.E["segments"]])
        # 10 ms dB per mic over its own threshold (> 0 = audible on that mic), same scale as build_edl
        self.thr = self.cfg["mic_threshold_db"]
        self.db = {}
        for k, v in self.pcm.items():
            m = len(v) // 160; r = np.sqrt((v[:m * 160].reshape(m, 160) ** 2).mean(1)) + 1e-3 / 32768
            self.db[k] = 20 * np.log10(r) - self.thr[k]
        raw = self.cfg["raw"]
        self.hq = {k: hq_track(W, raw[k], k) for k in self.S}
        self.hq_off = dict(sh)   # reference time t = file time + off
        self.items = self._items()

    def seg(self, spk, a, b):
        """float32 samples of one mic for reference time [a, b) at 48 kHz, gain applied, zeros outside the file"""
        x = self.hq[spk]; n = int(round((b - a) * HSR)); i = int(round((a - self.hq_off[spk]) * HSR))
        out = np.zeros(max(0, n), np.float32); lo, hi = max(0, i), min(len(x), i + n)
        if hi > lo: out[lo - i:hi - i] = x[lo:hi].astype(np.float32) * (self.gain[spk] / 32768)
        return out

    def ctx(self, a, b, n=5, spk=None):
        """The words a cut removes, plus n words before/after it (all speakers, in time order)."""
        def ins(w): return w["s"] < b and w["e"] > a and (min(w["e"], b) - max(w["s"], a)) >= 0.4 * max(0.01, w["e"] - w["s"])
        inside = [i for i, w in enumerate(self.words) if ins(w)]
        if inside: lo, hi = inside[0], inside[-1]
        else:
            lo = next((i for i, w in enumerate(self.words) if w["s"] >= b), len(self.words)); hi = lo - 1
        out = []
        for i in range(max(0, lo - n), min(len(self.words), hi + 1 + n)):
            w = self.words[i]
            shown = w["raw"]; inn = lo <= i <= hi and ins(w)
            if inn and w["fillers"] and w["w"] and (min(w["e"], b) - max(w["s"], a)) < 0.9 * (w["e"] - w["s"]):
                # glued "[UH]For" where only the filler part is cut: show the two parts separately
                tags = "".join(f"[{t.upper()}]" for t in w["fillers"])
                lead = w["raw"].startswith("[")
                part = {"t": round(w["s"], 2), "spk": w["spk"], "w": tags, "in": True}
                rest = {"t": round(w["s"], 2), "spk": w["spk"], "w": w["w"], "in": False}
                out.extend([part, rest] if lead else [rest, part]); continue
            out.append({"t": round(w["s"], 2), "spk": w["spk"], "w": shown, "in": inn})
        if not inside:   # sound-only cut: mark where it sits
            k = next((j for j, o in enumerate(out) if o["t"] >= b), len(out)); out.insert(k, {"t": a, "spk": "", "w": "(sound)", "in": True})
        return out

    def _items(self):
        items = []
        # cuts the EDL applied (only those that are really out of the timeline)
        manual = [(a, b) for a, b, _, why in self.E["cuts"] if why.startswith("manual:")]
        for a, b, spk, why in self.E["cuts"]:
            kind = why.split(":")[0]
            if kind != "manual" and any(x - 0.05 <= a and b <= y + 0.05 for x, y in manual): continue   # inside an editorial cut: moot
            if TR.overlap(a, b, self.kept) > 0.5 * (b - a): continue      # undone later (hot cut) -> not in the edit
            items.append({"id": cut_id("edl:" + kind, spk, a, b), "group": "edl", "kind": kind, "spk": spk, "a": a, "b": b,
                          "note": why.split(":", 1)[1] if ":" in why else "", "risk": RISKY.get(kind, "low"), "default": "cut"})
        for a, b, spk, kind, txt, conf in self.D["cuts"]:
            items.append({"id": cut_id("det:" + kind, spk, a, b), "group": "det", "kind": kind, "spk": spk, "a": a, "b": b,
                          "note": txt, "risk": conf, "default": "cut" if conf == "high" else "keep"})
        for a, b, spk, kind, txt, conf in self.D["mutes"]:
            items.append({"id": cut_id("mute:" + kind, spk, a, b), "group": "mute", "kind": kind, "spk": spk, "a": a, "b": b,
                          "note": txt, "risk": conf, "default": "mute" if conf == "high" else "keep"})
        # whole-program check (tq/check_render.py): the edit rendered and re-transcribed, differences tied to joins
        cp = os.path.join(self.W, "check_render.json")
        CR = json.load(open(cp, encoding="utf-8")) if os.path.exists(cp) else None
        flagged = [(j["src_gap"][0], j["src_gap"][1], j) for j in (CR or {}).get("flagged_joins", [])]
        for it in items:
            it["render"] = None
            if CR and it["group"] == "edl":
                j = next((j for x, y, j in flagged if x - 0.1 <= it["a"] and it["b"] <= y + 0.1), None)
                it["render"] = {"status": "flag", "missing": j["missing"], "extra": j["extra"], "filler": j["filler_left"]} if j else {"status": "pass"}
        for it in items:
            v = self.V.get(it["id"])
            it["verify"] = v["status"] if v else "unchecked"; it["verify_reason"] = v["reason"] if v else ""
            it["verify_words"] = v.get("words", []) if v else []
            # automatic cuts that fail the check are undone by default; editorial (manual) cuts stay, flagged
            if it["group"] == "edl" and it["kind"] != "manual" and it["verify"] == "fail": it["default"] = "keep"
            if it["group"] == "det" and it["default"] == "cut" and it["verify"] != "pass": it["default"] = "keep"
            # needs a look: anything a human should hear. Automatic cuts that pass both checks, confident candidates that
            # pass, and low-confidence candidates/mutes (left out by default) do not.
            r = it["render"]
            it["look"] = bool(
                it["kind"] == "manual" or it["group"] == "user"
                or (it["group"] == "edl" and (it["verify"] != "pass" or (r and r["status"] == "flag")))
                or (it["group"] in ("det", "mute") and it["risk"] == "medium"))
        for u in self.R.get("user_cuts", []):
            items.append(self.user_item(u))
        for it in items:
            d = self.R["decisions"].get(it["id"])
            # a saved decision that only repeated its default then follows the current default (the detector may have
            # improved since, e.g. cough mutes now pre-ticked); one the editor changed stays as set
            if it["group"] != "user":
                it["decision"] = d["action"] if d and d["action"] != d.get("default", it["default"]) else it["default"]
            if d and d.get("a2") is not None: it["a2"], it["b2"] = d["a2"], d["b2"]
            if d and d.get("fade") is not None: it["fade"] = d["fade"]
            ea, eb = it.get("a2", it["a"]), it.get("b2", it["b"])
            it["ctx"] = self.ctx(ea, eb)
            if "a2" in it or it["group"] == "user": it["check"] = self.check(ea, eb)
        items.sort(key=lambda x: x["a"])
        return items

    def user_item(self, u):
        # your own sections: kind "cut" (removed from the program) or "mute" (silences spk's mic there); anchor = the row
        # they were drawn in, so the panel keeps them in that row
        kind = u.get("kind", "cut")
        return {"id": cut_id("user:" + kind, u["spk"], u["a"], u["b"]), "group": "user", "kind": kind, "spk": u["spk"],
                "a": u["a"], "b": u["b"], "note": "", "risk": "-", "default": kind, "decision": u.get("decision", kind),
                "verify": "unchecked", "verify_reason": "", "verify_words": [], "look": True,
                **({"anchor": u["anchor"]} if u.get("anchor") else {}), **({"fade": u["fade"]} if u.get("fade") is not None else {})}

    def edge_db(self, t):
        """loudest mic at t (+-20 ms), dB over its threshold"""
        i = int(round(t * 100)); return round(float(max(v[max(0, i - 2):i + 2].max(initial=-99) for v in self.db.values())), 1)

    def check(self, a, b):
        """Quick check of any cut span from the aligned words and the sound (no GPU)."""
        ws = self.words
        # words the aligner squeezed to < 50 ms (e.g. "Just" glued after an "uh") are a point at their start: removed only
        # if the cut runs clearly past it
        removes = [w for w in ws if w["raw"] and w["s"] >= a - 0.02 and
                   (w["e"] <= b + 0.02 if w["e"] - w["s"] >= 0.05 else w["s"] < b - 0.04)]
        through = [w for w in ws if w["e"] - w["s"] >= 0.05 and w["s"] < b - 0.03 and w["e"] > a + 0.03 and w not in removes]
        ea, eb = self.edge_db(a), self.edge_db(b)
        # an edge is "on sound" when it is close to the speech level around it (mic floors differ a lot between mics and rooms)
        i0, i1 = max(0, int((a - 0.5) * 100)), int((b + 0.5) * 100)
        peak = float(max(v[i0:i1].max(initial=-99) for v in self.db.values()))
        lim = max(6.0, peak - 20)
        warn = []
        if through: warn.append("cuts through: " + " ".join(w["raw"] for w in through))
        if ea > lim: warn.append(f"start lands on sound ({ea - peak:.0f} dB under the speech)")
        if eb > lim: warn.append(f"end lands on sound ({eb - peak:.0f} dB under the speech)")
        return {"removes": [w["raw"] for w in removes], "warn": warn, "edge_db": [ea, eb]}

    def wave_data(self, a, b, bin_s=0.001):
        n = int(bin_s * HSR)
        peaks = {}
        for k in self.pcm:
            x = np.abs(self.seg(k, a, b)); m = len(x) // n
            peaks[k] = np.round(x[:m * n].reshape(m, n).max(1), 3).tolist() if m else []
        thr = {k: round(10 ** (self.thr[k] / 20) * self.gain[k], 4) for k in self.pcm}
        words = [{"s": round(w["s"], 3), "e": round(w["e"], 3), "w": w["raw"], "spk": w["spk"]} for w in self.words if w["e"] > a and w["s"] < b]
        return {"a": a, "b": b, "bin": bin_s, "peaks": peaks, "thr": thr, "words": words}

    def data(self):
        c = self.cfg
        PAL = ["#5fb37c", "#a487f0", "#f0a35e", "#5ea8f0", "#e06c8c", "#c9c95a"]
        spk = [{"id": s, "name": tqconfig.speaker(c, s).get("name", s), "color": tqconfig.speaker(c, s).get("color") or PAL[i % len(PAL)]}
               for i, s in enumerate(self.S)]
        return {"episode": c["episode"], "speakers": spk, "items": self.items, "chapters": c.get("chapters", []), "shorts": c.get("shorts", []),
                "cold_open": c.get("cold_open_frames", []), "show": c["show"], "saved_at": self.R.get("saved_at")}

    def wav(self, a, b, cut=None, mute=None, solo=None):
        """Mix of both mics for [a,b]. cut=(x,y): remove that span (picture cut). mute=(x,y,spk): silence one mic there."""
        a = max(0.0, a); SR = HSR
        tracks = {k: self.seg(k, a, b) if not solo or k == solo else np.zeros(int(round((b - a) * HSR)), np.float32) for k in self.pcm}
        i = int(round(a * SR))
        mutes = mute if isinstance(mute, list) else ([mute] if mute else [])
        for x, y, spk in mutes:
            xi, yi = max(0, int(x * SR) - i), max(0, int(y * SR) - i)
            tracks[spk] = tracks[spk].copy(); tracks[spk][xi:yi] = 0
        mix = sum(tracks[s] for s in self.S)
        cuts = cut if isinstance(cut, list) else ([cut] if cut else [])
        for c in sorted(cuts, reverse=True):     # last first, so earlier indices stay valid
            xi, yi = max(0, int(c[0] * SR) - i), max(0, int(c[1] * SR) - i)
            f = int(0.01 * SR)
            left, right = mix[:xi].copy(), mix[yi:].copy()
            if len(left) > f and len(right) > f:   # 10 ms crossfade like a real edit
                ramp = np.linspace(0, 1, f, dtype=np.float32)
                left[-f:] = left[-f:] * (1 - ramp) + right[:f] * ramp; right = right[f:]
            mix = np.concatenate([left, right])
        mix = np.clip(mix, -1, 1)
        buf = io.BytesIO(); w = wave.open(buf, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()
        return buf.getvalue()

    def wav_segments(self, segs):
        parts = [np.frombuffer(self.wav(a, b)[44:], np.int16) for a, b in segs]
        x = np.concatenate(parts) if parts else np.zeros(1, np.int16)
        buf = io.BytesIO(); w = wave.open(buf, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(HSR)
        w.writeframes(x.tobytes()); w.close(); return buf.getvalue()

    def save(self, body):
        with self.lock:
            dec = {}
            user = [u for u in body.get("user_cuts", []) if u["b"] - u["a"] >= 0.03]
            self.items = [it for it in self.items if it["group"] != "user"] + [self.user_item(u) for u in user]
            by_id = {it["id"]: it for it in self.items}
            edits = body.get("edits", {}); fades = body.get("fades", {})
            for iid, action in body.get("decisions", {}).items():
                it = by_id.get(iid)
                if not it or it["group"] == "user": continue
                it["decision"] = action
                dec[iid] = {"action": action, "group": it["group"], "kind": it["kind"], "spk": it["spk"], "a": it["a"], "b": it["b"],
                            "default": it["default"]}
                if iid in fades: it["fade"] = dec[iid]["fade"] = int(fades[iid])
                else: it.pop("fade", None)
                if iid in edits:
                    it["a2"], it["b2"] = edits[iid]; dec[iid]["a2"], dec[iid]["b2"] = edits[iid]
                else: it.pop("a2", None); it.pop("b2", None)
            import datetime
            self.R = {"decisions": dec, "user_cuts": user, "saved_at": datetime.datetime.now().isoformat(timespec="seconds")}
            json.dump(self.R, open(os.path.join(self.W, "review.json"), "w"), indent=1)
            changed = []
            for key in ("chapters", "shorts"):
                if key in body and body[key] != self.cfg.get(key):
                    self.cfg[key] = body[key]; changed.append(key)
            if "cold_open" in body and body["cold_open"] != self.cfg.get("cold_open_frames"):
                self.cfg["cold_open_frames"] = body["cold_open"]; changed.append("cold_open_frames")
            if changed: tqconfig.save(self.cfg, self.cfg_path)
            n = sum(1 for d in dec.values() if d["action"] != d["default"])
            return {"ok": True, "saved_at": self.R["saved_at"], "decisions": len(dec), "changed_from_default": n,
                    "edited": len(edits), "user_cuts": len(user), "config_changed": changed}

def PAGE():   # read fresh per request (edits to review.html show on reload)
    with open(os.path.join(HERE, "review.html"), encoding="utf-8") as f: return f.read()

def handler(ep):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def send(self, code, body, ctype):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/": return self.send(200, PAGE().encode(), "text/html; charset=utf-8")
                if u.path == "/data": return self.send(200, json.dumps(ep.data()).encode(), "application/json")
                if u.path == "/audio":
                    a, b = float(q["a"]), float(q["b"])
                    cut = [tuple(map(float, c.split(","))) for c in q["cut"].split(";") if c] if q.get("cut") else None
                    mute = None
                    if q.get("mute"): mute = [(float(x), float(y), s_) for x, y, s_ in (m.split(",") for m in q["mute"].split(";") if m)]
                    return self.send(200, ep.wav(a, b, cut, mute, q.get("solo")), "audio/wav")
                if u.path == "/wave":
                    return self.send(200, json.dumps(ep.wave_data(float(q["a"]), float(q["b"]))).encode(), "application/json")
                if u.path == "/check":
                    a, b = float(q["a"]), float(q["b"])
                    return self.send(200, json.dumps(dict(ep.check(a, b), ctx=ep.ctx(a, b))).encode(), "application/json")
                if u.path == "/audio_segs":
                    segs = [tuple(map(float, p.split(","))) for p in q["segs"].split(";") if p]
                    return self.send(200, ep.wav_segments(segs), "audio/wav")
                self.send(404, b"not found", "text/plain")
            except Exception as e:
                self.send(500, str(e).encode(), "text/plain")
        def do_POST(self):
            if self.path != "/save": return self.send(404, b"", "text/plain")
            n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
            self.send(200, json.dumps(ep.save(body)).encode(), "application/json")
    return H

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    ep = Episode(a.config)
    print(f"{len(ep.items)} items ({sum(1 for i in ep.items if i['group']=='edl')} EDL cuts, "
          f"{sum(1 for i in ep.items if i['group']=='det')} candidates, {sum(1 for i in ep.items if i['group']=='mute')} mutes)", flush=True)
    print(f"review panel: http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), handler(ep)).serve_forever()
