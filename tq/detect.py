# Cut detectors on top of build_edl.py, tuned on an editor's own pass over a real episode.
# Usage: python tq/detect.py <episode> [--score] [--dump]   (tunables: P below, overridden by show.json "detect")
#   writes <work>/detections.json: {"cuts": [[a,b,speaker,kind,text]], "mutes": [[a,b,speaker,kind,text]]}
#   --score compares with the editor's timeline (cfg["editor_truth"]) and prints precision/recall per detector.
# Time base = the reference speaker's audio seconds. Candidates are only proposed inside material the current EDL keeps.
#
# Detectors
#   filler      verbatim [UM]/[UH] tokens. Alone -> cut the token. Glued to a hesitation word ("[UM]But") -> cut both.
#               Glued to a content word ("[UH]technical") -> cut only the filler part, split at the quietest frame.
#               Hesitation words right after (before a pause) or right before ("you know [UH]") go with it.
#   fragment    a short run of hesitation words between two pauses, followed by the real start ("I And | to your point").
#   restart     the same word(s) said twice with a pause between ("That ... That business").
#   nonspeech   the active mic is loud but nobody says a word (cough, breath, click, laugh, noise).
#   backchannel (mute) a speaker's isolated "yeah/right/hmm" or wordless mic noise while the other holds the floor.
#   noise       (mute) a cough/breath/noise on the mic of a speaker who isn't talking; short+loud = pre-ticked.
import json, os, sys, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import verbatim as VB
import truth as TR

HOP = 0.01; FPS = 30
HESIT = {"and", "but", "so", "i", "you", "know", "like", "mean", "the", "a", "to", "that", "it's", "we", "because",
         "okay", "ok", "well", "just", "um", "uh", "this", "it", "in", "of", "or", "what", "my", "there's", "anyways",
         "anyway", "yeah", "then", "now", "right"}
BACKCH = {"yeah", "right", "mm", "hmm", "mhm", "mmhmm", "okay", "ok", "sure", "exactly", "yep", "true", "totally",
          "uhhuh", "yes", "nice", "wow", "oh"}
P = dict(  # tunables (seconds / dB)
    pause_before=0.25, pause_after=0.25, frag_max_words=3, frag_max_span=1.5, frag_next_within=2.5,
    restart_gap=0.25, restart_max=3,
    ns_min=0.12, ns_max=2.5, ns_margin_db=6.0, ns_word_pad=0.12,
    bc_iso=1.0, bc_other_within=1.2, bc_max=1.2,
    edge_tail=0.03, edge_head=0.03,
    filler_min=0.30,            # shorter filler cuts were mostly left in by the editor
    frag_min=0.30,
    ns_max_cut=0.55, ns_word_near=3.2,   # real coughs/breaths sit in a pause between words; long word-free stretches are transcript holes
    restart_on=False,           # matched none of the editor's cuts
    bc_audible_db=10.0,         # backchannel must be audible on its own mic
    nm_min=0.15, nm_max=2.5,    # wordless noise on a mic while the OTHER speaker talks -> mute
    # Every speaker by default: coughs etc. on the mic of whoever isn't talking. Muting the idle mic is safe: the talker's
    # own mic carries the voice, so it only removes that track's cough/noise/bleed. On a real episode neither loudness,
    # length nor own-vs-other mic level separated the ones the editor muted from the rest.
    nm_speakers=None,           # speakers that get noise mutes; None = all
    nm_cough_db=15.0,           # noise mute this loud (dB over the mic threshold) and this short -> high (pre-ticked)
    nm_cough_max=1.2,
    edge_quiet_min=0.08,        # quieter gap than this at an edge -> "tight": never pre-ticked, the editor decides
    edge_quiet_floor=0.04,      # no quiet gap at all -> try a valley
    valley_db=18.0,             # a dip this far below the sound on both sides can take a (tight) cut edge; if the words run together there's no clean cut
)

def load_db(work, name):
    import wave
    w = wave.open(os.path.join(work, name + ".wav")); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    h = 160; k = len(x) // h; r = np.sqrt((x[:k * h].reshape(k, h) ** 2).mean(1)) + 1e-3
    return 20 * np.log10(r / 32768)

def idx(t): return int(round(t / HOP))

class Ctx:
    def __init__(self, cfg, edl_file="edl.json"):
        self.cfg = cfg; W = cfg["work"]
        self.E = json.load(open(os.path.join(W, edl_file)))
        self.kept = TR.union([[a / FPS, b / FPS] for a, b in self.E["segments"]])
        import tqconfig
        self.S = tqconfig.speakers(cfg)
        self.audio = {s: TR.union([[a / FPS, b / FPS] for a, b in self.E["audio"][s]]) for s in self.S}
        from vadlib import shift
        self.db = {s: shift(load_db(W, s), idx(tqconfig.shift_s(cfg, s))) for s in self.S}
        self.N = min(len(v) for v in self.db.values())
        self.thr = cfg["mic_threshold_db"]
        self.words = {s: [w for w in VB.load(cfg, s)] for s in self.S}
        self.raw = {s: list(v) for s, v in self.words.items()}   # unfiltered: short backchannels fail the bleed test
        # drop transcript words where the mic wasn't active (bleed / hallucination), like build_edl
        for s in self.S:
            act = self.db[s][:self.N] > self.thr[s]
            # bleed = the mic never really opens. Smeared tokens (0.96 s for a 0.25 s word) have a low active
            # fraction but a clear peak, so keep a word if either test passes.
            def real(w):
                i, j = idx(w["s"]), max(idx(w["e"]), idx(w["s"]) + 1)
                return j < self.N and (act[i:j].mean() >= 0.3 or self.db[s][i:j].max() > self.thr[s] + 10)
            self.words[s] = [w for w in self.words[s] if real(w)]
        self.all = sorted(sum((self.words[s] for s in self.S), []), key=lambda w: w["s"])
        # every mic below threshold = a place where a cut can land
        self.quiet = np.logical_and.reduce([self.db[s][:self.N] < self.thr[s] for s in self.S])

    def token_ok(self, s, w):
        """Word timing looks trustworthy: short, or mostly voiced on its own mic. CrisperWhisper sometimes smears a token
        over the pause before it and shifts the next words one burst early."""
        d = w["e"] - w["s"]
        if d < 0.35: return True
        i, j = idx(w["s"]), idx(w["e"])
        return (self.db[s][i:j] > self.thr[s]).mean() >= 0.3

    def quiet_run(self, t, reach=0.3):
        """(start, end) of the quiet stretch around t (both mics), or None if t is in sound."""
        i = idx(t)
        if i <= 0 or i >= self.N or not self.quiet[i]: return None
        lo = hi = i; lim = int(reach / HOP)
        while lo > 0 and i - lo < lim and self.quiet[lo - 1]: lo -= 1
        while hi < self.N - 1 and hi - i < lim and self.quiet[hi + 1]: hi += 1
        return lo * HOP, (hi + 1) * HOP

    def snap_edge(self, t, search=0.12):
        """Nearest quiet stretch (both mics) of >= edge_quiet_floor within +-search s: (middle, length), else (None, 0)."""
        best = None
        for k in range(0, int(search / HOP) + 1):
            for tt in (t - k * HOP, t + k * HOP):
                r = self.quiet_run(tt)
                if r and r[1] - r[0] >= P["edge_quiet_floor"]:
                    m = (r[0] + r[1]) / 2
                    if best is None or abs(m - t) < abs(best[0] - t): best = (m, r[1] - r[0])
            if best is not None: return best
        # no silent gap: accept a clear valley (>= valley_db below the sound on both sides) -> tight edge
        lvl = np.max([self.db[s][:self.N] - self.thr[s] for s in self.S], axis=0)
        i0, i1 = idx(t - 0.06), idx(t + 0.06)
        if i0 < 15 or i1 > self.N - 15: return None, 0.0
        m = i0 + int(np.argmin(lvl[i0:i1]))
        left, right = lvl[m - 15:m].max(), lvl[m + 1:m + 16].max()
        if min(left, right) - lvl[m] >= P["valley_db"]: return m * HOP, 0.0
        return None, 0.0

    def sound_span(self, s, a, b, back=0.25, fwd=0.35):
        """Grow [a,b] to the whole sound burst(s) it touches on this mic. Aligned word starts are reliable;
        CTC alignment marks ends short, so the cut must run to where the sound actually stops."""
        th = self.thr[s]; i = idx(a); j = idx(b)
        lo = i
        while lo > 0 and i - lo < back / HOP and self.db[s][lo - 1] > th: lo -= 1
        hi = j
        while hi < self.N - 1 and hi - j < fwd / HOP and self.db[s][hi] > th: hi += 1
        return lo * HOP, hi * HOP

    def other(self, s): return [x for x in self.S if x != s]
    def in_kept(self, a, b, iv=None): return TR.overlap(a, b, iv or self.kept) >= 0.5 * (b - a)
    def talking(self, s, a, b, pad=0.0):
        if isinstance(s, list): return any(self.talking(x, a, b, pad) for x in s)
        return any(w["s"] < b + pad and w["e"] > a - pad and (w["w"] or w["fillers"]) for w in self.words[s])
    def peak(self, s, a, b):
        i, j = idx(a), max(idx(b), idx(a) + 1)
        return float(self.db[s][i:j].max()) - self.thr[s]
    def quietest(self, s, a, b):
        i, j = idx(a), max(idx(b), idx(a) + 1)
        return (i + int(np.argmin(self.db[s][i:j]))) * HOP

def gap_before(ws, i): return ws[i]["s"] - ws[i - 1]["e"] if i > 0 else 9.0
def gap_after(ws, i): return ws[i + 1]["s"] - ws[i]["e"] if i + 1 < len(ws) else 9.0
def text(ws, i, j): return " ".join(w["raw"] for w in ws[i:j])

def det_filler(c, s):
    ws = c.words[s]; out = []; used = set()
    for i, w in enumerate(ws):
        if not w["fillers"] or i in used: continue
        if not all(c.token_ok(s, ws[k]) for k in (i - 1, i + 1) if 0 <= k < len(ws)): continue   # neighbours smeared -> timing unreliable
        raw = w["raw"]; a, b = w["s"], w["e"]
        lead = raw.startswith("[")
        if w["n"] == "" or w["n"] in HESIT:
            pass                                            # cut the whole token
        elif lead:                                          # "[UH]technical": cut the filler part only
            b = c.quietest(s, a + 0.12, max(a + 0.13, b - 0.18))
        else:                                               # "go[UM]": cut the trailing filler
            a = c.quietest(s, a + 0.15, max(a + 0.16, b - 0.12))
        j = i + 1
        # absorb hesitation words after it when a pause follows them (the real start comes after the pause)
        if lead and (w["n"] == "" or w["n"] in HESIT):
            k = j
            while k < len(ws) and k - j < 2 and (ws[k]["n"] in HESIT or ws[k]["fillers"]) and ws[k]["s"] - ws[k - 1]["e"] < 0.8:
                if gap_after(ws, k) >= P["pause_after"]: j = k + 1; b = ws[k]["e"]
                k += 1
        # absorb "you know" / "and" / "I mean" right before it if a pause precedes them
        k = i - 1; a0 = a
        while k >= 0 and i - k <= 2 and ws[k]["n"] in {"and", "know", "you", "mean", "i", "so", "but", "like"} and ws[k + 1]["s"] - ws[k]["e"] < 0.6:
            if gap_before(ws, k) >= P["pause_before"]: a0 = ws[k]["s"]
            k -= 1
        a = a0
        used.update(range(i, j))
        a, b = c.sound_span(s, a, b)
        if b - a < P["filler_min"] * 0.7: continue
        out.append([a - P["edge_head"], b + P["edge_tail"], s, "filler", text(ws, max(0, i - 1), j + 1)])
    return out

def det_fragment(c, s):
    ws = c.words[s]; out = []
    i = 0
    while i < len(ws):
        if gap_before(ws, i) < P["pause_before"] or ws[i]["fillers"]: i += 1; continue
        j = i
        while j < len(ws) and j - i < P["frag_max_words"] and ws[j]["n"] in HESIT and not ws[j]["fillers"]:
            if gap_after(ws, j) >= P["pause_after"] and j + 1 < len(ws) and ws[j + 1]["s"] - ws[i]["s"] <= P["frag_next_within"] + (ws[j]["e"] - ws[i]["s"]):
                span = ws[j]["e"] - ws[i]["s"]
                nxt = ws[j + 1]
                # the restart must not itself be a hesitation-only tail, and the other speaker must be quiet
                if span <= P["frag_max_span"] and nxt["s"] - 0.10 - ws[i]["s"] >= P["frag_min"] and not ws[j]["punct_end"] and not c.talking(c.other(s), ws[i]["s"], nxt["s"], 0.1):
                    out.append([ws[i]["s"] - P["edge_head"], nxt["s"] - 0.10, s, "fragment", text(ws, max(0, i - 1), j + 2)])
                    i = j; break
            j += 1
        i += 1
    return out

def det_restart(c, s):
    ws = c.words[s]; out = []
    for i in range(len(ws)):
        for k in range(1, P["restart_max"] + 1):
            if i + 2 * k > len(ws): break
            A = [w["n"] for w in ws[i:i + k]]; B = [w["n"] for w in ws[i + k:i + 2 * k]]
            if A != B or not all(A): continue
            g = ws[i + k]["s"] - ws[i + k - 1]["e"]
            if g < P["restart_gap"] or g > 1.5: continue
            if k == 1 and A[0] in BACKCH: continue
            out.append([ws[i]["s"] - P["edge_head"], ws[i + k]["s"] - 0.08, s, "restart", text(ws, max(0, i - 1), i + 2 * k + 1)])
            break
    return out

def det_nonspeech(c):
    """Loud on a mic, no word from anyone. Picture cut if the other speaker is quiet, else an audio mute."""
    from vadlib import runs, close
    cuts, mutes = [], []
    cover = np.zeros(c.N, bool)
    for w in sum((c.raw[s] for s in c.S), []):      # any transcript word counts as speech here (never cut a word as 'noise')
        cover[max(0, idx(w["s"] - P["ns_word_pad"])):idx(w["e"] + P["ns_word_pad"])] = True
    for s in c.S:
        loud = (c.db[s][:c.N] > c.thr[s] + P["ns_margin_db"]) & ~cover
        from vadlib import runs, close
        loud = close(loud, 8)
        for a, b in runs(loud):
            A, B = a * HOP, b * HOP; d = B - A
            if not (P["ns_min"] <= d <= P["ns_max"]): continue
            if not c.in_kept(A, B, c.audio[s]): continue
            before = [w["e"] for w in c.all if w["e"] <= A + 0.01]; after = [w["s"] for w in c.all if w["s"] >= B - 0.01]
            if not before or not after or A - before[-1] > P["ns_word_near"] or after[0] - B > P["ns_word_near"]: continue
            o = c.other(s)
            if c.talking(o, A, B, 0.15):
                continue                                   # handled by the mute pass below
            elif c.in_kept(A, B) and d <= P["ns_max_cut"]:
                cuts.append([A - 0.05, B + 0.05, s, "nonspeech", ""])
    # mute pass: this mic is loud, its owner says nothing (raw transcript), the other speaker is talking
    for s in P["nm_speakers"]:
        own = np.zeros(c.N, bool)
        for w in c.raw[s]:
            if w["n"] in BACKCH: continue                  # backchannels count as noise here
            own[max(0, idx(w["s"] - P["ns_word_pad"])):idx(w["e"] + P["ns_word_pad"])] = True
        loud = close((c.db[s][:c.N] > c.thr[s] + P["ns_margin_db"]) & ~own, 15)
        for a, b in runs(loud):
            A, B = a * HOP, b * HOP
            if not (P["nm_min"] <= B - A <= P["nm_max"]): continue
            if not c.in_kept(A, B, c.audio[s]): continue
            if c.talking(c.other(s), A, B, 0.15):
                mutes.append([A - 0.05, B + 0.08, s, "noise", "cough/noise" if c.peak(s, A, B) >= P["nm_cough_db"] and B - A <= P["nm_cough_max"] else ""])
    return cuts, mutes

def det_backchannel(c, s):
    ws = c.raw[s]; o = c.other(s); out = []
    i = 0
    while i < len(ws):
        j = i
        while j < len(ws) and ws[j]["n"] in BACKCH and (j == i or ws[j]["s"] - ws[j - 1]["e"] < 0.4): j += 1
        if j > i:
            a, b = ws[i]["s"], ws[j - 1]["e"]
            iso = gap_before(ws, i) >= P["bc_iso"] and (j >= len(ws) or ws[j]["s"] - b >= P["bc_iso"])
            if iso and b - a <= P["bc_max"] and c.peak(s, a, b) > P["bc_audible_db"] and c.talking(o, a, b, P["bc_other_within"]) and c.in_kept(a, b, c.audio[s]):
                out.append([a - 0.05, b + 0.08, s, "backchannel", text(ws, i, j)])
            i = j
        else: i += 1
    return out

def confidence(x):
    """high = pre-ticked in the review panel; medium/low = shown unticked. From hit rates measured against an editor's own pass."""
    k, d = x[3], x[1] - x[0]
    if str(x[4]).startswith("[tight edge]"): return "tight"
    if k == "filler": return "high" if d >= 0.7 else ("medium" if d >= 0.45 else "low")
    if "backchannel" in k.split("+"): return "high"     # also when merged with a noise next to it
    if k == "noise": return "high" if x[4] == "cough/noise" else "medium"   # cough/noise on the mic of a speaker who isn't talking
    if k == "nonspeech": return "medium" if 0.2 <= d <= 0.5 else "low"
    return "low"

def clip_to(cands, iv):
    out = []
    for c in cands:
        for x, y in TR.inter([[c[0], c[1]]], iv):
            if y - x >= 0.06: out.append([round(x, 3), round(y, 3)] + c[2:])
    return out

def run(cfg):
    c = Ctx(cfg)
    cuts = []; mutes = []
    for s in c.S:
        cuts += det_filler(c, s) + det_fragment(c, s) + (det_restart(c, s) if P["restart_on"] else [])
        mutes += det_backchannel(c, s)
    ns_c, ns_m = det_nonspeech(c)
    cuts += ns_c; mutes += ns_m
    ok = []
    for x in cuts:     # every cut edge must land in a real quiet gap (both mics), snapped to its middle
        (a2, qa), (b2, qb) = c.snap_edge(x[0]), c.snap_edge(x[1])
        if a2 is None or b2 is None or b2 - a2 < 0.08: continue
        if x[3] != "nonspeech" and c.quiet[idx(a2):idx(b2)].all(): continue   # nothing audible left to remove
        y = [a2, b2] + x[2:]
        if min(qa, qb) < P["edge_quiet_min"]: y[4] = "[tight edge] " + y[4]
        ok.append(y)
    cuts = clip_to(sorted(ok), c.kept)
    merged = []
    for m in sorted(mutes, key=lambda m: (c.S.index(m[2]), m[0])):
        if merged and merged[-1][2] == m[2] and m[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], m[1]); merged[-1][3] = "+".join(sorted(set(merged[-1][3].split("+") + [m[3]]))); merged[-1][4] = (merged[-1][4] + " " + m[4]).strip()
        else: merged.append(list(m))
    mutes = [[round(m[0], 3), round(m[1], 3)] + m[2:] for m in merged]
    cuts = [x + [confidence(x)] for x in cuts]
    mutes = [m + [confidence(m)] for m in mutes]
    return c, cuts, mutes

def report(cfg, cuts, mutes, dump=False, ctx=None):
    global Cx; Cx = ctx
    t = TR.load(cfg)
    kinds = sorted({x[3] for x in cuts})
    print(f"{'detector':12} cands  hit   fp other  prec | targets covered recall  s_hit/s_target")
    for k in kinds + ["ALL"]:
        cs = [x for x in cuts if k == "ALL" or x[3] == k]
        r, rows = TR.score(cs, t["extra_removed"], t["his_kept"])
        print(f"{k:12} {r['cands']:5} {r['hit']:4} {r['fp']:4} {r['other']:5} {r['precision']:5} | {r['targets']:7} {r['targets_covered']:7} {r['recall']:6}  {r['seconds_hit']}/{r['seconds_target']}")
        if dump and k != "ALL":
            for row in rows: print("   ", row[-2].upper(), row[0], row[1], row[2], row[4][:80], row[-1])
    # restored spots we'd cut again
    rest = [x for x in cuts if TR.overlap(x[0], x[1], t["restored"]) > 0.05]
    for conf in ("high", "medium", "low"):
        cs = [x for x in cuts if x[-1] == conf]
        r, _ = TR.score(cs, t["extra_removed"], t["his_kept"])
        print(f"  conf={conf:6} cands {r['cands']:3} hit {r['hit']:3} fp {r['fp']:3} prec {r['precision']}  seconds {r['seconds_hit']}")
    print("cuts touching his restores:", len(rest), [x[:2] + [x[3]] for x in rest])
    for s in Cx.S:
        his_audio_kept = TR.minus(t["my_audio"][s], t["muted"][s])
        ms = [m for m in mutes if m[2] == s]
        for k in sorted({m[3] for m in ms}) + ["ALL"]:
            mm = [m for m in ms if k == "ALL" or m[3] == k]
            r, rows = TR.score(mm, t["muted"][s] + t["extra_removed"], his_audio_kept)
            print(f"MUTE {s:4} {k:11} cands {r['cands']:3} hit {r['hit']:3} fp {r['fp']:3} prec {r['precision']} | muted targets {len(t['muted'][s])}")
            if dump and k != "ALL":
                for row in rows: print("   ", row[-2].upper(), row[0], row[1], row[4][:60], row[-1])
        aud = [r for r in t["muted"][s] if float(Cx.db[s][int(r[0] / HOP):int(r[1] / HOP)].max()) > Cx.thr[s]]
        cov = sum(1 for x, y in aud if TR.overlap(x, y, [m[:2] for m in ms]) >= 0.3 * (y - x))
        print(f"MUTE {s} recall (audible targets): {cov}/{len(aud)}  (all targets {len(t['muted'][s])})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--score", action="store_true"); ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    import tqconfig; cfg = tqconfig.load(a.config)
    for k, v in tqconfig.show(cfg).get("detect", {}).items(): P[k] = v
    if P["nm_speakers"] is None: P["nm_speakers"] = tqconfig.speakers(cfg)
    sys.path.insert(0, HERE); os.chdir(cfg["work"])
    c, cuts, mutes = run(cfg)
    json.dump({"params": P, "cuts": cuts, "mutes": mutes}, open(os.path.join(cfg["work"], "detections.json"), "w"), indent=0)
    print(f"{len(cuts)} cut candidates, {len(mutes)} mute candidates -> detections.json")
    if a.score: report(cfg, cuts, mutes, a.dump, c)
