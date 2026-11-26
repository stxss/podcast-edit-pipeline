# Ground truth from the editor's own timeline, for tuning detectors.
# Time base = reference-audio seconds, same as build_edl.py.
#   his_kept      : picture the editor kept (union of the reference speaker's video tracks)
#   mine_kept     : picture our EDL kept (edl.json segments)
#   extra_removed : mine_kept - his_kept  (what he cut on top of us)  <- detector targets
#   restored      : his_kept - mine_kept  (what we cut and he put back) <- false-positive warnings
#   muted[s]      : audio of speaker s he muted while the picture stayed (backchannels etc.)
import json, os, re
import numpy as np

FPS = 30

def union(iv):
    iv = sorted([list(x) for x in iv]); out = []
    for a, b in iv:
        if out and a <= out[-1][1] + 0.002: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out

def inter(A, B):
    out = []; i = j = 0
    while i < len(A) and j < len(B):
        a = max(A[i][0], B[j][0]); b = min(A[i][1], B[j][1])
        if b > a: out.append([a, b])
        if A[i][1] < B[j][1]: i += 1
        else: j += 1
    return out

def minus(A, B):
    out = []
    for a, b in A:
        cur = a
        for c, d in B:
            if d <= cur or c >= b: continue
            if c > cur: out.append([cur, c])
            cur = max(cur, d)
        if cur < b: out.append([cur, b])
    return out

def total(iv): return sum(b - a for a, b in iv)

def overlap(a, b, iv):
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in iv)

def load(cfg, tracks_file=None, edl_file="edl.json"):
    """cfg = episode config dict. cfg["editor_truth"]: tracks (timeline dump in the work dir), intro_end (sequence time
    where the show starts on the editor's timeline), video_tracks (indexes of the reference speaker's video tracks),
    video_offset (reference time = source time of that video + offset), audio_tracks {speaker: audio track index}."""
    import tqconfig
    SP = tqconfig.speakers(cfg)
    W = cfg["work"]
    et = cfg.get("editor_truth", {})
    S = json.load(open(os.path.join(W, tracks_file or et["tracks"])))
    E = json.load(open(os.path.join(W, edl_file)))
    intro_end = et.get("intro_end", 0.0)
    ref_v = et["video_tracks"]
    v_off = et.get("video_offset", 0.0)
    a_tr = et.get("audio_tracks") or {s: str(i) for i, s in enumerate(SP)}
    V, A = S["V"], S["A"]
    his = union([[x[2] + v_off, x[2] + v_off + (x[1] - x[0])] for k in ref_v for x in V[k] if x[0] >= intro_end - 0.01])
    mine = union([[a / FPS, b / FPS] for a, b in E["segments"]])
    off = {s: tqconfig.shift_s(cfg, s) for s in SP}
    his_audio = {s: union([[x[2] + off[s], x[2] + off[s] + x[1] - x[0]] for x in A[a_tr[s]] if x[0] >= intro_end - 0.01]) for s in SP}
    my_audio = {s: union([[a / FPS, b / FPS] for a, b in E["audio"][s]]) for s in SP}
    t = {"his_kept": his, "mine_kept": mine,
         "extra_removed": [r for r in minus(mine, his) if r[1] - r[0] >= 0.05],
         "restored": [r for r in minus(his, mine) if r[1] - r[0] >= 0.05],
         "muted": {s: [r for r in minus(inter(my_audio[s], his), his_audio[s]) if r[1] - r[0] > 0.1] for s in SP},
         "my_audio": my_audio, "edl": E}
    return t

def score(cands, targets, kept, min_frac=0.5):
    """cands: [(a,b,label...)] cut proposals in LA. targets: intervals the editor removed.
    kept: intervals the editor kept (a cut there is a false positive).
    A candidate is a hit if >= min_frac of it lies in targets, a miss-fire if >= min_frac lies in kept."""
    hits = fp = other = 0; hit_t = 0.0; rows = []
    for c in cands:
        a, b = c[0], c[1]; d = max(1e-6, b - a)
        ft = overlap(a, b, targets) / d; fk = overlap(a, b, kept) / d
        lab = "hit" if ft >= min_frac else ("fp" if fk >= min_frac else "other")
        if lab == "hit": hits += 1; hit_t += overlap(a, b, targets)
        elif lab == "fp": fp += 1
        else: other += 1
        rows.append(list(c) + [lab, round(ft, 2)])
    covered = sum(1 for x, y in targets if overlap(x, y, [c[:2] for c in cands]) >= 0.3 * (y - x))
    return {"cands": len(cands), "hit": hits, "fp": fp, "other": other,
            "precision": round(hits / max(1, hits + fp), 2),
            "targets": len(targets), "targets_covered": covered, "recall": round(covered / max(1, len(targets)), 2),
            "seconds_hit": round(hit_t, 1), "seconds_target": round(total(targets), 1)}, rows
