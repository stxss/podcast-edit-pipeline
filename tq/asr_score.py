# Transcription bake-off, step 2: score each candidate transcript against the audio itself (no human reference needed).
#   <gpu_python from local.json> tq/asr_score.py episodes/EPxx.json
# Per candidate and mic:
#   missed   seconds of clear own speech (own mic >= thr+10 dB and >= 6 dB over the other mic) with no word within 0.25 s,
#            and the number of such holes >= 1 s  (CrisperWhisper's known weakness)
#   silent   words over audio where the own mic is quiet (< thr+3 dB): invented words
#   bleed    words over audio where only the OTHER mic is loud: the other host transcribed from bleed
#   fillers  um/uh tokens (the cut list needs them)
#   match    forced-alignment score of the candidate's own text on the audio (MMS_FA): median over words, and the share
#            of words under 0.1. Numbers are excluded (MMS_FA does not hear them).
#   agree    share of words matching the other candidates (difflib on the normalised word sequence, fillers excluded)
# Writes <work>/bake/score.json and bake/report.md, including a listen list: the biggest spots where the candidates disagree,
# with a link that plays them in the review panel (needs tq/review.py running on 8790).
import json, os, re, sys, glob, difflib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tqconfig

FILL = {"um", "uh", "umm", "uhm", "uhh", "er", "erm", "hmm", "mm"}
TAG = re.compile(r"\[(UM|UH|HMM|MM|AH|ER)\]", re.I)

def norm_tokens(raw):
    raw = TAG.sub(lambda m: " " + m.group(1).lower() + " ", raw)
    return [t for t in re.sub(r"[^a-z0-9' ]", " ", raw.lower()).split() if t]

def load_any(path):
    d = json.load(open(path, encoding="utf-8")); out = []
    for g in d["segments"]:
        for w in g.get("words", []):
            if "word" in w: s, e, txt = w["start"], w["end"], w["word"]
            elif "text" in w: s, e, txt = w["start"], w["start"] + w.get("duration", 0), w["text"]   # Premiere
            else: continue
            toks = norm_tokens(txt)
            for k, t in enumerate(toks):   # glued "[UH]Just" -> two tokens sharing the span
                out.append({"s": s, "e": e, "t": t, "raw": txt})
    return sorted(out, key=lambda w: w["s"])

def db10(path):
    import wave
    w = wave.open(path); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    k = len(x) // 160; return 20 * np.log10(np.sqrt((x[:k * 160].reshape(k, 160) ** 2).mean(1)) + 1e-3) - 20 * np.log10(32768)

def main():
    cfg = tqconfig.load(sys.argv[1]); W = cfg["work"]; B = os.path.join(W, "bake"); thr = cfg["mic_threshold_db"]
    S = tqconfig.speakers(cfg); SH = {x: tqconfig.shift_s(cfg, x) for x in S}
    cands = {"crisper (raw)": "{spk}_tr_crisper.json.orig", "crisper + holes filled (current)": "{spk}_tr_crisper.json"}
    for f in sorted(glob.glob(os.path.join(B, f"*_{S[0]}.json"))):
        n = os.path.basename(f)[:-len(f"_{S[0]}.json")]; cands[n] = os.path.join("bake", n + "_{spk}.json")
    cands["premiere"] = "{spk}_tr_premiere.json"
    # dB per mic on its own clock, and the other mic on that clock
    db = {x: db10(os.path.join(W, x + ".wav")) for x in S}
    def other_on(spk):   # the other mics (loudest) moved onto spk's own clock
        outs = []
        for x in S:
            if x == spk: continue
            o = db[x]; k = int(round((SH[x] - SH[spk]) * 100))
            if k > 0: o = np.concatenate([np.full(k, -120.0), o])
            elif k < 0: o = o[-k:]
            m = len(db[spk]); outs.append(np.concatenate([o[:m], np.full(max(0, m - len(o)), -120.0)]))
        return np.max(outs, axis=0)
    res = {}; words_by = {}
    import torch
    from torchaudio.pipelines import MMS_FA as bundle
    import align as AL
    dev = "cuda"; model = bundle.get_model(with_star=False).to(dev); tok = bundle.get_tokenizer(); aligner = bundle.get_aligner()
    for name, pat in cands.items():
        for spk in S:
            p = os.path.join(W, pat.format(spk=spk))
            if not os.path.exists(p): continue
            ws = load_any(p); words_by[(name, spk)] = ws
            own, oth = db[spk] - thr[spk], other_on(spk) - thr[spk]
            clear = (own >= 10) & (db[spk] >= other_on(spk) + 6)
            cov = np.zeros(len(own), bool)
            for w in ws: cov[max(0, int((w["s"] - 0.25) * 100)):int((w["e"] + 0.25) * 100) + 1] = True
            miss = clear & ~cov
            # holes: runs of missed speech, merging gaps < 0.3 s
            holes = []; i = 0; idx = np.where(miss)[0]
            for j in idx:
                if holes and j - holes[-1][1] <= 30: holes[-1][1] = j
                else: holes.append([j, j])
            big = [(a / 100, b / 100) for a, b in holes if b - a >= 100]
            silent = bleed = 0
            for w in ws:
                a, b = int(w["s"] * 100), max(int(w["e"] * 100), int(w["s"] * 100) + 1)
                if own[a:b].max(initial=-99) < 3:
                    if oth[a:b].max(initial=-99) >= 10: bleed += 1
                    else: silent += 1
            fillers = sum(1 for w in ws if w["t"] in FILL)
            # text-audio match: align this candidate's text with the same code the pipeline uses
            tr = json.load(open(p, encoding="utf-8"))
            if "words" in tr["segments"][0] and tr["segments"][0]["words"] and "text" in tr["segments"][0]["words"][0]:   # Premiere -> common format
                tr = {"segments": [{"words": [{"start": w["start"], "end": w["start"] + w.get("duration", 0), "word": w["text"]} for w in g["words"]]} for g in tr["segments"]]}
            units = AL.tokens_from_transcript(tr)
            wav = AL.load_wav(os.path.join(W, spk + ".wav"))
            r = AL.align_speaker(wav, units, model, tok, aligner, dev)
            sc = [x[2] for u, x in zip(units, r) if x and u["letters"] and not re.search(r"\d", u["word"]) and u["letters"] != ["uh"] and u["letters"] != ["um"]]
            sc = np.array(sc) if sc else np.zeros(1)
            res[f"{name}|{spk}"] = dict(words=len(ws), missed_s=round(float(miss.sum()) / 100, 1), holes_1s=len(big), holes=big[:40],
                                        silent=silent, bleed=bleed, fillers=fillers, match_median=round(float(np.median(sc)), 3),
                                        match_low=round(float((sc < 0.1).mean()), 3))
            print(name, spk, {k: v for k, v in res[f"{name}|{spk}"].items() if k != "holes"}, flush=True)
    # agreement + listen list
    names = [x for x in cands if (x, S[0]) in words_by]
    seq = lambda name, spk: [w for w in words_by[(name, spk)] if w["t"] not in FILL]
    agree = {}
    for spk in S:
        for a in names:
            for b in names:
                if a >= b or (a, spk) not in words_by or (b, spk) not in words_by: continue
                A, Bq = seq(a, spk), seq(b, spk)
                m = difflib.SequenceMatcher(None, [w["t"] for w in A], [w["t"] for w in Bq], autojunk=False)
                agree[f"{a} ~ {b}|{spk}"] = round(2 * sum(bl.size for bl in m.get_matching_blocks()) / max(1, len(A) + len(Bq)), 3)
    # listen list: where the current transcript and large-v3 disagree the most (both directions)
    spots = []
    ref, alt = "crisper + holes filled (current)", "large-v3" if ("large-v3", S[0]) in words_by else None
    if alt:
        for spk in S:
            A, Bq = seq(ref, spk), seq(alt, spk)
            m = difflib.SequenceMatcher(None, [w["t"] for w in A], [w["t"] for w in Bq], autojunk=False)
            for op, i1, i2, j1, j2 in m.get_opcodes():
                if op == "equal" or max(i2 - i1, j2 - j1) < 3: continue
                ts = [w["s"] for w in A[i1:i2]] + [w["s"] for w in Bq[j1:j2]]; te = [w["e"] for w in A[i1:i2]] + [w["e"] for w in Bq[j1:j2]]
                s0, e0 = min(ts), max(te)
                if e0 - s0 > 20: continue
                off = SH[spk]
                spots.append(dict(spk=spk, s=round(s0 + off, 2), e=round(e0 + off, 2), size=max(i2 - i1, j2 - j1),
                                  texts={nm: " ".join(r for _, r in dict.fromkeys((w["s"], w["raw"].strip()) for w in words_by[(nm, spk)] if s0 - 0.3 <= w["s"] <= e0 + 0.3))
                                         for nm in names if (nm, spk) in words_by}))
        spots.sort(key=lambda x: -x["size"])
    json.dump({"scores": res, "agree": agree, "spots": spots[:40]}, open(os.path.join(B, "score.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    # report
    L = [f"# Transcription bake-off: {cfg['episode']}", "", "| model | mic | words | missed speech s | holes >= 1 s | invented (silent) | from bleed | fillers | audio match (median) | words matching badly |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in res.items():
        nm, spk = k.split("|")
        L.append(f"| {nm} | {spk} | {v['words']} | {v['missed_s']} | {v['holes_1s']} | {v['silent']} | {v['bleed']} | {v['fillers']} | {v['match_median']} | {round(100 * v['match_low'], 1)}% |")
    L += ["", "## Agreement (share of words in common, fillers excluded)", ""] + [f"- {k}: {v}" for k, v in agree.items()]
    L += ["", f"## Listen list: biggest disagreements between '{ref}' and '{alt}'", ""]
    for sp in spots[:25]:
        L.append(f"### {sp['spk']} {int(sp['s'] // 60)}:{sp['s'] % 60:05.2f}  [play](http://127.0.0.1:8790/audio?a={sp['s'] - 0.5}&b={sp['e'] + 0.5})")
        for nm, t in sp["texts"].items(): L.append(f"- **{nm}**: {t}")
        L.append("")
    open(os.path.join(B, "report.md"), "w", encoding="utf-8").write("\n".join(L))
    print("wrote", os.path.join(B, "report.md"))

if __name__ == "__main__": main()
