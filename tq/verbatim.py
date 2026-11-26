# Verbatim (CrisperWhisper) transcript loader.
# CrisperWhisper quirks: tokens carry leading punctuation (",I"), and fillers are glued to the next word
# ("[UM]he", "[UH][UH]"): the token's time span covers the filler AND the word.
import json, os, re

FILLER_TAG = re.compile(r"\[(UM|UH|HMM|MM|AH|ER)\]", re.I)
PLAIN_FILLERS = {"um", "uh", "umm", "uhm", "uhh", "hmm", "mm", "er", "erm", "ah"}

def norm(t): return re.sub(r"[^a-z']", "", t.lower())

def load(cfg, spk):
    W = cfg["work"]
    f = (cfg.get("transcripts_verbatim") or cfg["transcripts"])[spk]
    import tqconfig; off = tqconfig.shift_s(cfg, spk)
    d = json.load(open(os.path.join(W, f), encoding="utf-8"))
    out = []
    for gi, g in enumerate(d["segments"]):
        if g["end"] + off < 0: continue
        for w in g["words"]:
            raw = w["word"].strip().lstrip(",.;:!? ")
            fillers = [m.group(1).lower() for m in FILLER_TAG.finditer(raw)]
            word = FILLER_TAG.sub("", raw).strip()
            if not fillers and re.sub(r"[^a-z]", "", raw.lower()) in PLAIN_FILLERS:   # Nemotron writes "uh," not "[UH]"
                fillers = [re.sub(r"[^a-z]", "", raw.lower())[:2]]; word = ""
            out.append({"spk": spk, "s": w["start"] + off, "e": w["end"] + off, "raw": raw,
                        "w": word, "n": norm(word), "fillers": fillers, "g": gi,
                        "punct_end": bool(re.search(r"[.?!,]$", word))})
    return out
