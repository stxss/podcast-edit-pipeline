# Forced alignment: keep the transcript's WORDS, take their TIMES from the audio.
# Whisper-type word times are estimates (one word was placed 1 s early, an "uh" merged into the next word), so cuts placed
# on them can hit the neighbouring word. MMS_FA (torchaudio) aligns the known text to the waveform frame by frame.
# Run with the GPU Python (torchaudio installed there):
#   <gpu_python from local.json> tq/align.py episodes/EPxx.json
# Reads  <work>/<transcripts_verbatim[spk]>  (own clock), writes <work>/<spk>_tr_aligned.json (same format, own clock)
# and points the config's transcripts / transcripts_verbatim at it (originals stay as transcripts_unaligned).
import json, os, re, sys, wave
import numpy as np
import torch, torchaudio
from torchaudio.pipelines import MMS_FA as bundle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tqconfig

TAG = re.compile(r"\[(UM|UH|HMM|MM|AH|ER)\]", re.I)
PLAIN = {"um", "uh", "umm", "uhm", "uhh", "hmm", "mm", "er", "erm", "ah"}
def is_filler(t):   # CrisperWhisper tags "[UH]", Nemotron writes "uh,"
    t = t.strip(); return bool(TAG.fullmatch(t)) or re.sub(r"[^a-z]", "", t.lower()) in PLAIN
NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}

def say_int(n):
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000: return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + say_int(n % 100))
    if 1100 <= n <= 2099 and n % 100 and not (2000 <= n < 2010):   # years: 2026 -> twenty twenty six, 1984 -> nineteen eighty four
        return say_int(n // 100) + " " + (say_int(n % 100) if n % 100 >= 10 else "oh " + ONES[n % 10])
    if n < 1000000: return say_int(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + say_int(n % 1000))
    return " ".join(NUM[d] for d in str(n))

def say_number(m):
    n, suf = int(m.group(1)), (m.group(2) or "").lower()
    words = say_int(n) if len(m.group(1)) < 7 else " ".join(NUM[d] for d in m.group(1))
    if suf in ("st", "nd", "rd", "th"):      # 25th -> twenty fifth
        *head, last = words.split()
        last = ORD.get(last, (last[:-1] + "ieth") if last.endswith("y") else last + "th")
        words = " ".join(head + [last])
    return " " + words + " "

def spoken(text):
    """Letters the aligner can match. Numbers as spoken ('25th' spelled per digit dragged the words after it 2 s late)."""
    t = text.lower().replace(",", "") if re.search(r"\d,\d", text) else text.lower()
    t = re.sub(r"(\d+)(st|nd|rd|th|s)?", say_number, t)
    t = re.sub(r"[^a-z' ]", " ", t).replace("'", " ")
    return [x for x in t.split() if x]

def tokens_from_transcript(tr):
    """Transcript words -> alignment units. '[UM]he' -> '[UM]' + 'he'. Keeps original word + times for fallback."""
    out = []
    for g in tr["segments"]:
        for w in g["words"]:
            raw = w["word"].strip()
            parts = []
            pos = 0
            for m in TAG.finditer(raw):
                if raw[pos:m.start()].strip(" ,.;:!?"): parts.append(raw[pos:m.start()])
                parts.append(m.group(0)); pos = m.end()
            if raw[pos:].strip(" ,.;:!?"): parts.append(raw[pos:])
            if not parts: continue
            for p in parts:
                filler = is_filler(p)
                letters = (["um"] if "m" in re.sub(r"[^a-z]", "", p.lower()) else ["uh"]) if filler else spoken(p)
                out.append({"word": " " + p.strip(), "letters": letters, "s0": w["start"], "e0": w["end"]})
    return out

def load_wav(path):
    w = wave.open(path); assert w.getframerate() == 16000 and w.getnchannels() == 1
    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768

def align_speaker(wav, units, model, tokenizer, aligner, dev, chunk_words=40, pad=1.5):
    sr = 16000; frame = 320 / sr   # MMS_FA emission: 20 ms per frame
    res = [None] * len(units)
    idx = [i for i, u in enumerate(units) if u["letters"]]
    # chunks: at most chunk_words, and never across a pause > 2 s (sparse backchannels on the listener's mic would
    # otherwise give one window minutes long, and the aligner stretches single words across the silence)
    chunks = []; cur = []
    for i in idx:
        if cur and (len(cur) >= chunk_words or units[i]["s0"] - units[cur[-1]]["e0"] > 2.0): chunks.append(cur); cur = []
        cur.append(i)
    if cur: chunks.append(cur)
    for k, ids in enumerate(chunks):
        # pad, but never into the neighbouring chunk's speech (a word that ended a chunk was dragged
        # 1.4 s late into the next chunk's 'on November 25th')
        lo_lim = max(units[i]["e0"] for i in chunks[k - 1]) if k else 0.0
        hi_lim = min(units[i]["s0"] for i in chunks[k + 1]) if k + 1 < len(chunks) else len(wav) / sr
        s_first = min(units[i]["s0"] for i in ids); e_last = max(units[i]["e0"] for i in ids)
        a = max(0.0, s_first - pad, min(lo_lim - 0.1, s_first - 0.05))
        b = min(len(wav) / sr, e_last + pad, max(hi_lim + 0.1, e_last + 0.05))
        seg = torch.from_numpy(wav[int(a * sr):int(b * sr)]).unsqueeze(0).to(dev)
        with torch.inference_mode():
            em, _ = model(seg)
        words = [" ".join(units[i]["letters"]).replace(" ", "") for i in ids]   # one aligner 'word' per unit
        try:
            spans = aligner(em[0], tokenizer(words))
        except Exception as ex:
            print("  chunk at", round(a, 1), "failed:", ex); continue
        for i, sp in zip(ids, spans):
            s = a + sp[0].start * frame; e = a + sp[-1].end * frame
            score = float(np.mean([t.score for t in sp]))
            res[i] = (round(s, 3), round(e, 3), round(score, 3))
    return res

def main():
    cfg_path = sys.argv[1]; cfg = tqconfig.load(cfg_path); W = cfg["work"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = bundle.get_model(with_star=False).to(dev); tokenizer = bundle.get_tokenizer(); aligner = bundle.get_aligner()
    src = cfg.get("transcripts_unaligned") or cfg.get("transcripts_verbatim") or cfg["transcripts"]
    out_names = {}
    for spk in tqconfig.speakers(cfg):
        tr = json.load(open(os.path.join(W, src[spk]), encoding="utf-8"))
        units = tokens_from_transcript(tr)
        trust_src = "nemotron" in str(tr.get("model", "")).lower()
        wav = load_wav(os.path.join(W, spk + ".wav"))
        res = align_speaker(wav, units, model, tokenizer, aligner, dev)
        words = []; moved = []; fallback = 0
        for u, r in zip(units, res):
            if r is None:
                fallback += 1; words.append({"start": u["s0"], "end": u["e0"], "word": u["word"], "aligned": False})
            else:
                words.append({"start": r[0], "end": r[1], "word": u["word"], "aligned": True, "score": r[2]}); moved.append(abs(r[0] - u["s0"]))
        # Words must stay in transcript order. A badly matched word (score < 0.05) that lands after the next word
        # or before the previous one goes back to its transcript time, squeezed between its neighbours.
        bad = 0
        for n_, w in enumerate(words):
            if not w["aligned"] or w.get("score", 1) >= 0.05: continue
            prev_e = max([x["end"] for x in words[max(0, n_ - 3):n_] if x["aligned"] and x.get("score", 1) >= 0.05] or [0.0])
            nxt_s = min([x["start"] for x in words[n_ + 1:n_ + 4] if x["aligned"] and x.get("score", 1) >= 0.05] or [1e9])
            # Nemotron's own CTC times hear numbers (MMS_FA doesn't): with it as the source, every unmatched word takes them
            if trust_src or w["start"] < prev_e - 0.02 or w["end"] > nxt_s + 0.02:
                s0, e0 = units[n_]["s0"], units[n_]["e0"]
                s0, e0 = max(s0, prev_e), min(e0, nxt_s)
                if e0 - s0 < 0.04: s0, e0 = prev_e, min(nxt_s, prev_e + 0.1)
                w["start"], w["end"], w["aligned"], w["reordered"] = round(s0, 3), round(max(e0, s0 + 0.02), 3), False, True; bad += 1
        # any remaining pair out of order: the weaker match of the two goes back between its neighbours
        for n_ in range(len(words) - 1):
            w, x = words[n_], words[n_ + 1]
            if not (w["aligned"] and x["aligned"]) or w["start"] <= x["start"] + 0.02: continue
            v, m = (w, n_) if w.get("score", 1) <= x.get("score", 1) else (x, n_ + 1)
            lo_ = words[m - 1]["end"] if m and words[m - 1] is not v else 0.0
            hi_ = words[m + 1]["start"] if m + 1 < len(words) else v["end"]
            s0 = min(max(units[m]["s0"], lo_), hi_ - 0.02)
            v["start"], v["end"], v["aligned"], v["reordered"] = round(s0, 3), round(max(min(units[m]["e0"], hi_), s0 + 0.02), 3), False, True; bad += 1
        print(f"  {spk}: {bad} out-of-order words put back at transcript times")
        # CTC marks word ends short (a word ended 0.27 s early, so its rest looked like an 'um').
        # Grow each aligned word to the end of its sound burst on this mic, never past the next word's start.
        thr = cfg["mic_threshold_db"][spk]
        k = len(wav) // 160; db = 20 * np.log10(np.sqrt((wav[:k * 160].reshape(k, 160) ** 2).mean(1)) + 1e-6)
        for n_, w in enumerate(words):
            if not w["aligned"]: continue
            if w["end"] - w["start"] > 1.5: w["end"] = w["start"] + 0.1; w["long_fixed"] = True   # re-time from the burst below
            nxt = next((x["start"] for x in words[n_ + 1:] if x["start"] > w["start"]), w["end"] + 1.0)
            j = int(w["end"] * 100); lim = min(int(nxt * 100), j + 60)
            while j < lim and j < len(db) and (db[j] > thr or db[j + 1:j + 3].max(initial=-150) > thr): j += 1
            w["end"] = round(max(w["end"], j / 100), 3)
            i = int(w["start"] * 100); prev_end = words[n_ - 1]["end"] if n_ else 0.0
            while i > int(prev_end * 100) and w["start"] - i / 100 < 0.08 and db[i - 1] > thr: i -= 1
            w["start"] = round(min(w["start"], i / 100), 3)
        # Fillers: the aligner barely models "uh"/"um" (scores ~0.01), so place them from the sound instead:
        # the word before runs to the end of its burst; the filler is the next separate burst; the next word starts after it.
        loud = db > thr
        def burst_at(t):          # (start, end) of the sound burst containing t, bridging gaps < 40 ms
            i = int(t * 100)
            if i >= len(loud) or not loud[max(0, i - 2):i + 3].any(): return None
            while i < len(loud) - 1 and not loud[i]: i += 1
            lo = i
            while lo > 0 and (loud[lo - 1] or loud[max(0, lo - 4):lo].any()): lo -= 1
            hi = i
            while hi < len(loud) - 1 and (loud[hi] or loud[hi + 1:hi + 4].any()): hi += 1
            return lo / 100, hi / 100
        for n_, w in enumerate(words):
            if not is_filler(w["word"]): continue
            p_ = words[n_ - 1] if n_ else None; nx = words[n_ + 1] if n_ + 1 < len(words) else None
            lo_t = p_["start"] if p_ else w["start"] - 1.0; hi_t = (nx["start"] if nx else w["end"] + 1.0) + 0.1
            pb = burst_at(p_["start"] + 0.02) if p_ else None
            t = (pb[1] + 0.01) if pb else lo_t
            fb = None
            while t < hi_t:
                b_ = burst_at(t)
                if b_ and b_[0] >= t - 0.02 and b_[1] - b_[0] >= 0.08: fb = b_; break
                t += 0.01
            if fb and fb[1] > hi_t + 0.3:
                # the filler runs into the next word without a silent gap: split at the deepest dip before that word
                i0, i1 = int((fb[0] + 0.12) * 100), int(min(fb[1], hi_t) * 100)
                if i1 > i0 + 2:
                    m = i0 + int(np.argmin(db[i0:i1]))
                    if min(db[max(0, m - 15):m].max(), db[m + 1:m + 16].max()) - db[m] >= 15: fb = (fb[0], m / 100)
                    else: fb = None
                else: fb = None
            if fb and fb[1] <= hi_t + 0.3:
                if p_ and pb and pb[1] <= fb[0]: p_["end"] = round(pb[1], 3)
                w["start"], w["end"] = round(fb[0], 3), round(fb[1], 3); w["placed"] = "burst"
                if nx and nx["start"] < w["end"]: nx["start"] = round(w["end"], 3); nx["end"] = max(nx["end"], round(w["end"] + 0.05, 3))
            else:
                w["aligned"] = False; w["placed"] = "none"     # no separate sound for it -> no cut will be proposed
        # one segment per ~sentence gap so downstream code that groups by segment still works
        segs = []; cur = []
        for w in words:
            if cur and w["start"] - cur[-1]["end"] > 1.5: segs.append(cur); cur = []
            cur.append(w)
        if cur: segs.append(cur)
        outj = {"source": tr.get("source"), "model": f"{tr.get('model')} + MMS_FA alignment", "language": "en",
                "segments": [{"start": s[0]["start"], "end": s[-1]["end"], "text": " ".join(w["word"].strip() for w in s), "words": s} for s in segs]}
        name = f"{spk}_tr_aligned.json"; json.dump(outj, open(os.path.join(W, name), "w", encoding="utf-8"), ensure_ascii=False)
        out_names[spk] = name
        mv = np.array(moved)
        print(f"{spk}: {len(words)} words, aligned {len(moved)}, fallback {fallback}; start moved median {np.median(mv):.3f}s, p90 {np.percentile(mv, 90):.3f}s, >0.3s: {(mv > 0.3).sum()}")
    cfg["transcripts_unaligned"] = dict(src)
    cfg["transcripts"] = dict(out_names); cfg["transcripts_verbatim"] = dict(out_names)
    tqconfig.save(cfg, cfg_path)
    print("config now uses", out_names)

if __name__ == "__main__": main()
