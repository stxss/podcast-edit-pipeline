# Whisper transcription that runs on any machine: mlx-whisper on Apple Silicon Macs (GPU through MLX),
# faster-whisper everywhere else (NVIDIA GPU if there is one, else CPU). Same output format as the other engines.
#   python tq/asr_whisper.py <EP> [--engine auto|mlx|faster] [--model turbo|large-v3] [--out-dir bake] [--name NAME]
# Writes <work>/<id>_tr_whisper.json per speaker (or <out-dir>/<name>_<id>.json) - then set "transcripts_unaligned" to
# them in the episode config and run tq/run_post.py (align.py fixes the word times from the sound).
#
# Settings = the best plain-Whisper result of a transcription bake-off on a real episode: large-v3-turbo, a prompt
# full of fillers (keeps most um/uh), word timestamps, no conditioning on earlier text, and voice detection so
# Whisper never sees long silences (it hallucinates there). faster-whisper's own voice detector isn't available in
# mlx-whisper, so by default both engines use this script's own: each mic records one speaker up close, so its level
# against the speaker's mic_threshold_db is a reliable speech detector; everything else is replaced by silence
# (timestamps stay unchanged).
# Install: Mac  -> pip install mlx-whisper        Windows/Linux -> pip install faster-whisper (+ CUDA torch for GPU)
import argparse, json, os, sys, time, wave
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tqconfig
from vadlib import runs, close

PROMPT = "Um, uh, so, like, you know, I- I mean, uh, we- we were, um, yeah."
MODELS = {"mlx": {"turbo": "mlx-community/whisper-large-v3-turbo", "large-v3": "mlx-community/whisper-large-v3-mlx"},
          "faster": {"turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "large-v3": "large-v3"}}
SR = 16000


def pick_engine(want):
    if want != "auto": return want
    import platform, importlib.util
    if sys.platform == "darwin" and platform.machine() == "arm64" and importlib.util.find_spec("mlx_whisper"): return "mlx"
    if importlib.util.find_spec("faster_whisper"): return "faster"
    if importlib.util.find_spec("mlx_whisper"): return "mlx"
    sys.exit("no Whisper engine: pip install mlx-whisper (Apple Silicon) or faster-whisper")


def load(path):
    w = wave.open(path); assert w.getframerate() == SR and w.getnchannels() == 1, f"{path}: needs 16 kHz mono"
    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def speech_only(x, thr_db):
    """Keep the stretches where this mic is above its speaker threshold (gaps < 1 s bridged, 0.3 s padding); zero the rest."""
    k = len(x) // 160
    db = 20 * np.log10(np.sqrt((x[:k * 160].reshape(k, 160) ** 2).mean(1)) + 1e-6)
    if thr_db is None: thr_db = max(np.percentile(db, 10) + 15, np.percentile(db, 95) - 30)
    v = close(db > thr_db, 100)
    keep = np.zeros(len(x), bool); kept = 0.0
    for a, b in runs(v):
        if (b - a) < 15: continue                       # < 0.15 s: a click, not a word
        a0, b0 = max(0, (a - 30) * 160), min(len(x), (b + 30) * 160)
        keep[a0:b0] = True
    y = np.where(keep, x, 0.0).astype(np.float32)
    return y, keep.mean()


def transcribe(engine, model_id, audio, vad):
    t0 = time.time()
    if engine == "mlx":
        import mlx_whisper
        r = mlx_whisper.transcribe(audio, path_or_hf_repo=model_id, language="en", word_timestamps=True,
                                   initial_prompt=PROMPT, condition_on_previous_text=False, verbose=None)
        segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip(),
                 "words": [{"start": w["start"], "end": w["end"], "word": w["word"].strip(), "prob": w.get("probability", 0)} for w in s.get("words", [])]}
                for s in r["segments"]]
    else:
        if os.name == "nt":
            try:   # ctranslate2 needs cuBLAS/cuDNN; torch ships them
                import torch; d = os.path.join(os.path.dirname(torch.__file__), "lib"); os.add_dll_directory(d); os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
            except Exception: pass
        from faster_whisper import WhisperModel
        try:
            import ctranslate2; dev = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception: dev = "cpu"
        m = WhisperModel(model_id, device=dev, compute_type="float16" if dev == "cuda" else "int8")
        it, _ = m.transcribe(audio, language="en", word_timestamps=True, vad_filter=(vad == "builtin"),
                             initial_prompt=PROMPT, condition_on_previous_text=False)
        segs = [{"start": s.start, "end": s.end, "text": s.text.strip(),
                 "words": [{"start": w.start, "end": w.end, "word": w.word.strip(), "prob": w.probability} for w in (s.words or [])]} for s in it]
    for s in segs:
        s["start"], s["end"] = round(s["start"], 3), round(s["end"], 3)
        for w in s["words"]: w["start"], w["end"], w["prob"] = round(w["start"], 3), round(w["end"], 3), round(float(w["prob"]), 3)
    return segs, time.time() - t0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episode"); ap.add_argument("--engine", default="auto", choices=("auto", "mlx", "faster"))
    ap.add_argument("--model", default="turbo", choices=("turbo", "large-v3")); ap.add_argument("--vad", default="own", choices=("own", "builtin", "none"))
    ap.add_argument("--out-dir", default=None); ap.add_argument("--name", default=None); ap.add_argument("--speakers", nargs="*")
    a = ap.parse_args()
    cfg = tqconfig.load(a.episode); W = cfg["work"]
    eng = pick_engine(a.engine)
    if eng == "mlx" and a.vad == "builtin": sys.exit("--vad builtin exists only in faster-whisper")
    mid = MODELS[eng][a.model]
    thr = cfg.get("mic_threshold_db") or {}
    print(f"engine {eng}, model {mid}, voice detection {a.vad}", flush=True)
    for spk in a.speakers or tqconfig.speakers(cfg):
        x = load(os.path.join(W, spk + ".wav"))
        if a.vad == "own":
            x, frac = speech_only(x, thr.get(spk)); print(f"{spk}: speech {frac * 100:.0f}% of the file", flush=True)
        segs, dt = transcribe(eng, mid, x, a.vad)
        if a.out_dir:
            d = os.path.join(W, a.out_dir); os.makedirs(d, exist_ok=True); dst = os.path.join(d, f"{a.name or 'whisper-' + a.model}_{spk}.json")
        else:
            dst = os.path.join(W, f"{spk}_tr_whisper.json")
        json.dump({"source": spk + ".wav", "model": mid, "engine": eng, "vad": a.vad, "prompt": PROMPT, "segments": segs, "seconds": round(dt, 1)},
                  open(dst, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"{spk}: {sum(len(s['words']) for s in segs)} words in {dt:.0f} s -> {dst}", flush=True)


if __name__ == "__main__":
    main()
