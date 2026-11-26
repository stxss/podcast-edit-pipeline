# Transcription bake-off, step 1: transcribe both mics with each candidate model, same settings as the pipeline
# (faster-whisper, word timestamps, VAD on, no conditioning on previous text, filler prompt for the plain Whisper models).
# Run with the GPU Python:
#   <gpu_python from local.json> tq/asr_bakeoff.py episodes/EPxx.json [model ...]
# Writes <work>/bake/<name>_<spk>.json (same format as transcribe_media). Existing files are skipped.
import json, os, sys, time, wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tqconfig

P = "Um, uh, so, like, you know, I- I mean, uh, we- we were, um, yeah."
MODELS = {   # name -> (faster-whisper model id, initial prompt, VAD filter)
    "large-v3": ("large-v3", P, True),
    "large-v3-turbo": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", P, True),
    "large-v3-noprompt": ("large-v3", None, True),
    # VAD swallows short fillers (and maybe CrisperWhisper's dropped stretches): same models without it
    "crisper-novad": ("nyrahealth/faster_CrisperWhisper", None, False),
    "large-v3-turbo-novad": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", P, False),
}

def dll_dirs():
    if os.name != "nt": return
    import torch   # ctranslate2 needs cuBLAS/cuDNN; torch ships them
    d = os.path.join(os.path.dirname(torch.__file__), "lib")
    os.add_dll_directory(d); os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

def main():
    cfg = tqconfig.load(sys.argv[1]); W = cfg["work"]; out_dir = os.path.join(W, "bake"); os.makedirs(out_dir, exist_ok=True)
    names = sys.argv[2:] or list(MODELS)
    dll_dirs()
    from faster_whisper import WhisperModel
    for name in names:
        mid, prompt, vad = MODELS[name]; model = None
        for spk in tqconfig.speakers(cfg):
            dst = os.path.join(out_dir, f"{name}_{spk}.json")
            if os.path.exists(dst): print("skip", dst); continue
            if model is None: model = WhisperModel(mid, device="cuda", compute_type="float16")
            w = wave.open(os.path.join(W, spk + ".wav")); audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
            t0 = time.time()
            segs, info = model.transcribe(audio, language="en", word_timestamps=True, vad_filter=vad,
                                          initial_prompt=prompt, condition_on_previous_text=False)
            out = [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
                    "words": [{"start": round(x.start, 3), "end": round(x.end, 3), "word": x.word.strip(), "prob": round(x.probability, 3)} for x in (s.words or [])]}
                   for s in segs]
            json.dump({"source": spk + ".wav", "model": mid, "prompt": prompt, "segments": out, "seconds": round(time.time() - t0, 1)},
                      open(dst, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"{name} {spk}: {sum(len(s['words']) for s in out)} words in {time.time() - t0:.0f} s", flush=True)
        del model

if __name__ == "__main__": main()
