# NVIDIA Nemotron 3.5 Transcribe (verbatim mode, word times from its CTC head) via NeMo-Speech.cpp, all mics in one session
# (the 23 GB model loads once). Converts the output to the pipeline's transcript format.
#   python tq/asr_nemotron.py <episode> [--out-dir bake] [--convert-only]
# Needs: tools/nemo-speech/<release>/bin/nemo-speech(.exe) (github.com/NVIDIA/NeMo-Speech.cpp releases) and the GGUF files
# nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF: ...-NVFP4.gguf (Blackwell, 32 GB) or ...-IQ4_NL.gguf (other GPUs) + ...-Encoder-F16.gguf.
# local.json may set "nemo_speech" (binary), "nemotron_model" (decoder gguf) and "asr_device" (default cuda).
import json, os, sys, glob, shutil, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tqconfig
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
INSTR = ("Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. "
         "Formatting: punctuation and capitalization. Output: transcript only.")

def local(): return tqconfig.local()

def find_bin():
    b = local().get("nemo_speech")
    if b: return b
    c = glob.glob(os.path.join(ROOT, "tools", "nemo-speech", "*", "bin", "nemo-speech*"))
    if not c: raise SystemExit("nemo-speech not found under tools/nemo-speech (see the header)")
    return c[0]

def find_model():
    m = local().get("nemotron_model")
    if m: return m
    snap = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF/snapshots/*/")
    c = glob.glob(snap + "*-NVFP4.gguf") or glob.glob(snap + "*-IQ4_NL.gguf")
    if not c: raise SystemExit("Nemotron GGUF not found in the Hugging Face cache")
    return c[0]

def to_transcript(nemo_json, src):
    d = json.load(open(nemo_json, encoding="utf-8"))
    ws = [{"start": round(w["start"], 3), "end": round(w["end"], 3), "word": w["word"]} for w in d.get("words", [])]
    segs, cur = [], []
    for w in ws:   # segments at pauses > 1 s (downstream code groups by segment)
        if cur and w["start"] - cur[-1]["end"] > 1.0: segs.append(cur); cur = []
        cur.append(w)
    if cur: segs.append(cur)
    return {"source": src, "model": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B (NVFP4, verbatim)", "prompt": INSTR,
            "segments": [{"start": s[0]["start"], "end": s[-1]["end"], "text": " ".join(w["word"] for w in s), "words": s} for s in segs]}

def main():
    cfg = tqconfig.load(sys.argv[1]); W = cfg["work"]
    out_dir = os.path.join(W, sys.argv[sys.argv.index("--out-dir") + 1]) if "--out-dir" in sys.argv else W
    tmp_in, tmp_out = os.path.join(W, "bake", "nemo_in"), os.path.join(W, "bake", "nemo_out")
    if "--convert-only" not in sys.argv:
        os.makedirs(tmp_in, exist_ok=True); os.makedirs(tmp_out, exist_ok=True)
        for spk in tqconfig.speakers(cfg): shutil.copyfile(os.path.join(W, spk + ".wav"), os.path.join(tmp_in, spk + ".wav"))
        t0 = time.time()
        subprocess.run([find_bin(), "transcribe", tmp_in, "--model", find_model(), "--device", local().get("asr_device", "cuda"), "--format", "json",
                        "--word-times", "--instruction", INSTR, "--output-dir", tmp_out, "--force", "--quiet"], check=True)
        print(f"nemotron: all mics in {time.time() - t0:.0f} s")
    for spk in tqconfig.speakers(cfg):
        src = os.path.join(tmp_out, spk + ".json")
        dst = os.path.join(out_dir, f"nemotron_{spk}.json" if out_dir != W else f"{spk}_tr_nemotron.json")
        tr = to_transcript(src, spk + ".wav")
        json.dump(tr, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
        print(spk, sum(len(s["words"]) for s in tr["segments"]), "words ->", dst)

if __name__ == "__main__": main()
