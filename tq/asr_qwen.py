# Bake-off candidate: Qwen3-ASR-1.7B (transformers) + Qwen3-ForcedAligner-0.6B for word times.
# The audio is cut into ~25 s chunks at the quietest moment of the speaker's own mic, transcribed in batches, then each
# chunk's text is force-aligned on that chunk. Output: <work>/bake/qwen3-asr_<spk>.json (same format as the others).
#   <gpu_python from local.json> tq/asr_qwen.py episodes/EPxx.json
import json, os, sys, time, wave
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tqconfig

SR = 16000; TARGET = 25.0; BATCH = 8
CONTEXT = "Conversational podcast. Transcribe verbatim: keep um, uh, repeated words and false starts exactly as spoken."

def chunks_at_pauses(x, thr_db):
    k = len(x) // 160; db = 20 * np.log10(np.sqrt((x[:k * 160].reshape(k, 160) ** 2).mean(1)) + 1e-6)
    out = []; a = 0
    while a < k:
        if k - a <= int(TARGET * 100 * 1.3): out.append((a, k)); break
        lo, hi = a + int(TARGET * 100 * 0.7), a + int(TARGET * 100 * 1.3)
        sm = np.convolve(db[lo:hi], np.ones(30) / 30, mode="same")   # quietest 0.3 s
        b = lo + int(np.argmin(sm)); out.append((a, b)); a = b
    return [(a * 160, b * 160) for a, b in out]

def main():
    cfg = tqconfig.load(sys.argv[1]); W = cfg["work"]; B = os.path.join(W, "bake"); os.makedirs(B, exist_ok=True)
    from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoModelForTokenClassification
    asr_id, al_id = "Qwen/Qwen3-ASR-1.7B-hf", "Qwen/Qwen3-ForcedAligner-0.6B-hf"
    proc = AutoProcessor.from_pretrained(asr_id); model = AutoModelForMultimodalLM.from_pretrained(asr_id, dtype=torch.bfloat16, device_map="cuda")
    aproc = AutoProcessor.from_pretrained(al_id); amodel = AutoModelForTokenClassification.from_pretrained(al_id, dtype=torch.bfloat16, device_map="cuda")
    for spk in tqconfig.speakers(cfg):
        dst = os.path.join(B, f"qwen3-asr_{spk}.json")
        if os.path.exists(dst): print("skip", dst); continue
        t0 = time.time()
        w = wave.open(os.path.join(W, spk + ".wav")); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        ch = chunks_at_pauses(x, cfg["mic_threshold_db"][spk]); segs = []
        for i in range(0, len(ch), BATCH):
            part = ch[i:i + BATCH]; auds = [x[a:b] for a, b in part]
            inputs = proc.apply_transcription_request(audio=auds, language=["English"] * len(auds), prompt=[CONTEXT] * len(auds)).to(model.device, model.dtype)
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=400, do_sample=False)
            texts = proc.decode(ids[:, inputs["input_ids"].shape[1]:], return_format="transcription_only")
            for (a, b), aud, txt in zip(part, auds, texts):
                txt = (txt or "").strip()
                if not txt: continue
                ai, wl = aproc.prepare_forced_aligner_inputs(audio=aud, transcript=txt, language="English")
                ai = ai.to(amodel.device, amodel.dtype)
                with torch.inference_mode():
                    lo = amodel(**ai).logits
                ts = aproc.decode_forced_alignment(logits=lo, input_ids=ai["input_ids"], word_lists=wl, timestamp_token_id=amodel.config.timestamp_token_id)[0]
                off = a / SR
                words = [{"start": round(off + t["start_time"], 3), "end": round(off + t["end_time"], 3), "word": t["text"]} for t in ts]
                segs.append({"start": round(off, 3), "end": round(b / SR, 3), "text": txt, "words": words})
            print(f"  {spk}: {min(i + BATCH, len(ch))}/{len(ch)} chunks, {time.time() - t0:.0f} s", flush=True)
        json.dump({"source": spk + ".wav", "model": asr_id + " + " + al_id, "prompt": CONTEXT, "segments": segs, "seconds": round(time.time() - t0, 1)},
                  open(dst, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"qwen3-asr {spk}: {sum(len(s['words']) for s in segs)} words in {time.time() - t0:.0f} s", flush=True)

if __name__ == "__main__": main()
