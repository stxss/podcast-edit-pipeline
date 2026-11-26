# Acoustic filler detector: voiced, steady-pitch sounds the transcript has no word for (um / uh / erm / mm).
import json, wave, os, re
import numpy as np
W = os.path.dirname(os.path.abspath(__file__))
SR = 16000; HOP = 160; WIN = 640

def audio(name):
    w = wave.open(os.path.join(W, name + ".wav"))
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768

def features(x):
    n = (len(x) - WIN) // HOP
    outs = [_feat(x, i, min(n, i+20000)) for i in range(0, n, 20000)]
    return tuple(np.concatenate([o[k] for o in outs]) for k in range(3))

def _feat(x, i0, i1):
    n = i1 - i0
    idx = np.arange(WIN)[None, :] + HOP * np.arange(i0, i1)[:, None]
    fr = x[idx] * np.hanning(WIN)
    db = 20*np.log10(np.sqrt((fr**2).mean(1)) + 1e-6)
    F = np.fft.rfft(fr, 2*WIN)
    P = np.abs(F)**2
    ac = np.fft.irfft(P)[:, :WIN]
    ac = ac / (ac[:, :1] + 1e-9)
    lo, hi = SR//400, SR//70           # 70-400 Hz pitch range
    seg = ac[:, lo:hi]
    k = seg.argmax(1)
    strength = seg[np.arange(n), k]
    f0 = SR / (k + lo)
    return db, strength, f0
