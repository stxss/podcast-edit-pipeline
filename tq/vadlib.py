import wave, numpy as np, json, re, os
HOP=0.01
def load_db(n):
    w=wave.open(n+".wav"); x=np.frombuffer(w.readframes(w.getnframes()),np.int16).astype(np.float32)
    h=160; k=len(x)//h; r=np.sqrt((x[:k*h].reshape(k,h)**2).mean(1))+1e-3; return 20*np.log10(r/32768)
def runs(v):
    d=np.diff(np.concatenate([[0],v.astype(np.int8),[0]])); return list(zip(np.where(d==1)[0],np.where(d==-1)[0]))
def close(v,n):
    v=v.copy(); r=runs(v)
    for (a,b),(c,_) in zip(r,r[1:]):
        if c-b<n: v[b:c]=True
    return v
def open_(v,n):
    v=v.copy()
    for a,b in runs(v):
        if b-a<n: v[a:b]=False
    return v
def vad(db):
    thr=max(np.percentile(db,10)+15,np.percentile(db,95)-30)
    v=db>thr; v=close(v,15); v=open_(v,8); return v,thr
def words(n):
    d=json.load(open(n+'_tr.json'))
    return [{"w":w["word"],"s":w["start"],"e":w["end"],"g":gi} for gi,s in enumerate(d["segments"]) for w in s["words"]]
def shift(arr, n, fill=-150.0):
    """A mic array -> reference time base. n = that mic's shift in samples/frames: >0 pads the start, <0 drops samples."""
    n = int(n)
    if n >= 0: return np.concatenate([np.full(n, fill, dtype=arr.dtype), arr])
    return arr[-n:]
