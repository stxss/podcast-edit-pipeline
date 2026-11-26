# Speech-overlap sync check between two mics. Usage (in the work dir): python <repo>/tq/sync_overlap.py <speaker_a> <speaker_b>
import sys,numpy as np
from vadlib import load_db, vad
a,b=sys.argv[1],sys.argv[2]
L,T=load_db(a),load_db(b); N=min(len(L),len(T))-300
vl,_=vad(L[:N]); vt,_=vad(T[:N])
# onsets of speech (turn starts) - compare: count frames both active
res=[]
for off in range(-100,101,2):  # b index = a index + off
    x=vl[200:N-200]; y=vt[200+off:N-200+off]
    res.append((round((x&y).mean()*100,3),off/100))
res.sort(); print(a,b,"least overlap (pct, b minus a s):",res[:6])
