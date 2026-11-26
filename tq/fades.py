# Builds fades.json (per dialogue clip: fade-in/out lengths). 60 ms where the mic is audible at the edge, 30 ms elsewhere.
# Usage: python tq/fades.py <episode>
import json,os,sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import tqconfig; CFG=tqconfig.load(sys.argv[1]); os.chdir(CFG["work"])
from vadlib import load_db, shift
plan=json.load(open("plan.json")); thr=CFG["mic_threshold_db"]; FPS=plan.get("fps",30)
TRK=tqconfig.show(CFG)["premiere"]["tracks"]["audio"]
out=[]
for s in tqconfig.speakers(CFG):
    t,key=TRK[s],s+"A"; sh=tqconfig.shift_s(CFG,s); db=shift(load_db(s),int(round(sh*100))); th=thr[s]
    for c in plan["clips"]:
        if c["t"]!=t: continue
        a=c["i"]/FPS+sh; b=a+c["d"]/FPS
        def hot(x,side):
            i=int(round(x*100)); w=db[i:i+3] if side=="in" else db[max(0,i-3):i]
            return bool(w.size and w.max()>th+6)
        cap=c["d"]/FPS/3
        out.append(dict(t=t,s=c["s"],fi=min(0.06 if hot(a,"in") else 0.03,cap),fo=min(0.06 if hot(b,"out") else 0.03,cap)))
json.dump(out,open("fades.json","w")); print(len(out),"clips")
