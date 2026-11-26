# Pacing stats of an edl.json (cuts per minute, pause kept around cuts). Usage: python tq/stats_edl.py <episode> [edl file]
import json,re,sys,os,bisect
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import tqconfig
CFG=tqconfig.load(sys.argv[1]); os.chdir(CFG["work"])
def norm(t): return re.sub(r"[^a-z']","",t.lower())
def pct(x,ps=(10,25,50,75,90)):
    x=sorted(x); return [round(x[min(len(x)-1,int(p/100*len(x)))],2) for p in ps]
FPS=tqconfig.show(CFG).get("fps",30)
e=json.load(open(sys.argv[2] if len(sys.argv)>2 else "edl.json")); kept=[[a/FPS,b/FPS] for a,b in e["segments"]]
W=[]
for spk in tqconfig.speakers(CFG):
    off=tqconfig.shift_s(CFG,spk); j=json.load(open(CFG["transcripts"][spk],encoding="utf-8"))
    for g in j["segments"]:
        for w in g["words"]: W.append((w["start"]+off,w["end"]+off))
W.sort(); lo,hi=kept[0][0],kept[-1][1]
starts=[a for a,b in kept]
def iskept(t):
    i=bisect.bisect_right(starts,t)-1; return i>=0 and kept[i][0]<=t<kept[i][1]
Ws=[w for w in W if lo<=w[0]<=hi]
tails=[];heads=[];outp=[];gaps=[]
for (a1,b1),(a2,b2) in zip(kept,kept[1:]):
    gap=a2-b1
    if gap<0.04 or gap>600: continue
    before=[w for w in Ws if w[1]<=b1+0.15 and iskept((w[0]+w[1])/2) and w[0]<b1]
    after=[w for w in Ws if w[0]>=a2-0.15 and iskept((w[0]+w[1])/2) and w[1]>a2]
    if not before or not after: continue
    t=b1-before[-1][1]; h=after[0][0]-a2
    if -0.3<t<1.5 and -0.3<h<1.5: tails.append(t); heads.append(h); outp.append(t+h)
    gaps.append(gap)
kd=sum(b-a for a,b in kept)
print(json.dumps(dict(out_min=round(kd/60,2),removed_pct=round(100*(1-kd/(hi-lo)),1),cuts=len(gaps),cuts_per_min=round(len(gaps)/(kd/60),1),tail=pct(tails),head=pct(heads),pause_at_cut=pct(outp),removed_gap_len=pct(gaps))))
