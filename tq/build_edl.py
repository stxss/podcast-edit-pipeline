# Cut list builder. Usage: python tq/build_edl.py <episode> [--no-review]
# Time base = the first speaker's audio clock (reference audio). Per-episode values come from the episode config,
# speakers and tuning from the show profile (show.json "cut").
import json, re, os, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tqconfig; CFG=tqconfig.load(sys.argv[1]); os.chdir(CFG["work"])
from vadlib import load_db, runs, close, open_, shift
import fillers as F
F.W = os.getcwd()
HOP=0.01; FPS=30
# pacing (defaults measured on an editor's own cuts): pauses < MERGE_GAP are left alone; when a pause is cut,
# keep ~TAIL after the last word and ~HEAD before the next one
_c=lambda k,d: tqconfig.setting(CFG,"cut",k,d)
MERGE_GAP=_c("merge_gap",0.6); TAIL=_c("tail",0.20); HEAD=_c("head",0.14)
SHOW_START, SHOW_END = CFG["show"]["start"], CFG["show"]["end"]
MANUAL_CUTS=[tuple(x) for x in CFG["manual_cuts"]]
THR=CFG["mic_threshold_db"]
S=tqconfig.speakers(CFG)
TR=CFG.get("transcripts",{s:s+"_tr.json" for s in S})
SH={s:tqconfig.shift_s(CFG,s) for s in S}
def any_other(V_,s,a,b): return any(V_[o][a:b].any() for o in tqconfig.others(CFG,s))
def idx(t): return int(round(t/HOP))
def norm(t): return re.sub(r"[^a-z']","",t.lower())
db={s:shift(load_db(s),int(round(SH[s]/HOP))) for s in S}
N=min(len(db[s]) for s in S)
V={}
for s in S:
    v=db[s][:N]>THR[s]; v=close(v,15); v=open_(v,8); V[s]=v
def words(s):
    d=json.load(open(TR[s])); off=SH[s]
    out=[]
    for gi,g in enumerate(d["segments"]):
        for w in g["words"]:
            a,b=w["start"]+off,w["end"]+off
            if a<0: continue
            ia,ib=idx(a),max(idx(b),idx(a)+1)
            if ib>=N: continue
            if V[s][ia:ib].mean()<0.3: continue        # bleed / hallucination: mic wasn't active
            out.append({"w":w["word"],"s":a,"e":b,"g":gi})
    return out
W={s:words(s) for s in S}
FILLERS={"um","uh","umm","uhm","uhh","er","erm","ah","hmm","hm","mm","eh"}
KEEP_REPEATS={"really","very","no","yeah","yes","blah","so","bye","okay","ok","ha","go","more","much","far","long","many","again","over","now","right","well","huge","super","too","bad","good","great","true","exactly","sure","totally","hundred","thousand","on","down"}
KEEP_REPEATS|=set(_c("keep_repeats_extra",[]))      # show-specific words that are said twice on purpose
# number words never stutter: "twenty twenty four", "one one two" (Nemotron writes numbers as words)
KEEP_REPEATS|={"zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety","oh","million","billion"}
removed={s:np.zeros(N,bool) for s in S}
# ---- review panel decisions (tq/review.py -> review.json) ----
REVIEW=json.load(open("review.json")) if os.path.exists("review.json") and "--no-review" not in sys.argv else {"decisions":{}}
def _rid(kind,spk,a,b): return f"{kind}|{spk}|{a:.2f}|{b:.2f}"
# a2/b2 = edges the editor set by hand in the panel; user_cuts = cuts drawn in the panel. Both are PINNED: exact frame edges, never moved/undone.
_ed=lambda d: d.get("a2") is not None
UNDO={k for k,d in REVIEW["decisions"].items() if d["group"]=="edl" and (d["action"]=="keep" or _ed(d))}
ADD_CUTS=[(d["a"],d["b"],d["spk"],d["kind"]) for d in REVIEW["decisions"].values() if d["group"]=="det" and d["action"]=="cut" and not _ed(d)]
ADD_MUTES=[((d["a2"],d["b2"]) if _ed(d) else (d["a"],d["b"]))+(d["spk"],) for d in REVIEW["decisions"].values() if d["group"]=="mute" and d["action"]=="mute"]
ADD_MUTES+=[(u["a"],u["b"],u["spk"]) for u in REVIEW.get("user_cuts",[]) if u.get("decision")=="mute"]   # mutes drawn in the panel
PINNED=[(d["a2"],d["b2"],d["spk"],"edited:"+d["kind"]) for d in REVIEW["decisions"].values() if d["group"] in ("edl","det") and d["action"]=="cut" and _ed(d)]
PINNED+=[(u["a"],u["b"],u["spk"],"user") for u in REVIEW.get("user_cuts",[]) if u.get("decision","cut")=="cut"]
# audio fade (Constant Power, frames) set on a cut in the panel; default is the 2f pass on every dialogue edit
FADES=[[d.get("a2",d["a"]) if _ed(d) else d["a"], d["b2"] if _ed(d) else d["b"], int(d["fade"])] for d in REVIEW["decisions"].values()
       if d.get("fade") is not None and d["action"]!="keep"]
FADES+=[[u["a"],u["b"],int(u["fade"])] for u in REVIEW.get("user_cuts",[]) if u.get("fade") is not None and u.get("decision","cut")=="cut"]
def undone(kind,spk,a,b): return _rid("edl:"+kind.split(":")[0],spk,a,b) in UNDO
stats={"fillers_cut":0,"fillers_kept_overlap":0,"repeats_cut":0,"restarts_cut":0,"acoustic_um_cut":0,"manual_cuts":len(MANUAL_CUTS),
       "review_undone":0,"review_added_cuts":len(ADD_CUTS),"review_mutes":len(ADD_MUTES),"review_pinned":len(PINNED)}
log=[]
# editorial cut edges were set on transcript times; with aligned words, an edge inside a word moves off it:
# start edge -> after the (kept) word it lands in, end edge -> before the (kept) word it lands in
_allw=sorted([(w["s"],w["e"]) for s_ in S for w in W[s_] if w["e"]-w["s"]>=0.05])
def _snap_manual(a,b):
    for s0,e0 in _allw:
        if s0+0.02<a<e0-0.02: a=e0+0.01
    for s0,e0 in _allw:
        if s0+0.02<b<e0-0.02: b=min(b,s0-0.01)
    return a,b
_snapped=[]
for a,b,why in MANUAL_CUTS:
    a2,b2=_snap_manual(a,b)
    if b2-a2<0.1: a2,b2=a,b
    _snapped.append((round(a2,2),round(b2,2),why,a,b))
stats["manual_edges_moved"]=sum(1 for x in _snapped if (x[0],x[1])!=(round(x[3],2),round(x[4],2)))
for a,b,why,a0,b0 in _snapped:
    if undone("manual","both",a0,b0) or undone("manual","both",a,b): stats["review_undone"]+=1; stats["manual_cuts"]-=1; continue
    for s in S: removed[s][idx(a):idx(b)]=True
    log.append((a,b,"both","manual:"+why))
_Eq=np.max([db[s][:N]-THR[s] for s in S],axis=0)
def _snap(t,lo,hi):   # quietest 10 ms in [t+lo, t+hi] (transcript word edges are loose)
    i,j=max(0,idx(t+lo)),min(N,idx(t+hi)+1); return (i+int(np.argmin(_Eq[i:j])))*HOP if j>i else t
for a,b,spk,kind in ADD_CUTS:      # approved detector candidates: a picture cut, like a manual cut
    a2=_snap(a,-0.02,0.12); b2=_snap(b,-0.12,0.02)
    if b2-a2>=0.08: a,b=a2,b2
    for s in S: removed[s][idx(a):idx(b)]=True
    log.append((round(a,2),round(b,2),spk,"review:"+kind))
for a,b,spk,kind in PINNED:        # hand-set edges: no snapping
    for s in S: removed[s][idx(a):idx(b)]=True
    log.append((round(a,3),round(b,3),spk,"review:"+kind))
for s in S:
    ws=W[s]
    for i,w in enumerate(ws):
        if w["s"]<SHOW_START or w["e"]>SHOW_END: continue
        t=norm(w["w"]); rng=None
        if t in FILLERS:
            a,b=w["s"],w["e"]
            if i+1<len(ws): b=min(b+0.05,ws[i+1]["s"])
            if i>0: a=max(a-0.03,ws[i-1]["e"])
            rng=(a,b,"filler")
        else:
            for k in (1,2,3,4,5):
                if i+2*k<=len(ws):
                    A=[norm(x["w"]) for x in ws[i:i+k]]; B=[norm(x["w"]) for x in ws[i+k:i+2*k]]
                    rh=(k==1 and A[0] in KEEP_REPEATS)
                    if all(a in KEEP_REPEATS for a in A) and ((i+3*k<=len(ws) and [norm(x["w"]) for x in ws[i+2*k:i+3*k]]==A) or (i-k>=0 and [norm(x["w"]) for x in ws[i-k:i]]==A)): rh=True
                    if A==B and all(A) and not rh and ws[i+k]["s"]-ws[i+k-1]["e"]<1.0 and ws[i+2*k-1]["e"]-ws[i]["s"]<3.0:
                        rng=(ws[i]["s"],ws[i+k]["s"],"repeat:"+" ".join(A)); break
            if rng is None and w["w"].strip().endswith("-") and i+1<len(ws) and ws[i+1]["s"]-w["s"]<2.5:
                rng=(w["s"],ws[i+1]["s"],"restart:"+w["w"].strip())
        if rng is None: continue
        a,b,kind=rng
        if kind!="filler": a,b=a-0.02,b-0.02
        if b-a<0.08 or b-a>3.0: continue
        ia,ib=idx(a),idx(b)
        if any_other(V,s,max(0,ia-10),ib+10):
            if kind=="filler": stats["fillers_kept_overlap"]+=1
            continue
        if undone(kind,s,round(a,2),round(b,2)): stats["review_undone"]+=1; continue
        removed[s][ia:ib]=True
        stats["fillers_cut" if kind=="filler" else ("restarts_cut" if kind.startswith("restart") else "repeats_cut")]+=1
        log.append((round(a,2),round(b,2),s,kind))
# acoustic um/uh that whisper skipped
for s in S:
    x=F.audio(s); dbf,st,f0=F.features(x)
    if s!=S[0]:
        sh=int(round(SH[s]/HOP)); dbf=shift(dbf,sh); st=shift(st,sh,0.0); f0=shift(f0,sh,100.0)
    n=min(len(dbf),N)
    cov=np.zeros(n,bool)
    _d=json.load(open(TR[s])); _o=SH[s]
    for g in _d["segments"]:
        for w in g["words"]:
            if norm(w["word"]) in FILLERS: continue
            cov[max(0,idx(w["start"]+_o)):max(0,idx(w["end"]+_o)+1)]=True   # any transcribed word (even if timing is loose) is not an um
    voiced=(st[:n]>0.55)&V[s][:n]
    for a,b in runs(V[s][:n]&~cov):
        dur=(b-a)*HOP
        if not (0.20<=dur<=1.2): continue
        if a*HOP<SHOW_START or b*HOP>SHOW_END: continue
        if any_other(V,s,max(0,a-10),b+10): continue
        if voiced[a:b].mean()<0.7: continue
        p=f0[a:b][voiced[a:b]]
        if len(p)<10 or np.std(12*np.log2(p/np.median(p)))>1.5: continue
        if removed[s][a:b].any(): continue
        if undone("um(acoustic)",s,round(a*HOP,2),round(b*HOP,2)): stats["review_undone"]+=1; continue
        removed[s][a-2:b+3]=True; stats["acoustic_um_cut"]+=1; log.append((round(a*HOP,2),round(b*HOP,2),s,"um(acoustic)"))
speech={s:open_(V[s]&~removed[s],12) for s in S}
anyv=np.logical_or.reduce([speech[s] for s in S])
lo,hi=idx(SHOW_START),idx(SHOW_END)
anyv[:lo]=False; anyv[hi:]=False
def who(a,b): return max(S,key=lambda s: speech[s][a:b].sum())
keep=[list(r) for r in runs(anyv)]
for j in range(len(keep)-1):
    b=keep[j][1]; c=keep[j+1][0]; gap=(c-b)*HOP
    turn=who(*keep[j])!=who(*keep[j+1])
    fin=any(r[b:c].any() for r in removed.values())
    if gap<=MERGE_GAP and not fin: keep[j][1]=c; continue
    tail=min(0.10,gap/2); lead=min(0.06,gap/2)   # VAD edges; word-based padding is enforced in the refine step
    keep[j][1]=b+int(round(tail/HOP)); keep[j+1][0]=c-int(round(lead/HOP))
merged=[]
for a,b in keep:
    if merged and a<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],b)
    else: merged.append([a,b])
merged[0][0]=lo; merged[-1][1]=hi
final=[r for r in merged if (r[1]-r[0])*HOP>=0.25 or any(speech[s][r[0]:r[1]].sum()>15 for s in S)]
rem=np.logical_or.reduce([removed[s] for s in S])
hard=[]
for a,b in final:
    for x,y in runs(~rem[a:b]):
        if y-x>=3: hard.append([a+x,a+y])
segs=[]   # in LA frames (1/30 s)
for a,b in hard:
    fa=int(np.floor(a*HOP*FPS+1e-6)); fb=int(np.ceil(b*HOP*FPS-1e-6))   # keep partial frames (cuts never grow by rounding)
    if fb-fa<2: continue
    if segs and fa<=segs[-1][1]: segs[-1][1]=max(segs[-1][1],fb)
    else: segs.append([fa,fb])
# ---- refine cut points: never inside a kept word, land on the quietest frame nearby ----
E=np.max([db[s][:N]-THR[s] for s in S],axis=0)   # >0 = someone audibly talking
allw=sorted([(w["s"],w["e"]) for s_ in S for w in W[s_]])
def fquiet(t,lo,hi):
    best=None
    for f in range(int(np.floor((t+lo)*FPS)),int(np.ceil((t+hi)*FPS))+1):
        c=idx(f/FPS); val=E[max(0,c-2):c+3].max()
        if best is None or val<best[0]-1e-9 or (abs(val-best[0])<1e-9 and abs(f/FPS-t)<abs(best[1]/FPS-t)): best=(val,f)
    return best[1]
moved=0; refined=[]
remall=np.logical_or.reduce([removed[s] for s in S])
def free_fwd(t,maxs):   # how far forward from t (s) we can go without entering removed material
    i=idx(t); j=i
    while j<N and j-i<maxs/HOP and not remall[j]: j+=1
    return (j-i)*HOP
def free_bwd(t,maxs):
    i=idx(t); j=i
    while j>0 and i-j<maxs/HOP and not remall[j-1]: j-=1
    return (i-j)*HOP
for k,(fa,fb) in enumerate(segs):
    A,B=fa/FPS,fb/FPS
    fwd=free_fwd(B,TAIL+0.02); bwd=free_bwd(A,HEAD+0.02)
    kw=[(a,b) for a,b in allw if a<B and b>A and (b-a)<1.2]
    tB,tA=B,A
    if kw:
        tB=max(B,min(max(b for _,b in kw)+TAIL,B+fwd))
        tA=min(A,max(min(a for a,_ in kw)-HEAD,A-bwd))
    nb=fquiet(tB,-0.04,max(0.0,min(0.04,B+fwd-tB)))
    na=fquiet(tA,-max(0.0,min(0.04,tA-(A-bwd))),0.04)
    prevB=refined[-1][1] if refined else 0
    na=max(na,prevB)
    if k+1<len(segs): nb=min(nb,segs[k+1][0])
    if nb-na<2: na,nb=fa,fb
    if (na,nb)!=(fa,fb): moved+=1
    refined.append([na,nb])
segs=[]
for a,b in refined:
    if segs and a<=segs[-1][1]: segs[-1][1]=max(segs[-1][1],b)
    else: segs.append([a,b])
stats["boundaries_moved"]=moved
# a cut that still lands on running speech (no quiet valley) is worse than the filler it removes -> undo it
def hot(t,side):
    c=idx(t); w=E[max(0,c-3):c] if side=="e" else E[c:c+3]
    return w.size and w.max()>6
manual=[(a,b) for a,b,_,_,_ in _snapped]+[(a,b) for a,b,_,_ in ADD_CUTS]+[(a,b) for a,b,_,_ in PINNED]   # approved cuts are never auto-undone
undone=0; out=[segs[0]]
for a,b in segs[1:]:
    pa,pb=out[-1]; gap=(a-pb)/FPS
    is_manual=any(ma-0.3<=pb/FPS<=mb+0.3 for ma,mb in manual)
    if gap<1.5 and not is_manual and (hot(pb/FPS,"e") or hot(a/FPS,"s")):
        out[-1][1]=b; undone+=1
    else: out.append([a,b])
segs=out
stats["hot_cuts_undone"]=undone
# pinned cuts: exact frames. Kept frames on either side that the refine step trimmed (<= 0.2 s, no other cut there) come back.
if PINNED:
    nf=int(np.ceil(N*HOP*FPS))+2; keepf=np.zeros(nf,bool)
    for a,b in segs: keepf[a:b]=True
    fr_removed=lambda f: remall[idx(f/FPS):idx((f+1)/FPS)].any()
    for a,b,_,_ in PINNED:
        pa=int(round(a*FPS)); pb=int(round(b*FPS))
        keepf[pa:pb]=False

        if pa-1>=0 and not keepf[pa-1]:
            g=pa-1
            while g>=0 and not keepf[g] and pa-g<=6: g-=1
            if g>=0 and keepf[g]:
                if not any(fr_removed(x) and not (pa<=x<pb) for x in range(g+1,pa) if not any(int(round(p*FPS))<=x<int(round(q*FPS)) for p,q,_,_ in PINNED)):
                    keepf[g+1:pa]=True
        if pb<nf and not keepf[pb]:
            g=pb
            while g<nf and not keepf[g] and g-pb<6: g+=1
            if g<nf and keepf[g]:
                if not any(fr_removed(x) for x in range(pb,g) if not any(int(round(p*FPS))<=x<int(round(q*FPS)) for p,q,_,_ in PINNED)):
                    keepf[pb:g]=True
    for a,b,_,_ in PINNED: keepf[int(round(a*FPS)):int(round(b*FPS))]=False
    segs=[[int(x),int(y)] for x,y in runs(keepf)]
# ---- layout: split ("both") by default, solo for long single-speaker stretches ----
SOLO_MIN=_c("solo_min_s",110.0); SOLO_ENTER=_c("solo_enter_s",4.0); SOLO_ABSORB=_c("solo_absorb_s",2.5)
SP=np.array([speech[s] for s in S]); NACT=SP.sum(0)
lab=np.zeros(N,np.int8); cur=0
for i in range(lo,hi):
    if NACT[i]==1: cur=int(np.argmax(SP[:,i]))
    lab[i]=cur
rl=[]; i=lo
while i<hi:
    j=i
    while j<hi and lab[j]==lab[i]: j+=1
    rl.append([i,j,int(lab[i])]); i=j
changed=True
while changed:   # absorb short interjections (backchannels) into the surrounding floor-holder
    changed=False; out=[]
    for r in rl:
        if out and (r[2]==out[-1][2] or (r[1]-r[0])*HOP<SOLO_ABSORB):
            out[-1][1]=r[1]; changed=changed or r[2]!=out[-1][2]
        else: out.append(r)
    rl=out
# measure kept (output) duration of each floor run
segmask=np.zeros(N,bool)
for fa,fb in segs: segmask[int(fa/FPS/HOP):int(fb/FPS/HOP)]=True
solo=[]
for a,b,c in rl:
    kept=segmask[a:b].sum()*HOP
    if kept>=SOLO_MIN:
        # enter solo 4s after the floor change, leave at the end of the run
        solo.append((a*HOP+SOLO_ENTER,b*HOP,S[c],round(kept,1)))
def layout_at(t):
    for a,b,c,_ in solo:
        if a<=t<b: return c
    return "both"
pieces=[]   # [fa,fb,layout] in LA frames
for fa,fb in segs:
    a=fa
    bounds=sorted({int(round(x*FPS)) for s_ in solo for x in s_[:2] if fa<x*FPS<fb})
    for bnd in bounds+[fb]:
        if bnd-a>0:
            lay=layout_at((a+bnd)/2/FPS)
            if pieces and pieces[-1][1]==a and pieces[-1][2]==lay: pieces[-1][1]=bnd
            else: pieces.append([a,bnd,lay])
        a=bnd
# ---- audio gating: each mic only while its owner talks (drops bleed from the other mics + backchannels) ----
def gate(s):
    """Mic on-regions. Never starts/stops inside one of this speaker's words, edges sit on the quietest nearby 10 ms."""
    v=speech[s].copy()
    oth=np.logical_or.reduce([speech[o] for o in tqconfig.others(CFG,s)])
    isl=runs(v)
    for k,(a,b) in enumerate(isl):
        # drop ISOLATED backchannels sitting on the other speaker ("yeah", "right")
        prev_gap=(a-isl[k-1][1]) if k>0 else 999
        next_gap=(isl[k+1][0]-b) if k+1<len(isl) else 999
        if (b-a)*HOP<1.0 and oth[max(0,a-30):b+30].mean()>0.6 and prev_gap>60 and next_gap>60:
            v[a:b]=False
    v=close(v,80)                       # bridge pauses < 0.8 s
    # never cut through a transcribed word of this speaker
    for w in W[s]:
        ia,ib=idx(w["s"]),idx(w["e"])
        if ib<=ia or ib>=N: continue
        seg=v[ia:ib]
        if seg.any() and not seg.all(): v[ia:ib]=True
    d=np.zeros_like(v)
    e=db[s][:N]
    for a,b in runs(v):
        a2=max(0,a-25); b2=min(N,b+35)          # 0.25 s pre-roll, 0.35 s tail
        # slide edges outward to the quietest point within 0.12 s
        lo_=max(0,a2-12); a2=lo_+int(np.argmin(e[lo_:a2+1])) if a2>lo_ else a2
        hi_=min(N,b2+12); b2=b2+int(np.argmin(e[b2:hi_])) if hi_>b2 else b2
        d[a2:b2]=True
    return d
G={s:gate(s) for s in S}
for a,b,spk in ADD_MUTES: G[spk][idx(a):idx(b)]=False
audio={s:[] for s in S}
for fa,fb in segs:
    a0=int(fa/FPS/HOP); b0=int(fb/FPS/HOP)
    for s in S:
        for x,y in runs(G[s][a0:b0]):
            xa=int(round((a0+x)*HOP*FPS)); ya=int(round((a0+y)*HOP*FPS))
            xa=max(xa,fa); ya=min(ya,fb)
            if ya-xa>=3: audio[s].append([xa,ya])
out_len=sum(b-a for a,b in segs)/FPS
stats.update({"segments":len(segs),"source_min":round((SHOW_END-SHOW_START)/60,2),"output_min":round(out_len/60,2),
  "removed_pct":round(100*(1-out_len/(SHOW_END-SHOW_START)),1),"pieces":len(pieces),
  "solo":solo,"audio_clips":{s:len(audio[s]) for s in S}})
stats["review_fades"]=len(FADES)
json.dump({"segments":segs,"video":pieces,"audio":audio,"stats":stats,"cuts":log,"fades":FADES},open("edl.json","w"))
print(json.dumps(stats))
