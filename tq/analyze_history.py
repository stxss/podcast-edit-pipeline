# Stats on an editor's own cuts in past episodes (pacing, filler/repeat handling) -> hist_stats.json.
# Usage: python tq/analyze_history.py <dir> --episodes EP02:<ref key>:<other key> ...
#   <dir> holds allseq_tracks.json (timeline dumps of the finished sequences) and hist/<EP>_<ref|other>.json
#   (transcripts of the raw files, named by the first two speaker ids of the show: <EP>_<id>.json); hist_stats.json is
#   written there. Each --episodes item is EPxx:<part of the reference speaker's raw file name>:<same for the second
#   speaker>, used to tell the two speakers' clips apart. Two-speaker shows only.
import json,re,statistics as st,collections,sys,os,argparse
ap=argparse.ArgumentParser(); ap.add_argument("dir",nargs="?",default="."); ap.add_argument("--episodes",nargs="+",required=True)
ap.add_argument("--speakers",nargs=2,default=["a","b"],help="ids used in the hist/ transcript file names")
args=ap.parse_args(); os.chdir(args.dir)
EPS=[tuple(e.split(":")) for e in args.episodes]
D=json.load(open("allseq_tracks.json"))
FILL={"um","uh","umm","uhm","uhh","er","erm","ah","hmm","hm","mm","eh"}
def norm(t): return re.sub(r"[^a-z']","",t.lower())
def union(iv):
    iv=sorted(iv); out=[]
    for a,b in iv:
        if out and a<=out[-1][1]+1e-3: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return out
def at(items,t):
    for s,e,i,*_ in items:
        if s<=t<e: return i+(t-s)
def pct(x,ps=(10,25,50,75,90)):
    x=sorted(x); return [round(x[min(len(x)-1,int(p/100*len(x)))],2) for p in ps] if x else []
res={}
for ep,refkey,othkey in EPS:
    S=D[ep]
    LV=[x for k,a in S["V"].items() for x in a if refkey in x[4]]
    TV=[x for k,a in S["V"].items() for x in a if othkey in x[4]]
    LA=[x for k,a in S["A"].items() for x in a if refkey in x[4]]
    TA=[x for k,a in S["A"].items() for x in a if othkey in x[4]]
    # offsets
    def med(f):
        v=[f(x) for x in LV]; v=[y for y in v if y is not None]; return st.median(v) if v else None
    o_la=med(lambda x: (lambda a: a-(x[2]+0.01) if a is not None else None)(at(LA,x[0]+0.01)))   # refA - refV
    o_ta=med(lambda x: (lambda a,b: a-b if a is not None and b is not None else None)(at(TA,x[0]+0.01),at(LV,x[0]+0.01)))  # otherA - refV
    if o_ta is None or o_la is None: print(ep,"offsets missing",o_la,o_ta); continue
    d=o_ta-o_la   # otherA - refA
    kept=union([[x[2]+o_la, x[2]+(x[1]-x[0])+o_la] for x in LV])   # in reference-audio time
    # words in LA
    W=[]
    for spk,off in ((args.speakers[0],0.0),(args.speakers[1],-d)):
        j=json.load(open(f"hist/{ep}_{spk}.json"))
        for g in j["segments"]:
            for w in g["words"]: W.append((w["start"]+off,w["end"]+off,norm(w["word"]),spk,w["word"].strip()))
    W.sort()
    lo=min(a for a,b in kept if b-a>5); hi=max(b for a,b in kept)
    # show span: from first kept interval after the cold open... use kept intervals overall
    def iskept(t):
        for a,b in kept:
            if a<=t<b: return True
        return False
    Ws=[w for w in W if lo<=w[0]<=hi]
    k=[iskept((w[0]+w[1])/2) for w in Ws]
    fill=[kk for w,kk in zip(Ws,k) if w[2] in FILL]
    rep=[]
    for i in range(len(Ws)-1):
        if Ws[i][2] and Ws[i][2]==Ws[i+1][2] and Ws[i][3]==Ws[i+1][3] and Ws[i+1][0]-Ws[i][1]<0.6: rep.append(k[i] and k[i+1])
    # cut stats
    ints=[iv for iv in kept if iv[0]>=lo-1]
    tails=[];heads=[];gaps=[];outp=[];longcuts=[]
    for (a1,b1),(a2,b2) in zip(ints,ints[1:]):
        gap=a2-b1
        if gap<0.04 or gap>600: continue
        before=[w for w in Ws if w[1]<=b1+0.15 and iskept((w[0]+w[1])/2) and w[0]<b1]
        after=[w for w in Ws if w[0]>=a2-0.15 and iskept((w[0]+w[1])/2) and w[1]>a2]
        if not before or not after: continue
        t=b1-before[-1][1]; h=after[0][0]-a2
        if -0.3<t<1.5 and -0.3<h<1.5: tails.append(t); heads.append(h); outp.append(t+h)
        gaps.append(gap)
        if gap>4:
            txt=" ".join(w[4] for w in Ws if b1<=w[0]<a2)
            longcuts.append((round(b1,1),round(gap,1),txt[:160]))
    nat=[Ws[i+1][0]-Ws[i][1] for i in range(len(Ws)-1) if k[i] and k[i+1] and Ws[i+1][0]-Ws[i][1]>0.25]
    keptdur=sum(min(b,hi)-max(a,lo) for a,b in kept if b>lo and a<hi)
    r=dict(offsets=dict(refA_minus_refV=round(o_la,3),otherA_minus_refA=round(d,3)),
      span_min=round((hi-lo)/60,1), kept_min=round(keptdur/60,1), removed_pct=round(100*(1-keptdur/(hi-lo)),1),
      cuts=len(gaps), cuts_per_min=round(len(gaps)/(keptdur/60),1),
      words=len(Ws), words_removed_pct=round(100*(1-sum(k)/len(k)),1),
      fillers=len(fill), fillers_kept_pct=round(100*sum(fill)/max(1,len(fill)),1),
      immediate_repeats=len(rep), repeats_fully_kept_pct=round(100*sum(rep)/max(1,len(rep)),1),
      tail_after_last_word_p10_25_50_75_90=pct(tails), head_before_first_word=pct(heads), pause_at_cut=pct(outp),
      natural_kept_pauses_over_250ms=pct(nat), removed_gap_len=pct(gaps), n_long_removals=len(longcuts))
    res[ep]=r; res[ep+"_long"]=longcuts
json.dump(res,open("hist_stats.json","w"),indent=1)
for ep,_,_ in EPS:
    if ep in res: print(ep, json.dumps(res[ep]))
