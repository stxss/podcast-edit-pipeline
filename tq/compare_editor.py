# Compares the pipeline's edl.json with the editor's own timeline for the same episode (after their pass).
# Usage: python tq/compare_editor.py <episode> [--tracks <timeline dump>] [--transcripts config|large|verbatim]
# The timeline dump and how to read it come from cfg["editor_truth"] (see tq/truth.py). Writes compare_rows_<tag>.json and audio_muted_<spk>_<tag>.json in the work dir.
import json,re,collections,sys,os,argparse
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import tqconfig, truth as TR
from vadlib import load_db, shift
ap=argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--tracks",default=None,help="timeline dump in the work dir (default cfg editor_truth.tracks)")
ap.add_argument("--transcripts",default="config",choices=("config","small","large","verbatim"),help="config/small = cfg transcripts, large = <spk>_tr_large.json, verbatim = cfg transcripts_verbatim")
args=ap.parse_args()
CFG=tqconfig.load(args.config); os.chdir(CFG["work"])
TRN="small" if args.transcripts=="config" else args.transcripts   # output file tag, as before
def norm(t): return re.sub(r"[^a-z']","",t.lower())
from truth import minus
TT=TR.load(CFG, args.tracks)   # editor's/ours kept + audio, in reference-audio time
his=TT["his_kept"]; mine=TT["mine_kept"]; my_audio=TT["my_audio"]
SP=tqconfig.speakers(CFG)
W=[]
for spk,off in ((s,tqconfig.shift_s(CFG,s)) for s in SP):
    if args.transcripts in ("config","small"): f=CFG["transcripts"][spk]
    elif args.transcripts=="large": f=f"{spk}_tr_large.json"
    else: f=(CFG.get("transcripts_verbatim") or CFG["transcripts"])[spk]
    j=json.load(open(f,encoding="utf-8"))
    for g in j["segments"]:
        for w in g["words"]: W.append((w["start"]+off,w["end"]+off,spk,w["word"].strip()))
W.sort()
DB=[shift(load_db(s),int(round(tqconfig.shift_s(CFG,s)*100))) for s in SP]
def words_in(a,b,spk=None): return [w for w in W if w[0]<b and w[1]>a and (spk is None or w[2]==spk) and (min(w[1],b)-max(w[0],a))>0.5*(w[1]-w[0])]
def loud(a,b):
    i,j=int(a*100),int(b*100)
    return round(float(max(d[i:j].max(initial=-150) for d in DB)),1)
extra=minus(mine,his)       # we kept, the editor removed
restored=minus(his,mine)    # the editor kept, we removed
FILL={"um","uh","umm","uhm","uhh","er","erm","ah","hmm","hm","mm","mmhmm","uhhuh"}
def classify(a,b):
    ws=words_in(a,b); toks=[norm(w[3]) for w in ws]
    if not ws: return "no words (cough/breath/noise/silence)" if loud(a,b)>-50 else "silence"
    if all(t in FILL for t in toks): return "filler um/uh"
    if all(t in FILL|{"like"} for t in toks): return "filler 'like'"
    if all(t in {"yeah","right","yes","mhm","okay","sure","exactly","yep","totally","true","nice"} for t in toks): return "backchannel"
    return "content"
rows=[]
for a,b in extra:
    if b-a<0.05: continue
    rows.append(dict(kind="his_cut",a=round(a,2),b=round(b,2),dur=round(b-a,2),cls=classify(a,b),text=" ".join(f"{w[2][0].upper()}:{w[3]}" for w in words_in(a,b))[:140]))
for a,b in restored:
    if b-a<0.05: continue
    rows.append(dict(kind="his_restore",a=round(a,2),b=round(b,2),dur=round(b-a,2),cls=classify(a,b),text=" ".join(f"{w[2][0].upper()}:{w[3]}" for w in words_in(a,b))[:140]))
json.dump(rows,open(f"compare_rows_{TRN}.json","w"),indent=0)
c=collections.defaultdict(lambda:[0,0.0])
for r in rows: c[(r["kind"],r["cls"])][0]+=1; c[(r["kind"],r["cls"])][1]+=r["dur"]
for k,v in sorted(c.items()): print(k, v[0], round(v[1],1),"s")
print("mine kept",round(sum(b-a for a,b in mine)/60,2),"min | editor kept",round(sum(b-a for a,b in his)/60,2))
# audio-only removals (picture kept, mic muted) per speaker
for s in SP:
    rem=TT["muted"][s]   # picture kept, editor muted the mic, ours had it on (> 0.1 s)
    cl=collections.Counter(classify(a,b) for a,b in rem)
    print("audio muted by you,",s,len(rem),dict(cl))
    json.dump([[round(a,2),round(b,2)," ".join(w[3] for w in words_in(a,b,s))] for a,b in rem],open(f"audio_muted_{s}_{TRN}.json","w"))
