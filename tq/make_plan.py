# Turns edl.json into plan.json (every Premiere clip placement, layout, markers). Usage: python tq/make_plan.py <episode>
# Track layout, template items, intro/outro nests and transforms come from the show profile (show.json "premiere").
# plan.json["spec"] carries all of it resolved, so the premiere-mcp batch tools need nothing else.
import json, os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import tqconfig
CFG=tqconfig.load(sys.argv[1]); os.chdir(CFG["work"])
SHOW=tqconfig.show(CFG); PR=SHOW["premiere"]; TRK=PR["tracks"]
S=tqconfig.speakers(CFG); REF=S[0]
FPS=SHOW.get("fps",30)
e=json.load(open("edl.json"))
OFF=CFG["sync"]["offsets_frames"]
CO=CFG["cold_open_frames"]
C=sum(b-a for a,b in CO)
INTRO_LEN=PR.get("intro_len",0)                    # cold-open end -> show start (frames)
SHOW0=C+INTRO_LEN
clips=[]   # dict(track, kind, item, src_in(frames), dur, start(frames), disabled)
bg=[]
def vid(start,a,b,layout):
    d=b-a
    for s in S: clips.append(dict(t=TRK["solo"][s],item=s+"V",i=a+OFF[s+"V"],d=d,s=start,dis=layout!=s))
    for s in S: clips.append(dict(t=TRK["split"][s],item=s+"V",i=a+OFF[s+"V"],d=d,s=start,dis=layout!="both",fx="split_"+s))
# cold open (split; audio of the speakers listed in show.json premiere.cold_open_audio, default all)
pos=0
for a,b in CO:
    vid(pos,a,b,"both")
    for s in PR.get("cold_open_audio",S): clips.append(dict(t=TRK["audio"][s],item=s+"A",i=a+(OFF[s+"A"] if s!=REF else 0),d=b-a,s=pos,dis=False))
    pos+=b-a
# show
segmap=[]; pos=SHOW0
for a,b in e["segments"]:
    segmap.append([a,b,pos]); pos+=b-a
SE=pos
def tl(f):   # reference-audio frame -> timeline frame
    for a,b,p in segmap:
        if a<=f<b: return p+(f-a)
        if f<a: return p
    return SE
# outro overlay window: D = overlay appears (speakers slide down), U = speakers slide back up
OW=PR.get("outro_window") or {}
D=SE-OW.get("overlay_from_end",0); U=SE-OW.get("slide_back_from_end",0)
def la_of(tf):
    for a,b,p in segmap:
        if p<=tf<p+(b-a): return a+(tf-p)
vpieces=[]
for a,b,lay in e["video"]:
    cuts=[a]
    for T_ in ((D,U) if OW else ()):
        x=la_of(T_)
        if x is not None and a<x<b: cuts.append(x)
    cuts.append(b)
    for x,y in zip(cuts,cuts[1:]): vpieces.append([x,y,lay])
# no layout slivers: a layout run shorter than 1 s takes the previous piece's layout
for k in range(1,len(vpieces)):
    j=k; tot=0
    while j<len(vpieces) and vpieces[j][2]==vpieces[k][2]: tot+=vpieces[j][1]-vpieces[j][0]; j+=1
    if vpieces[k][2]!=vpieces[k-1][2] and tot<FPS:
        for q in range(k,j): vpieces[q][2]=vpieces[k-1][2]
for a,b,lay in vpieces:
    vid(tl(a),a,b,lay)
for s in S:
    key=s+"A"
    for a,b in e["audio"][s]:
        clips.append(dict(t=TRK["audio"][s],item=key,i=a+OFF[key],d=b-a,s=tl(a),dis=False))
# backgrounds per layout run
runs=[[0,C,"both"]]
for a,b,lay in vpieces:
    s=tl(a); d=b-a
    if runs[-1][2]==lay and runs[-1][1]==s: runs[-1][1]=s+d
    else: runs.append([s,s+d,lay])
BG=PR.get("background"); SBG=PR.get("solo_background",{})
for s,t,lay in runs:
    if BG: bg.append(dict(t=BG["track"],item=BG["key"],i=s if BG.get("in")=="timeline" else BG.get("in",0),d=t-s,s=s,dis=False))
    for sp in S:
        if sp in SBG: bg.append(dict(t=SBG[sp]["track"],item=SBG[sp]["key"],i=0,d=t-s,s=s,dis=lay!=sp))
clips=bg+clips
# layout switches (timeline frames) -> layout transition centered, all layout tracks
switches=[r[0] for p_,r in zip(runs,runs[1:]) if r[0]==p_[1] and r[2]!=p_[2]]
# markers
MC=PR.get("marker_colors",{})
def mk(la_sec): return tl(int(round(la_sec*FPS)))
chapters=[tuple(x) for x in CFG["chapters"]]
shorts=[tuple(x) for x in CFG["shorts"]]
markers=[dict(name=n,start=mk(t)/FPS,dur=0,color=MC.get("chapter",0)) for t,n in chapters]
if markers: markers[0]["start"]=SHOW0/FPS
markers+=[dict(name=n,start=mk(a)/FPS,dur=(mk(b)-mk(a))/FPS,color=MC.get("short",1)) for a,b,n in shorts]
markers.append(dict(name=CFG.get("cold_open_label",PR.get("cold_open_label","COLD OPEN")),start=0,dur=C/FPS,color=MC.get("cold_open",1)))
# intro / outro nests: n.in..n.out inside the nest sequence; n.anchor (a frame inside the nest) lands at
#   intro: cold-open end + offset      outro: D (the outro overlay start) + offset
def nest(n,at):
    if not n: return None
    return dict(s=at+n.get("offset",0)-(n["anchor"]-n["in"]),i=n["in"],d=n["out"]-n["in"])
NI,NO=PR.get("intro"),PR.get("outro")
intro_nest=nest(NI,C); outro_nest=nest(NO,D)
if intro_nest and intro_nest["s"]<0: raise SystemExit(f"cold open too short for the intro nest ({C} frames)")
# per-cut audio fades from the review panel: the edit point is where the kept audio resumes (reference frame of the cut end)
audio_fades=[dict(t=round(tl(int(round(b*FPS)))/FPS,4),frames=n,src=[a,b]) for a,b,n in e.get("fades",[])]
# ---- spec: everything the Premiere side needs, resolved for this episode ----
EPR=CFG.get("premiere",{})
names={}
for s in S:
    for k,leg in (("V","video"),("A","audio")):
        names[s+k]=(EPR.get("items") or {}).get(s+k) or EPR.get(f"{s}_{leg}")
labels={}; temp={}
for s in S:
    lb=tqconfig.speaker(CFG,s).get("label")
    if lb is not None: labels[s+"V"]=lb; labels[s+"A"]=lb
for it in ([BG] if BG else [])+[SBG[s] for s in S if s in SBG]+([PR["divider"]] if PR.get("divider") else []):
    names[it["key"]]=it["name"]
    if it.get("label") is not None: temp[it["key"]]=it["label"]
layout_tracks=([BG["track"]] if BG else [])+[SBG[s]["track"] for s in S if s in SBG]+[TRK["solo"][s] for s in S]+[TRK["split"][s] for s in S]
spec=dict(
  fps=FPS, speakers=S, names=names, labels=labels, temp_labels=temp,
  layout_tracks=layout_tracks, audio_tracks=[TRK["audio"][s] for s in S],
  clear_tracks=layout_tracks+[TRK["audio"][s] for s in S], keep_markers=PR.get("keep_markers",[]),
  strip_audio_of=[BG["key"]] if BG and BG.get("strip_audio") else [],
  divider=dict(track=PR["divider"]["track"],item=PR["divider"]["key"]) if PR.get("divider") else None,
  layout_transition=PR.get("layout_transition"),
  speaker_tracks=[TRK["solo"][s] for s in S]+[TRK["split"][s] for s in S],
  split_transforms={TRK["split"][s]:PR["split_transforms"][s] for s in S if s in PR.get("split_transforms",{})},
  outro_slide=PR.get("outro_slide"),
  nests=[dict(key=k,name=n["nest"],track=n["track"],audio_track=n.get("audio_track"),level_db=n.get("level_db")) for k,n in (("intro_nest",NI),("outro_nest",NO)) if n],
  remove_before_nests=dict(names=PR.get("nest_shell_items",[]),tracks=PR.get("nest_shell_tracks",[])),
  park_sequence=PR.get("park_sequence"), template_sequences=PR.get("template_sequences",[]),
)
plan=dict(audio_fades=audio_fades,fps=FPS,C=C,D=D,U=U,switches=switches,SHOW0=SHOW0,SE=SE,
          intro_shift=(C-NI["ref_cold_open_frames"])/FPS if NI and "ref_cold_open_frames" in NI else None,
          outro_shift=(SE/FPS)-NO["ref_show_end_s"] if NO and "ref_show_end_s" in NO else None,
          intro_nest=intro_nest,outro_nest=outro_nest,clips=clips,markers=markers,spec=spec)
json.dump(plan,open("plan.json","w"))
from collections import Counter
print("switches",[x/FPS for x in switches],"D",D/FPS,"U",U/FPS); print("C",C/FPS,"show0",SHOW0/FPS,"SE",SE/FPS,"runs",len(runs),Counter(c["t"] for c in clips), "intro_shift",plan["intro_shift"],"outro_shift",plan["outro_shift"])
for m in markers: print(round(m["start"],2), m["color"], m["name"])
