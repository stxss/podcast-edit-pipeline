# Creates <show>/episodes/<EP>.json for a new episode, plus its work folder.
# Usage: python tq/new_episode.py <EP> <raw file of speaker 1> <raw file of speaker 2> ... [--show <show dir>]
#   Raw files in speaker order of show.json. Defaults for the rest come from show.json "episode_defaults".
#   Paths inside media_root (local.json) are stored relative to it, so the config works on any machine.
import json,sys,os,argparse
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import tqconfig
ap=argparse.ArgumentParser(); ap.add_argument("episode"); ap.add_argument("raw",nargs="+"); ap.add_argument("--show",default=None)
a=ap.parse_args()
sd=a.show or tqconfig.show_dir()
if not sd: sys.exit("set show_dir in local.json or pass --show")
show=tqconfig.load_show(os.path.join(sd,"show.json")); ep=a.episode
S=[s["id"] for s in show["speakers"]]
if len(a.raw)!=len(S): sys.exit(f"need {len(S)} raw files, in this order: {', '.join(S)}")
mr=tqconfig.media_root(show.get("id"))
if not mr: sys.exit("set media_root in local.json first (copy local.example.json)")
def rel(p):
    p=os.path.abspath(os.path.expanduser(p))
    try: r=os.path.relpath(p,os.path.abspath(mr))
    except ValueError: r=".."            # other drive
    return (p if r.startswith("..") else r).replace("\\","/")
names=show.get("premiere",{}).get("item_names",{"V":"{episode} {Name} VIDEO","A":"{episode} {Name} AUDIO"})
c={"episode":ep,"work":show.get("work_pattern","{episode}/_work").format(episode=ep),
   "raw":{s:rel(p) for s,p in zip(S,a.raw)},
   "transcripts":{s:f"{s}_tr_aligned.json" for s in S},
   "transcripts_verbatim":{s:f"{s}_tr_aligned.json" for s in S},
   "transcripts_unaligned":{s:f"{s}_tr_nemotron.json" for s in S},
   "sync":{"shift_s":{s:(0.0 if i==0 else None) for i,s in enumerate(S)},"offsets_frames":None,"note":"fill from the clap step"},
   "show":{"start":None,"end":None},"mic_threshold_db":{s:None for s in S},
   "manual_cuts":[],"cold_open_frames":[],"chapters":[],"shorts":[],
   "premiere":{"sequence":ep,"items":{s+k:names[k].format(episode=ep,Name=x.get("name",s),id=s) for x in show["speakers"] for s in [x["id"]] for k in ("V","A")}}}
for k,v in show.get("episode_defaults",{}).items(): c.setdefault(k,v)
dst=os.path.join(sd,"episodes",f"{ep}.json")
if os.path.exists(dst): sys.exit(f"{dst} exists - not overwriting")
os.makedirs(os.path.dirname(dst),exist_ok=True)
os.makedirs(tqconfig.resolve(c["work"],show.get("id")),exist_ok=True)
json.dump(c,open(dst,"w",encoding="utf-8"),indent=1); print("wrote",dst)
