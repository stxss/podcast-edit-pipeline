# Layout runs for a divider graphic placed by hand (fallback for premiere/08_divider.js).
# Usage: python tq/graphic_runs.py <episode> <graphic_start_s> <graphic_end_s>
import json,os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import tqconfig
CFG=tqconfig.load(sys.argv[1]); os.chdir(CFG["work"])
plan=json.load(open("plan.json")); FPS=plan.get("fps",30); g0,g1=round(float(sys.argv[2])*FPS),round(float(sys.argv[3])*FPS)
PR=tqconfig.show(CFG)["premiere"]; SBG=PR.get("solo_background",{}); BG=PR.get("background")
solo={v["track"]:s for s,v in SBG.items()}            # solo background track -> speaker
bgt={BG["track"]} if BG else set()
runs={}
for c in plan["clips"]:
    if c["t"] not in bgt|set(solo): continue
    r=runs.setdefault(c["s"],{"s":c["s"],"e":c["s"]+c["d"],"lay":"both"})
    if c["t"] in solo and not c["dis"]: r["lay"]=solo[c["t"]]
m=[]
for s in sorted(runs):
    r=runs[s]; a=max(r["s"],g0); b=min(r["e"],g1)
    if b<=a: continue
    if m and m[-1][2]==r["lay"] and m[-1][1]==a: m[-1][1]=b
    else: m.append([a,b,r["lay"]])
json.dump({"runs":m,"g0":g0,"g1":g1,"switches":plan["switches"]},open("graphic_runs.json","w")); print(m)
