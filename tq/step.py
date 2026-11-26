# Prints a Premiere step script with __WORK__/__SEQ__ filled in, ready to pass to ppro_execute_script.
# Usage: python tq/step.py <episode> <script>   (script: a path, or a file name in premiere/ or in the show's premiere/)
import json,sys,os
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); import tqconfig
CFG=tqconfig.load(sys.argv[1]); sp=sys.argv[2]
if not os.path.isfile(sp):
    for d in (os.path.join(CFG["_show_dir"],"premiere"),os.path.join(os.path.dirname(HERE),"premiere")):
        if os.path.isfile(os.path.join(d,os.path.basename(sp))): sp=os.path.join(d,os.path.basename(sp)); break
src=open(sp,encoding="utf-8").read()
print(src.replace("__WORK__",CFG["work"].replace("\\","/")).replace("__SEQ__",CFG["premiere"]["sequence"]))
