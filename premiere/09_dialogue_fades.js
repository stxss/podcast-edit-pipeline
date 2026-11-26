// Fallback for audio transitions: equal-power volume-keyframe fades at every dialogue clip edge.
// Needs __WORK__/fades.json from tq/fades.py and __WORK__/plan.json (spec.audio_tracks). Don't combine with ppro_audio_transitions.
const fs=require('fs'); const fades=JSON.parse(fs.readFileSync("__WORK__/fades.json",{encoding:'utf-8'}));
const plan=JSON.parse(fs.readFileSync("__WORK__/plan.json",{encoding:'utf-8'})); const FPS=plan.fps||30;
const want={}; for(const f of fades) want[f.t+":"+f.s]=f;
const p=await ppro.Project.getActiveProject(); const seq=(await p.getSequences()).find(s=>s.name==="__SEQ__");
const tk=s=>ppro.TickTime.createWithTicks(String(Math.round(s*254016000000))); const rows=[];
for(const t of plan.spec.audio_tracks){ const ai=parseInt(t.slice(1))-1; for(const ti of (await seq.getAudioTrack(ai)).getTrackItems(1,false)){ const s=Math.round(secs(await ti.getStartTime())*FPS); const f=want[t+":"+s]; if(!f) continue;
  const L=(await ti.getComponentChain()).getComponentAtIndex(0).getParam(1); if(await L.isTimeVarying()) continue;
  rows.push({L,inS:secs(await ti.getInPoint()),dur:secs(await ti.getEndTime())-secs(await ti.getStartTime()),v:(await L.getStartValue()).value.value,fi:f.fi,fo:f.fo}); } }
p.lockedAccess(()=>{ p.executeTransaction(ca=>{ for(const r of rows){ const k=(s,v)=>{ const x=r.L.createKeyframe(v); x.position=tk(s); return x; }; ca.addAction(r.L.createSetTimeVaryingAction(true));
  for(let j=0;j<5;j++){ const x=j/4; ca.addAction(r.L.createAddKeyframeAction(k(r.inS+x*r.fi,r.v*Math.sin(Math.PI/2*x)))); }
  for(let j=0;j<5;j++){ const x=j/4; ca.addAction(r.L.createAddKeyframeAction(k(r.inS+r.dur-r.fo+x*r.fo,r.v*Math.cos(Math.PI/2*x)))); } } },"dialogue fades"); });
return {clips:rows.length};
