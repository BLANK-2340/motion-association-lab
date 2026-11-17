'use strict';
function validate(data) {
  if (data.schema !== 1 || data.scope !== 'saved CPU results, not live inference' || data.cases.length !== 9 || data.sweep.length !== 24) throw Error('Unexpected evidence format');
  for (const c of data.cases) {
    if (c.frames.length !== 40 || c.metrics.length !== 2) throw Error('Incomplete sequence');
    for (const m of c.metrics) {
      if (![m.truth,m.matches,m.misses,m.false_positives,m.identity_switches].every(Number.isInteger) || m.truth !== 80 || m.matches+m.misses !== 80 || Object.values(m).some(v=>typeof v === 'number' && (!Number.isFinite(v)||v<0))) throw Error('Invalid diagnostics');
    }
    for (const f of c.frames) {
      for (const p of [...f.foreground,...f.detections,...f.truth.map(t=>t[1]),...f.tracks.flat().map(t=>t[1])]) {
        if (p.length !== 2 || !p.every(Number.isFinite) || p[0]<0 || p[0]>=100 || p[1]<0 || p[1]>=64) throw Error('Invalid image coordinates');
      }
      if (f.tracks.length !== 2) throw Error('Missing variant');
    }
  }
  return data;
}
if (typeof module !== 'undefined') module.exports = {validate};
if (typeof document !== 'undefined') {
  const $ = id => document.getElementById(id);
  let data, timer;
  function stop(){clearInterval(timer);timer=null;$('play').textContent='Play';}
  function draw(){
    const c=data.cases[+$('case').value], k=+$('frame').value, f=c.frames[k];
    $('number').textContent=`${k} / 39`;
    $('status').textContent=`Frame ${k}: ${f.detections.length} components; hidden truth IDs: ${f.hidden.join(', ')||'none'}.`;
    ['last','velocity'].forEach((id,v)=>{
      const ctx=$(id).getContext('2d');ctx.resetTransform();ctx.clearRect(0,0,800,512);ctx.scale(8,8);
      ctx.fillStyle='#64747a';for(const [x,y] of f.foreground) ctx.fillRect(x-.5,y-.5,1,1);
      if($('truth').checked) for(const [i,[x,y]] of f.truth){ctx.strokeStyle='#ffffff';ctx.lineWidth=.18;ctx.setLineDash(f.hidden.includes(i)?[.7,.6]:[]);ctx.strokeRect(x-2.5,y-2.5,5,5);ctx.setLineDash([]);ctx.fillStyle='#ffffff';ctx.font='2.1px monospace';ctx.fillText(`GT${i}`,x-3,y-4);}
      for(const [i,[x,y]] of f.tracks[v]){
        ctx.strokeStyle=['#a8b0ff','#7ae6cc','#ffbf72','#ff849a'][i%4];ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=.3;ctx.beginPath();let started=false;
        for(let t=Math.max(0,k-7);t<=k;t++){const p=c.frames[t].tracks[v].find(q=>q[0]===i);if(p){if(!started)ctx.moveTo(...p[1]);else ctx.lineTo(...p[1]);started=true;}}
        ctx.stroke();ctx.beginPath();ctx.arc(x,y,.8,0,Math.PI*2);ctx.fill();ctx.font='2.4px monospace';ctx.fillText(`T${i}`,x+2,y+4);
      }
      const m=c.metrics[v];$('metric'+v).textContent=`Whole sequence: ${m.matches}/${m.truth} matches · ${m.misses} misses · ${m.identity_switches} ID switches · ${m.false_positives} false positives`;
    });
  }
  fetch('evidence.json').then(r=>{if(!r.ok)throw Error('Evidence request failed');return r.json();}).then(validate).then(value=>{
    data=value;$('case').replaceChildren(...data.cases.map((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.name;return o;}));$('case').value='2';
    ['case','frame','play','reset'].forEach(id=>$(id).disabled=false);
    $('case').addEventListener('change',()=>{stop();$('frame').value=0;draw();});$('frame').addEventListener('input',()=>{stop();draw();});$('truth').addEventListener('change',draw);
    $('reset').addEventListener('click',()=>{stop();$('frame').value=0;draw();});
    $('play').addEventListener('click',()=>{if(timer){stop();return;}if(+$('frame').value===39)$('frame').value=0;$('play').textContent='Pause';timer=setInterval(()=>{if(+$('frame').value===39){stop();return;}$('frame').value=+$('frame').value+1;draw();},180);});
    const sums=[0,1].map(v=>data.sweep.reduce((s,r)=>s+r.metrics[v].identity_switches,0));
    $('study').textContent=`Across 24 fixed controlled cells: ${sums[0]} total identity switches for last-position versus ${sums[1]} for velocity. These are sums over designed cases, not a population estimate.`;draw();
  }).catch(error=>{$('status').textContent='Evidence unavailable: '+error.message;});
}
