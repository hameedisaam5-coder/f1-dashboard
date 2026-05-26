import re

with open('dashboard.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Header
html = re.sub(
    r'<div class="header-left">.*?</div>',
    '<div class="header-left">\n    <div class="header-title" id="headerTitle">Replay Dashboard</div>\n    <div class="header-sub">Historical Data Playback</div>\n  </div>',
    html, flags=re.DOTALL
)

html = re.sub(
    r'<div class="header-right".*?</div>\s*</div>',
    '''<div class="header-right" style="gap: 12px; display:flex; align-items:center;">
    <select id="rpYear" style="background:#1a1a24;color:#f3f4f6;border:1px solid #2a2a35;border-radius:4px;padding:4px;font-size:12px;">
      <option value="2025" selected>2025</option><option value="2024">2024</option>
    </select>
    <select id="rpRace" style="background:#1a1a24;color:#f3f4f6;border:1px solid #2a2a35;border-radius:4px;padding:4px;font-size:12px;">
      <option value="Australian Grand Prix" selected>Australian GP</option><option value="Bahrain Grand Prix">Bahrain GP</option>
      <option value="Saudi Arabian Grand Prix">Saudi Arabian GP</option><option value="Japanese Grand Prix">Japanese GP</option>
      <option value="Chinese Grand Prix">Chinese GP</option><option value="Miami Grand Prix">Miami GP</option>
      <option value="Emilia Romagna Grand Prix">Imola GP</option><option value="Monaco Grand Prix">Monaco GP</option>
      <option value="Canadian Grand Prix">Canadian GP</option><option value="Spanish Grand Prix">Spanish GP</option>
      <option value="Austrian Grand Prix">Austrian GP</option><option value="British Grand Prix">British GP</option>
    </select>
    <select id="rpSession" style="background:#1a1a24;color:#f3f4f6;border:1px solid #2a2a35;border-radius:4px;padding:4px;font-size:12px;">
      <option value="R" selected>Race</option><option value="S">Sprint</option>
    </select>
    <button onclick="rpLoad()" style="background:#e10600;color:#fff;border:none;border-radius:4px;padding:5px 12px;font-weight:bold;cursor:pointer;">▶ LOAD</button>
  </div>
</div>''',
    html, flags=re.DOTALL
)

# 2. Track Section
html = re.sub(
    r'<div class="track-section">.*?</div>\s*</div>',
    '''<div class="track-section" style="position:relative;">
  <div id="replayWrap" style="height: 360px; position: relative; overflow: hidden; background: #000; border-radius: 6px; margin: 0 10px;">
    <canvas id="replayCanvas" style="display:block; width:100%; height:100%; cursor:grab;"></canvas>
    <div id="rpLoader" style="position:absolute; inset:0; background:rgba(0,0,0,0.8); z-index:50; display:none; flex-direction:column; align-items:center; justify-content:center;">
      <div style="width:40px; height:40px; border:4px solid transparent; border-top-color:#e10600; border-radius:50%; animation:rp-spin 1s linear infinite; margin-bottom:10px;"></div>
      <div id="rpLoaderTxt" style="font-weight:bold;">Downloading Telemetry...</div>
    </div>
  </div>
  <style>@keyframes rp-spin { 100%{transform:rotate(360deg)} } input[type=range]{-webkit-appearance:none;background:transparent} input[type=range]::-webkit-slider-runnable-track{height:4px;background:#333;border-radius:2px} input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;background:#e10600;border-radius:50%;margin-top:-5px;cursor:pointer}</style>
  <div style="display:flex; justify-content:center; align-items:center; gap: 16px; padding: 10px;">
    <button onclick="rpTogglePlay()" id="rpPlayBtn" style="background:#222;color:#fff;border:1px solid #444;border-radius:4px;padding:4px 12px;cursor:pointer;">▶</button>
    <div id="rpTimeCur" style="font-variant-numeric:tabular-nums;font-size:13px;width:40px;text-align:right;">0:00</div>
    <input type="range" id="rpScrubber" min="0" max="1000" value="0" style="flex:1;max-width:600px;">
    <div id="rpTimeMax" style="font-variant-numeric:tabular-nums;font-size:13px;width:40px;">0:00</div>
    <button onclick="rpCycleSpeed()" id="rpSpeedBtn" style="background:#222;color:#fff;border:1px solid #444;border-radius:4px;padding:4px 12px;cursor:pointer;">1x</button>
  </div>
</div>''',
    html, flags=re.DOTALL
)

# 3. Tabs
html = re.sub(
    r'<button class="tab-btn" id="tbtn-replay".*?Race Replay</button>',
    '', html
)
html = re.sub(
    r'<!-- Tab 4: Race Replay -->.*?<!-- Tab 5: Guide -->',
    '<!-- Tab 5: Guide -->',
    html, flags=re.DOTALL
)

# 4. JS Replace
html = re.sub(
    r'// ── Main load ──.*?</html>',
    '''// ── Replay Dashboard Logic ─────────────────────────────────────────────
let rpData = null, rpTime = 0, rpPlaying = false, rpLastTick = 0;
let rpSpeeds = [0.5, 1, 2, 4, 10], rpSpeedIdx = 1;
let rpCam = {x:0, y:0, scale:1}, rpDrag = false, rpDragStart = {x:0, y:0};
let rpRafId = null, rpScrubbing = false;
let lastStrategyUpdate = -1;

const rpCanvas = document.getElementById('replayCanvas');
const rpCtx = rpCanvas.getContext('2d');

function rpResize(){
  const wrap = document.getElementById('replayWrap');
  if(!wrap) return;
  rpCanvas.width = wrap.clientWidth;
  rpCanvas.height = wrap.clientHeight;
  if(rpData && !rpPlaying) rpDraw();
}
window.addEventListener('resize', rpResize);
setTimeout(rpResize, 100);

function rpFmtTime(s){ 
  if(isNaN(s)||s<0) return '0:00';
  return Math.floor(s/60)+':'+Math.floor(s%60).toString().padStart(2,'0');
}

async function rpLoad() {
    const year = document.getElementById('rpYear').value;
    const race = document.getElementById('rpRace').value;
    const sess = document.getElementById('rpSession').value;
    
    document.getElementById('rpLoader').style.display = 'flex';
    document.getElementById('raceStatus').textContent = "🚀 Loading Telemetry...";
    document.getElementById('raceStatus').className = "s-yellow";
    
    rpPlaying = false; 
    if(rpRafId) cancelAnimationFrame(rpRafId);
    
    try {
        const res = await fetch(`/api/replay_data?year=${year}&race=${encodeURIComponent(race)}&session=${sess}&v=${Date.now()}`);
        const data = await res.json();
        if(data.error) throw new Error(data.error);
        
        rpData = data; 
        rpTime = 0; 
        lastStrategyUpdate = -1;
        rpSpeedIdx = 1; document.getElementById('rpSpeedBtn').textContent='1x';
        
        if(data.track_x && data.track_x.length){
            let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
            data.track_x.forEach((x,i)=>{ 
                let y=data.track_y[i]; 
                if(x<minX)minX=x; if(x>maxX)maxX=x;
                if(y<minY)minY=y; if(y>maxY)maxY=y; 
            });
            const pad=60, tw=(maxX-minX)+pad*2, th=(maxY-minY)+pad*2;
            const sc=Math.min(rpCanvas.width/tw, rpCanvas.height/th)*0.95;
            rpCam={scale:sc, x:rpCanvas.width/2-((minX+maxX)/2)*sc, y:rpCanvas.height/2+((minY+maxY)/2)*sc};
        }
        
        document.getElementById('rpLoader').style.display = 'none';
        
        document.getElementById('headerTitle').textContent = year + ' ' + race;
        
        rpTogglePlay();
    } catch(e) {
        document.getElementById('rpLoaderTxt').textContent = 'Error: ' + e.message;
        document.getElementById('raceStatus').textContent = "⚠️ " + e.message;
        document.getElementById('raceStatus').className = "s-red";
        setTimeout(()=>document.getElementById('rpLoader').style.display='none',4000);
    }
}

function rpTogglePlay(){
    if(!rpData) return;
    rpPlaying = !rpPlaying;
    document.getElementById('rpPlayBtn').textContent = rpPlaying ? '⏸' : '▶';
    if(rpPlaying){ 
        rpLastTick = performance.now(); 
        rpRafId = requestAnimationFrame(rpTick); 
    }
    else cancelAnimationFrame(rpRafId);
}
function rpCycleSpeed(){
    rpSpeedIdx = (rpSpeedIdx+1) % rpSpeeds.length;
    document.getElementById('rpSpeedBtn').textContent = rpSpeeds[rpSpeedIdx]+'x';
}

const rpScrubEl = document.getElementById('rpScrubber');
rpScrubEl.addEventListener('mousedown', ()=>rpScrubbing=true);
window.addEventListener('mouseup', ()=>{ if(rpScrubbing){rpScrubbing=false; rpLastTick=performance.now();} });
rpScrubEl.addEventListener('input', e=>{ 
    if(!rpData) return; 
    rpTime = (e.target.value/1000) * rpData.t_max; 
    rpDraw(); 
    doStrategyUpdate();
});

rpCanvas.addEventListener('mousedown', e=>{ rpDrag=true; rpDragStart={x:e.clientX-rpCam.x, y:e.clientY-rpCam.y}; });
window.addEventListener('mouseup', ()=>rpDrag=false);
window.addEventListener('mousemove', e=>{ if(rpDrag){rpCam.x=e.clientX-rpDragStart.x; rpCam.y=e.clientY-rpDragStart.y; if(!rpPlaying)rpDraw();} });
rpCanvas.addEventListener('wheel', e=>{
    e.preventDefault();
    const zm = e.deltaY<0 ? 1.1 : 0.9;
    rpCam.x = e.offsetX - (e.offsetX-rpCam.x)*zm;
    rpCam.y = e.offsetY - (e.offsetY-rpCam.y)*zm;
    rpCam.scale *= zm;
    if(!rpPlaying) rpDraw();
},{passive:false});

function rpTick(t){
    if(!rpPlaying) return;
    const dt = t - rpLastTick; rpLastTick = t;
    rpTime += (dt/1000) * rpSpeeds[rpSpeedIdx];
    if(rpTime >= rpData.t_max){ rpTime = rpData.t_max; rpPlaying = false; document.getElementById('rpPlayBtn').textContent='▶'; }
    rpDraw();
    
    if (Math.abs(rpTime - lastStrategyUpdate) > 2.0) {
        doStrategyUpdate();
        lastStrategyUpdate = rpTime;
    }
    
    if(rpPlaying && !rpScrubbing) rpRafId = requestAnimationFrame(rpTick);
}

function linearSlopeJs(values) {
    const n = values.length;
    if (n < 2) return null;
    const sx = n * (n - 1) / 2;
    let sy = 0, sxy = 0;
    for (let i = 0; i < n; i++) {
        sy += values[i];
        sxy += i * values[i];
    }
    const sx2 = n * (n - 1) * (2 * n - 1) / 6;
    const denom = n * sx2 - sx * sx;
    return denom ? (n * sxy - sx * sy) / denom : null;
}

function formatLapTime(time) {
    if(!time) return "–";
    const m = Math.floor(time/60);
    const s = (time%60).toFixed(3).padStart(6, '0');
    return `${m}:${s}`;
}

function doStrategyUpdate() {
    if(!rpData) return;
    const state = rpGetState(true);
    if(!state) return;

    const sEl = document.getElementById("raceStatus");
    const ts = state.sc ? state.sc.phase : null;
    let sText="🟢  GREEN FLAG", sCls="s-green";
    if(state.sc && state.sc.alpha > 0.1) { sText="🟡  SAFETY CAR"; sCls="s-yellow"; }
    sEl.textContent = sText; sEl.className = sCls;
    
    const classified = state.drivers.filter(d => !d.retired).sort((a,b)=>a.position - b.position);
    const dnf = state.drivers.filter(d => d.retired);
    
    renderLiveTable(classified, dnf);
    updateFastestLap(classified);
    generateInsights(classified);
    renderPitWindows(classified);
    drawStintGantt(classified);
    renderUndercut(classified);
    renderOvercut(classified);
    renderPaceTable(classified);
    renderDegTable(classified);
    renderSectorTimes(classified);
    
    const all = [...classified, ...dnf];
    updateDriverChips(all);
    drawLapTimeChart(all);
}

function rpGetState(detailed = false){
    if(!rpData || !rpData.frames || rpData.frames.length===0) return null;
    const fi = rpTime / rpData.frame_step;
    const i0 = Math.max(0, Math.floor(fi)), i1 = Math.min(i0+1, rpData.frames.length-1);
    const fr = fi - i0;
    const f0 = rpData.frames[i0], f1 = rpData.frames[i1];
    
    let state = { drivers: [], sc: null, weather: f0.weather || null };
    let d1m = {}; f1.drivers.forEach(d=>d1m[d.code]=d);
    
    f0.drivers.forEach(d0 => {
        let d1 = d1m[d0.code] || d0;
        state.drivers.push({...d0,
            x: d0.x+(d1.x-d0.x)*fr, y: d0.y+(d1.y-d0.y)*fr,
            s: d0.s+(d1.s-d0.s)*fr, t: d0.t+(d1.t-d0.t)*fr, b: d0.b+(d1.b-d0.b)*fr
        });
    });
    
    if(f0.sc){ 
        state.sc = f1.sc ? {
            x: f0.sc.x+(f1.sc.x-f0.sc.x)*fr, y: f0.sc.y+(f1.sc.y-f0.sc.y)*fr, 
            phase: f0.sc.phase, alpha: f0.sc.alpha+(f1.sc.alpha-f0.sc.alpha)*fr
        } : f0.sc; 
    }
    
    const tx = rpData.track_x, ty = rpData.track_y, tn = tx ? tx.length : 0;
    state.drivers.forEach(d => {
        if(tn>0){
            let best=0, bDist=Infinity;
            for(let i=0; i<tn; i++){
                const dx=d.x-tx[i], dy=d.y-ty[i], dsq=dx*dx+dy*dy;
                if(dsq<bDist){ bDist=dsq; best=i; }
            }
            d.prog = best / tn;
        } else d.prog = 0;
    });
    
    state.drivers.forEach(d => {
        const history = (rpData.driver_laps && rpData.driver_laps[d.code]) ? rpData.driver_laps[d.code] : [];
        let curIdx = -1;
        for(let j=0; j<history.length; j++){
            if(history[j].t <= rpTime) curIdx = j;
            else break;
        }
        
        d.last_lap_rec = curIdx >= 0 ? history[curIdx] : null;
        d.lap = curIdx >= 0 ? d.last_lap_rec.lap : 0;
        d.abs_prog = d.lap + d.prog;
        
        const is_stopped = d.s < 5;
        d.retired = false; 
        
        if (detailed && history.length > 0) {
            d.driver_number = d.code;
            d.driver_code = d.code;
            d.position = null;
            d.lap_number = Math.floor(d.abs_prog);
            d.compound = d.last_lap_rec ? d.last_lap_rec.tyre : history[0].tyre;
            d.tyre_age = Math.round(((d.last_lap_rec ? d.last_lap_rec.age : 0) + d.prog) * 10) / 10; 
            d.stint = d.last_lap_rec ? d.last_lap_rec.stint : 1;
            d.last_lap = d.last_lap_rec ? formatLapTime(d.last_lap_rec.time) : "–";
            
            const hist_slice = history.slice(0, curIdx+1);
            
            d.stint_history = [];
            let c_stint = null;
            for(let hr of hist_slice) {
                if(!c_stint || c_stint.stint !== hr.stint) {
                    if(c_stint) c_stint.end_lap = hr.lap - 1;
                    c_stint = {stint: hr.stint, compound: hr.tyre, start_lap: hr.lap, end_lap: hr.lap, laps_on_set: 1};
                    d.stint_history.push(c_stint);
                } else {
                    c_stint.end_lap = hr.lap;
                    c_stint.laps_on_set = hr.lap - c_stint.start_lap + 1;
                }
            }
            if(d.stint_history.length) {
                d.stint_history[d.stint_history.length-1].end_lap = d.lap;
                d.stint_history[d.stint_history.length-1].laps_on_set = d.lap - d.stint_history[d.stint_history.length-1].start_lap + 1;
            }
            
            const last6 = history.slice(Math.max(0, curIdx-5), curIdx+1).map(x => x.time);
            d.lap_history = last6;
            
            const last8 = history.slice(Math.max(0, curIdx-7), curIdx+1).map(x => x.time);
            d.tyre_deg = linearSlopeJs(last8);
            
            const last3 = history.slice(Math.max(0, curIdx-2), curIdx+1).map(x => x.time);
            d.avg_pace_3 = last3.length >= 2 ? (last3.reduce((a,b)=>a+b, 0)/last3.length) : null;
            
            const s1s = hist_slice.map(x=>x.s1).filter(v=>v);
            const s2s = hist_slice.map(x=>x.s2).filter(v=>v);
            const s3s = hist_slice.map(x=>x.s3).filter(v=>v);
            d.s1_best = s1s.length ? Math.min(...s1s) : null;
            d.s2_best = s2s.length ? Math.min(...s2s) : null;
            d.s3_best = s3s.length ? Math.min(...s3s) : null;
            d.s1_last = d.last_lap_rec ? d.last_lap_rec.s1 : null;
            d.s2_last = d.last_lap_rec ? d.last_lap_rec.s2 : null;
            d.s3_last = d.last_lap_rec ? d.last_lap_rec.s3 : null;
            
            const total_lap_time = hist_slice.reduce((acc, curr) => acc + curr.time, 0);
            const current_lap_est = (d.last_lap_rec ? d.last_lap_rec.time : 90) * d.prog;
            d._cum_time = total_lap_time + current_lap_est;
        } else {
            d.driver_number = d.code; d.driver_code = d.code; d.position = null; d.lap_number = 0;
            d.compound = "–"; d.tyre_age = 0; d.stint = 1; d.last_lap = "–"; d.stint_history = [];
            d.lap_history = []; d.tyre_deg = null; d.avg_pace_3 = null;
            d.s1_best = null; d.s2_best = null; d.s3_best = null; d.s1_last = null; d.s2_last = null; d.s3_last = null;
            d._cum_time = (90) * d.prog;
        }
    });
    
    state.drivers.sort((a,b) => b.abs_prog - a.abs_prog);
    
    const maxProg = state.drivers[0] ? state.drivers[0].abs_prog : 0;
    
    if (detailed && state.drivers.length > 0) {
        state.drivers.forEach((d, i) => {
            // Find retired
            if (maxProg > 5 && (maxProg - d.abs_prog > 5) && d.s < 2) {
                d.retired = true;
                d.retire_reason = "DNF";
                d.dnf_lap = d.lap;
                d._cum_time = Infinity;
            }
        });
        
        state.drivers.sort((a,b) => a._cum_time - b._cum_time);
        
        const cum_leader = state.drivers[0]._cum_time;
        state.drivers.forEach((d, i) => {
            if (!d.retired) {
                d.position = i + 1;
                d.gap_to_leader = d._cum_time - cum_leader;
                d.gap_to_ahead = i === 0 ? 0 : (d._cum_time - state.drivers[i-1]._cum_time);
            }
        });
    }
    
    return state;
}

function rpDraw(){
    if(!rpData) return;
    rpCtx.clearRect(0,0, rpCanvas.width, rpCanvas.height);
    rpCtx.save();
    rpCtx.translate(rpCam.x, rpCam.y);
    rpCtx.scale(rpCam.scale, -rpCam.scale);
    
    if(rpData.track_x){
        rpCtx.beginPath();
        for(let i=0; i<rpData.track_x.length; i++){
            i===0 ? rpCtx.moveTo(rpData.track_x[i], rpData.track_y[i]) : rpCtx.lineTo(rpData.track_x[i], rpData.track_y[i]);
        }
        rpCtx.strokeStyle='rgba(40,40,40,0.9)'; rpCtx.lineWidth=14/rpCam.scale; rpCtx.lineJoin='round'; rpCtx.lineCap='round'; rpCtx.stroke();
        rpCtx.strokeStyle='rgba(255,255,255,0.7)'; rpCtx.lineWidth=1.5/rpCam.scale; rpCtx.stroke();
    }
    
    const state = rpGetState(false); 
    if(!state){ rpCtx.restore(); return; }
    
    if(state.sc && state.sc.alpha){
        rpCtx.beginPath(); rpCtx.arc(state.sc.x, state.sc.y, 8/rpCam.scale, 0, Math.PI*2);
        rpCtx.fillStyle = `rgba(255,165,0,${state.sc.alpha})`; rpCtx.fill();
        rpCtx.lineWidth = 2/rpCam.scale; rpCtx.strokeStyle = `rgba(255,200,0,${state.sc.alpha})`; rpCtx.stroke();
        rpCtx.save(); rpCtx.scale(1,-1);
        rpCtx.font = `bold ${9/rpCam.scale}px Arial`; rpCtx.fillStyle = `rgba(255,255,255,${state.sc.alpha})`;
        rpCtx.fillText('SC', state.sc.x+9/rpCam.scale, -state.sc.y-10/rpCam.scale);
        rpCtx.restore();
    }
    
    state.drivers.forEach(d => {
        const col = teamColors[d.code] || '#fff';
        const r = 5/rpCam.scale;
        rpCtx.beginPath(); rpCtx.arc(d.x, d.y, r, 0, Math.PI*2);
        rpCtx.fillStyle = col; rpCtx.fill();
        rpCtx.save(); rpCtx.scale(1,-1);
        rpCtx.font = `bold ${10/rpCam.scale}px Arial`; rpCtx.fillStyle = col;
        rpCtx.fillText(d.code, d.x+7/rpCam.scale, -d.y);
        rpCtx.restore();
    });
    
    rpCtx.restore();
    
    if(!rpScrubbing) rpScrubEl.value = (rpTime/(rpData.t_max||1))*1000;
    document.getElementById('rpTimeCur').textContent = rpFmtTime(rpTime);
    if(rpData.t_max) document.getElementById('rpTimeMax').textContent = rpFmtTime(rpData.t_max);
}

// Initial draw empty
rpCtx.fillStyle='#000';
rpCtx.fillRect(0,0,rpCanvas.width,rpCanvas.height);

// Boot
switchTab('live');
</script>
</body>
</html>''', html, flags=re.DOTALL
)

with open('dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)
