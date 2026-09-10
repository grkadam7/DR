const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');
const els = {
  mode: document.getElementById('modeBadge'), fix: document.getElementById('fixText'), sats: document.getElementById('satText'),
  speed: document.getElementById('speed'), heading: document.getElementById('heading'), distance: document.getElementById('distance'), drift: document.getElementById('drift'),
  start: document.getElementById('startBtn'), outage: document.getElementById('outageBtn'), reset: document.getElementById('resetBtn'), message: document.getElementById('message')
};

let running = false, outage = false, timer = null, t = 0, distance = 0, trail = [];
const road = Array.from({length: 360}, (_, i) => {
  const x = 70 + i * 2.3;
  const y = 260 + 48 * Math.sin(i / 42) + 18 * Math.sin(i / 17);
  return [x, y];
});

function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = canvas.clientWidth * dpr; canvas.height = canvas.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
}
window.addEventListener('resize', resize);

function draw() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0b1727'; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#16283d'; ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 36) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  for (let y = 0; y < h; y += 36) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  ctx.strokeStyle = '#49627c'; ctx.lineWidth = 13; ctx.lineCap = 'round';
  ctx.beginPath(); road.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y)); ctx.stroke();
  ctx.strokeStyle = '#a8bfd8'; ctx.lineWidth = 2; ctx.setLineDash([10,10]);
  ctx.beginPath(); road.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y)); ctx.stroke(); ctx.setLineDash([]);
  if (trail.length > 1) {
    ctx.strokeStyle = '#58d68d'; ctx.lineWidth = 4; ctx.beginPath(); trail.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y)); ctx.stroke();
  }
  const p = road[Math.min(Math.floor(t), road.length-1)];
  ctx.fillStyle = outage ? '#ff5c5c' : '#55b7ff'; ctx.beginPath(); ctx.arc(p[0],p[1],9,0,Math.PI*2); ctx.fill();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
}

function setMode() {
  els.mode.textContent = outage ? 'DEAD RECKONING' : 'GNSS AIDED';
  els.mode.className = `badge ${outage ? 'dr' : 'gnss'}`;
  els.fix.textContent = outage ? 'GNSS OUTAGE' : 'GNSS FIX';
  els.sats.textContent = outage ? 'SAT --' : 'SAT 18';
  els.outage.textContent = outage ? 'RESTORE GNSS' : 'SIMULATE GNSS OUTAGE';
}
function tick() {
  t += .7; distance += .7 * 14 / 3.6;
  const i = Math.min(Math.floor(t), road.length - 1); trail.push(road[i]); if (trail.length > 90) trail.shift();
  const speed = 48 + 7 * Math.sin(t/18), heading = (90 + 25*Math.cos(t/31) + 360) % 360;
  els.speed.textContent = speed.toFixed(1); els.heading.textContent = Math.round(heading); els.distance.textContent = Math.round(distance);
  els.drift.textContent = outage ? (0.12 * Math.sqrt(distance)).toFixed(1) : '0.0';
  els.message.textContent = outage ? 'GNSS unavailable — DR engine is propagating the vehicle state.' : 'GNSS aiding active — navigation state is being corrected.';
  draw();
  if (i >= road.length-1) stop();
}
function start() { if (running) return; running=true; els.start.textContent='PAUSE DEMO'; timer=setInterval(tick,100); }
function stop() { running=false; clearInterval(timer); timer=null; els.start.textContent='START DEMO'; }
function reset() { stop(); outage=false; t=0; distance=0; trail=[]; els.speed.textContent='0.0'; els.heading.textContent='0'; els.distance.textContent='0'; els.drift.textContent='0.0'; setMode(); els.message.textContent='Ready. Start the replay to simulate vehicle navigation.'; draw(); }
els.start.onclick=()=>running?stop():start();
els.outage.onclick=()=>{ outage=!outage; setMode(); };
els.reset.onclick=reset;
resize(); setMode();
