const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');
const els = {
  mode: document.getElementById('modeBadge'), fix: document.getElementById('fixText'), sats: document.getElementById('satText'),
  speed: document.getElementById('speed'), heading: document.getElementById('heading'), distance: document.getElementById('distance'), drift: document.getElementById('drift'),
  start: document.getElementById('startBtn'), outage: document.getElementById('outageBtn'), reset: document.getElementById('resetBtn'), message: document.getElementById('message')
};

const replay = window.DEMO_TRAJECTORY || [];
let running = false, outage = false, timer = null, frame = 0, distance = 0, trail = [];

function project(lat, lon, w, h) {
  if (!replay.length) return [w / 2, h / 2];
  const lats = replay.map(p => p.lat), lons = replay.map(p => p.lon);
  const minLat = Math.min(...lats), maxLat = Math.max(...lats), minLon = Math.min(...lons), maxLon = Math.max(...lons);
  const pad = 45;
  return [pad + (lon-minLon)/Math.max(maxLon-minLon,1e-9)*(w-2*pad), h-pad-(lat-minLat)/Math.max(maxLat-minLat,1e-9)*(h-2*pad)];
}
function resize(){const dpr=Math.min(devicePixelRatio||1,2);canvas.width=canvas.clientWidth*dpr;canvas.height=canvas.clientHeight*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);draw();}
window.addEventListener('resize',resize);
function draw(){
  const w=canvas.clientWidth,h=canvas.clientHeight;ctx.clearRect(0,0,w,h);ctx.fillStyle='#0b1727';ctx.fillRect(0,0,w,h);
  ctx.strokeStyle='#16283d';ctx.lineWidth=1;
  for(let x=0;x<w;x+=36){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}for(let y=0;y<h;y+=36){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}
  if(replay.length){ctx.strokeStyle='#49627c';ctx.lineWidth=13;ctx.lineCap='round';ctx.beginPath();replay.forEach((p,i)=>{const q=project(p.lat,p.lon,w,h);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();ctx.strokeStyle='#a8bfd8';ctx.lineWidth=2;ctx.setLineDash([10,10]);ctx.beginPath();replay.forEach((p,i)=>{const q=project(p.lat,p.lon,w,h);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();ctx.setLineDash([])}
  if(trail.length>1){ctx.strokeStyle='#58d68d';ctx.lineWidth=4;ctx.beginPath();trail.forEach((q,i)=>i?ctx.lineTo(...q):ctx.moveTo(...q));ctx.stroke()}
  const p=replay[Math.min(frame,replay.length-1)];if(p){const q=project(p.lat,p.lon,w,h);ctx.fillStyle=outage?'#ff5c5c':'#55b7ff';ctx.beginPath();ctx.arc(q[0],q[1],9,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke()}
}
function setMode(){els.mode.textContent=outage?'DEAD RECKONING':'GNSS AIDED';els.mode.className=`badge ${outage?'dr':'gnss'}`;els.fix.textContent=outage?'GNSS OUTAGE':'GNSS FIX';els.sats.textContent=outage?'SAT --':'SAT 18';els.outage.textContent=outage?'RESTORE GNSS':'SIMULATE GNSS OUTAGE';}
function tick(){
  if(!replay.length)return;const p=replay[frame];distance+=p.speedKmh/36;const q=project(p.lat,p.lon,canvas.clientWidth,canvas.clientHeight);trail.push(q);if(trail.length>100)trail.shift();
  els.speed.textContent=p.speedKmh.toFixed(1);els.heading.textContent=Math.round(p.headingDeg);els.distance.textContent=Math.round(distance);els.drift.textContent=outage?(0.12*Math.sqrt(distance)).toFixed(1):'0.0';
  els.message.textContent=outage?'GNSS unavailable — replay is running in DR mode.':'GNSS aiding active — replay position is available.';draw();frame=(frame+1)%replay.length;
}
function start(){if(running){stop();return}running=true;els.start.textContent='PAUSE DEMO';timer=setInterval(tick,100)}
function stop(){running=false;clearInterval(timer);timer=null;els.start.textContent='START DEMO'}
function reset(){stop();outage=false;frame=0;distance=0;trail=[];els.speed.textContent='0.0';els.heading.textContent='0';els.distance.textContent='0';els.drift.textContent='0.0';setMode();els.message.textContent='Ready. Start the replay to simulate vehicle navigation.';draw()}
els.start.onclick=start;els.outage.onclick=()=>{outage=!outage;setMode()};els.reset.onclick=reset;resize();setMode();
