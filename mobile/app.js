const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');
const els = {
  mode: document.getElementById('modeBadge'), fix: document.getElementById('fixText'), sats: document.getElementById('satText'),
  speed: document.getElementById('speed'), heading: document.getElementById('heading'), distance: document.getElementById('distance'), drift: document.getElementById('drift'),
  start: document.getElementById('startBtn'), outage: document.getElementById('outageBtn'), reset: document.getElementById('resetBtn'), message: document.getElementById('message')
};

const rawReplay = window.DEMO_TRAJECTORY || [];
const replay = Array.isArray(rawReplay) ? rawReplay : (rawReplay.points || []);
// Accept both the original demo format and the generated IO-VNBD format.
const points = replay.map(p => ({
  t: Number(p.t || 0), lat: Number(p.lat), lon: Number(p.lon),
  speed: Number.isFinite(Number(p.speed)) ? Number(p.speed) : Number(p.speedKmh || 0) / 3.6,
  heading: Number.isFinite(Number(p.heading)) ? Number(p.heading) : Number(p.headingDeg || 0)
}));

let running = false, outage = false, timer = null, frame = 0, distance = 0, trail = [];
let drLat = null, drLon = null, lastReplayTime = null;

function project(lat, lon, w, h) {
  if (!points.length) return [w / 2, h / 2];
  const lats = points.map(p => p.lat), lons = points.map(p => p.lon);
  const minLat = Math.min(...lats), maxLat = Math.max(...lats), minLon = Math.min(...lons), maxLon = Math.max(...lons);
  const pad = 45;
  return [pad + (lon-minLon)/Math.max(maxLon-minLon,1e-9)*(w-2*pad), h-pad-(lat-minLat)/Math.max(maxLat-minLat,1e-9)*(h-2*pad)];
}
function resize(){const dpr=Math.min(devicePixelRatio||1,2);canvas.width=canvas.clientWidth*dpr;canvas.height=canvas.clientHeight*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);draw();}
window.addEventListener('resize',resize);

function haversineM(lat1, lon1, lat2, lon2) {
  const R = 6371000, r = Math.PI / 180;
  const dLat = (lat2-lat1)*r, dLon = (lon2-lon1)*r;
  const a = Math.sin(dLat/2)**2 + Math.cos(lat1*r)*Math.cos(lat2*r)*Math.sin(dLon/2)**2;
  return 2*R*Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

function propagate(lat, lon, speedMps, headingDeg, dt) {
  const R = 6378137;
  const d = Math.max(0, speedMps) * Math.max(0, dt);
  const brg = headingDeg * Math.PI / 180;
  const latR = lat * Math.PI / 180;
  const dLat = (d * Math.cos(brg)) / R;
  const dLon = (d * Math.sin(brg)) / (R * Math.max(Math.cos(latR), 1e-6));
  return [lat + dLat * 180 / Math.PI, lon + dLon * 180 / Math.PI];
}

function draw(){
  const w=canvas.clientWidth,h=canvas.clientHeight;ctx.clearRect(0,0,w,h);ctx.fillStyle='#0b1727';ctx.fillRect(0,0,w,h);
  ctx.strokeStyle='#16283d';ctx.lineWidth=1;
  for(let x=0;x<w;x+=36){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}
  for(let y=0;y<h;y+=36){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}
  if(points.length){
    // GNSS reference trajectory.
    ctx.strokeStyle='#49627c';ctx.lineWidth=11;ctx.lineCap='round';ctx.beginPath();
    points.forEach((p,i)=>{const q=project(p.lat,p.lon,w,h);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();
    ctx.strokeStyle='#a8bfd8';ctx.lineWidth=1.5;ctx.setLineDash([8,8]);ctx.beginPath();
    points.forEach((p,i)=>{const q=project(p.lat,p.lon,w,h);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();ctx.setLineDash([]);
  }
  if(trail.length>1){
    ctx.strokeStyle='#58d68d';ctx.lineWidth=4;ctx.beginPath();trail.forEach((q,i)=>i?ctx.lineTo(...q):ctx.moveTo(...q));ctx.stroke();
  }
  const p=points[Math.min(frame,points.length-1)];
  if(p){
    const pos = outage && drLat !== null ? [drLat,drLon] : [p.lat,p.lon];
    const q=project(pos[0],pos[1],w,h);
    ctx.fillStyle=outage?'#ff5c5c':'#55b7ff';ctx.beginPath();ctx.arc(q[0],q[1],9,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
  }
}

function setMode(){
  els.mode.textContent=outage?'DEAD RECKONING':'GNSS AIDED';
  els.mode.className=`badge ${outage?'dr':'gnss'}`;
  els.fix.textContent=outage?'GNSS OUTAGE':'GNSS FIX';
  els.sats.textContent=outage?'SAT --':'SAT 18';
  els.outage.textContent=outage?'RESTORE GNSS':'SIMULATE GNSS OUTAGE';
}

function tick(){
  if(!points.length)return;
  const p=points[frame];
  const prev = points[Math.max(0, frame-1)];
  const dt = frame === 0 ? 0.1 : Math.max(0.01, p.t - prev.t || 0.1);
  distance += p.speed * dt;

  if (!outage) {
    drLat = p.lat; drLon = p.lon;
  } else if (drLat !== null) {
    [drLat, drLon] = propagate(drLat, drLon, p.speed, p.heading, dt);
  }

  const shownLat = outage ? drLat : p.lat;
  const shownLon = outage ? drLon : p.lon;
  const q=project(shownLat,shownLon,canvas.clientWidth,canvas.clientHeight);
  trail.push(q);if(trail.length>180)trail.shift();

  els.speed.textContent=(p.speed*3.6).toFixed(1);
  els.heading.textContent=Math.round(p.heading);
  els.distance.textContent=Math.round(distance);
  els.drift.textContent=outage?haversineM(drLat,drLon,p.lat,p.lon).toFixed(1):'0.0';
  els.message.textContent=outage
    ? 'GNSS unavailable — speed + course replay is propagating the DR estimate.'
    : 'GNSS aiding active — replay position is available.';
  draw();
  lastReplayTime = p.t;
  frame=(frame+1)%points.length;
}

function start(){if(running){stop();return}running=true;els.start.textContent='PAUSE DEMO';timer=setInterval(tick,100)}
function stop(){running=false;clearInterval(timer);timer=null;els.start.textContent='START DEMO'}
function reset(){
  stop();outage=false;frame=0;distance=0;trail=[];drLat=points.length?points[0].lat:null;drLon=points.length?points[0].lon:null;lastReplayTime=null;
  els.speed.textContent='0.0';els.heading.textContent='0';els.distance.textContent='0';els.drift.textContent='0.0';setMode();
  els.message.textContent='Ready. Start the replay to simulate vehicle navigation.';draw();
}

els.start.onclick=start;
els.outage.onclick=()=>{
  if (!points.length) return;
  outage=!outage;
  const p=points[Math.min(frame,points.length-1)];
  // GNSS outage starts exactly from the last known GNSS position.
  if(outage){drLat=p.lat;drLon=p.lon;lastReplayTime=p.t;}
  else {drLat=p.lat;drLon=p.lon;}
  setMode();draw();
};
els.reset.onclick=reset;
resize();setMode();reset();
