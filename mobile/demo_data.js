// Replay-friendly trajectory generated from the same vehicle-navigation concept as IO-VNBD.
// The UI consumes this format so real CSV/engine output can replace it without changing the screen.
window.DEMO_TRAJECTORY = Array.from({ length: 240 }, (_, i) => {
  const t = i * 0.1;
  const speedKmh = 45 + 8 * Math.sin(t / 4);
  const headingDeg = 72 + 16 * Math.sin(t / 7);
  const rad = headingDeg * Math.PI / 180;
  const lat = 12.9716 + (i * 0.000004 * Math.cos(rad));
  const lon = 77.5946 + (i * 0.000004 * Math.sin(rad));
  return { t, speedKmh, headingDeg, lat, lon };
});
