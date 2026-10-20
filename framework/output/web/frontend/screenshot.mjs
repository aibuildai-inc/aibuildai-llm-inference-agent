import { chromium } from "playwright";

const S = "/tmp/claude-1011/-data2-peijia-dr-claw-aibuildai-Experiment/3c63f76a-c023-48db-84be-ddb0ea42fef6/scratchpad";

function tierColor(pct) {
  if (pct >= 90) return "#D4483B";
  if (pct >= 70) return "#E8590C";
  if (pct >= 40) return "#2E8B57";
  return "#5B7A9D";
}

function gpuRow(g) {
  const memPct = Math.round((g.memUsed / g.memTotal) * 100);
  const pwrPct = Math.round((g.power / g.powerCap) * 100);
  const utilColor = tierColor(g.util);
  const memColor = tierColor(memPct);
  const highlight = g.thisRun ? 'data-owned="true"' : '';
  return `
  <div class="gpu-row" ${highlight}>
    <span class="gpu-idx">${g.idx}</span>
    <span class="gpu-name">${g.name}</span>
    <span class="gpu-temp ${g.temp >= 75 ? 'hot' : ''}">${g.temp}°C</span>
    <span class="gpu-pwr num">${g.power}W<span class="pwr-cap">/${g.powerCap}W</span></span>

    <span class="gpu-meter-cell">
      <span class="meter-label">UTIL</span>
      <span class="meter-track"><span class="meter-fill" style="width:${g.util}%;background:${utilColor}"></span></span>
      <span class="meter-val num" style="color:${utilColor}">${g.util}%</span>
    </span>

    <span class="gpu-meter-cell mem-cell">
      <span class="meter-label">MEM</span>
      <span class="meter-track"><span class="meter-fill" style="width:${memPct}%;background:${memColor}"></span></span>
      <span class="meter-val num" style="color:${memColor}">${g.memUsed}<span class="mem-total">/${g.memTotal}</span></span>
    </span>

    <span class="gpu-meter-cell bw-cell">
      <span class="meter-label">BW</span>
      <span class="meter-track sm"><span class="meter-fill" style="width:${g.memBw}%;background:#5B7A9D"></span></span>
      <span class="meter-val num">${g.memBw}%</span>
    </span>

    <span class="gpu-owner ${g.thisRun ? 'owned' : ''}">${g.ownerLabel}</span>
  </div>`;
}

const gpus = [
  { idx: 0, name: "A100 80GB", temp: 71, power: 294, powerCap: 300, memUsed: 41.9, memTotal: 80, util: 92, memBw: 59, ownerLabel: "ruiyi · vLLM eval" },
  { idx: 1, name: "A100 80GB", temp: 66, power: 280, powerCap: 300, memUsed: 68.4, memTotal: 80, util: 86, memBw: 47, ownerLabel: "ruiyi · vLLM eval" },
  { idx: 2, name: "A100 80GB", temp: 74, power: 315, powerCap: 300, memUsed: 54.8, memTotal: 80, util: 100, memBw: 56, ownerLabel: "ruiyi · ptb-gsm8k" },
  { idx: 3, name: "A100 80GB", temp: 72, power: 287, powerCap: 300, memUsed: 66.6, memTotal: 80, util: 98, memBw: 46, thisRun: true, ownerLabel: "peijia · this run" },
];

const HTML = `<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', system-ui, sans-serif; background: #1A1816; color: #D0CCC6; }
.num { font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; }

.dock { padding: 10px 16px 14px; }

/* ── Top bar ── */
.dock-bar {
  display: flex; align-items: center; gap: 16px;
  padding-bottom: 8px; border-bottom: 1px solid #2A2622;
  font-size: 10px; color: #6D675F;
}
.dock-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #8A8076; }
.dock-bar b { color: #D0CCC6; font-weight: 600; }

/* ── GPU rows ── */
.gpu-stack { display: flex; flex-direction: column; margin-top: 8px; }

.gpu-row {
  display: flex; align-items: center; gap: 10px;
  padding: 6px 10px;
  border-bottom: 1px solid #222;
  font-size: 11px;
}
.gpu-row:last-child { border-bottom: 0; }
.gpu-row[data-owned] { background: rgba(232,89,12,0.06); }

.gpu-idx {
  width: 18px; text-align: center;
  font-weight: 700; font-size: 13px; color: #5A5550;
  flex: none;
}
.gpu-row[data-owned] .gpu-idx { color: #E8590C; }

.gpu-name { width: 80px; font-size: 10px; color: #5A5550; flex: none; }
.gpu-temp { width: 38px; font-weight: 600; color: #84661F; flex: none; text-align: right; }
.gpu-temp.hot { color: #D4483B; }
.gpu-pwr { width: 90px; font-size: 10px; color: #5A5550; flex: none; text-align: right; }
.pwr-cap { color: #3A3630; }

/* Meter cells */
.gpu-meter-cell {
  display: flex; align-items: center; gap: 5px;
  flex: none;
}
.meter-label { font-size: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #4A4640; width: 26px; flex: none; }
.meter-track { width: 100px; height: 8px; background: #2A2622; border-radius: 4px; overflow: hidden; flex: none; }
.meter-track.sm { width: 50px; }
.meter-fill { height: 100%; border-radius: 4px; transition: width 1s; }
.meter-val { width: 42px; font-size: 11px; font-weight: 600; flex: none; text-align: right; }
.mem-cell .meter-val { width: 70px; }
.mem-total { color: #3A3630; font-weight: 400; }
.bw-cell .meter-val { width: 30px; }

.gpu-owner {
  margin-left: auto; font-size: 10px; color: #5A5550;
  flex: none; width: 130px; text-align: right;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.gpu-owner.owned { color: #E8590C; font-weight: 600; }

/* ── System strip ── */
.sys-strip {
  display: flex; gap: 16px; margin-top: 8px;
  padding-top: 8px; border-top: 1px solid #2A2622;
  font-size: 10px;
}
.sys-item { display: flex; align-items: center; gap: 6px; }
.sys-label { font-size: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #4A4640; }
.sys-val { font-size: 11px; font-weight: 600; }
.sys-bar { width: 50px; height: 4px; background: #2A2622; border-radius: 2px; overflow: hidden; }
.sys-fill { height: 100%; border-radius: 2px; }
.sys-fill.ok { background: #2E8B57; }
.sys-fill.warn { background: #84661F; }
.sys-fill.crit { background: #D4483B; }
.sys-note { font-size: 9px; color: #4A4640; }
.sys-note.alert { color: #D4483B; }
</style>
</head>
<body>

<div class="dock">
  <div class="dock-bar">
    <span class="dock-title">Resources</span>
    <span class="num">NVIDIA 550.120 · CUDA 12.4</span>
    <span>Load <b class="num">17.50</b></span>
    <span>CPU <b class="num">18.3%</b></span>
    <span style="margin-left:auto">AVG Util <b class="num">94%</b></span>
    <span>AVG Mem <b class="num">72%</b></span>
    <span class="num" style="color:#4A4640">ecepxiegpu3</span>
  </div>

  <div class="gpu-stack">
    ${gpus.map(gpuRow).join("")}
  </div>

  <div class="sys-strip">
    <div class="sys-item">
      <span class="sys-label">Mem</span>
      <span class="sys-val num">50.7<span style="color:#3A3630">/503 G</span></span>
      <div class="sys-bar"><div class="sys-fill ok" style="width:10%"></div></div>
    </div>
    <div class="sys-item">
      <span class="sys-label">Swap</span>
      <span class="sys-val num" style="color:#D4483B">14.8<span style="color:#3A3630">/18 G</span></span>
      <div class="sys-bar"><div class="sys-fill crit" style="width:82%"></div></div>
      <span class="sys-note alert">82%</span>
    </div>
    <div class="sys-item">
      <span class="sys-label">/data2</span>
      <span class="sys-val num" style="color:#D4483B">19T<span style="color:#3A3630">/20T</span></span>
      <div class="sys-bar"><div class="sys-fill crit" style="width:97%"></div></div>
      <span class="sys-note alert">760G</span>
    </div>
    <div class="sys-item">
      <span class="sys-label">/data5</span>
      <span class="sys-val num" style="color:#84661F">16T<span style="color:#3A3630">/19T</span></span>
      <div class="sys-bar"><div class="sys-fill warn" style="width:92%"></div></div>
      <span class="sys-note">1.5T</span>
    </div>
  </div>
</div>

</body>
</html>`;

const b = await chromium.launch({ headless: true });
const p = await b.newPage({ viewport: { width: 1440, height: 280 } });
await p.setContent(HTML, { waitUntil: "networkidle" });
await p.waitForTimeout(2000);
await p.screenshot({ path: `${S}/resource-compact.png`, fullPage: true });
console.log("Done");
await b.close();
