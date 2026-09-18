"""棒人間ビューアの HTML 雛形。

file:// で開くため、データは fetch せず JSON をそのまま埋め込む
（cap-pitch-analysis/tools/make-viewer.py と同じ方針）。
配色もそちらに合わせ、2投球を色で見分ける。
"""

TEMPLATE = r'''<!doctype html>
<html lang="ja">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>__TITLE__</title>
<style>
  :root {
    color-scheme: dark;
    --ink:   #0E1417;
    --panel: #172026;
    --rule:  #2A3A42;
    --text:  #E8EFF2;
    --muted: #7C949E;
    --a:     #BFE9F2;  /* 投球A の骨格 */
    --b:     #FFB23E;  /* 投球B の骨格 */
    --a-vec: #6EE7A0;  /* 投球A のベクトル */
    --b-vec: #C77DFF;  /* 投球B のベクトル */
    --warn:  #FF6B6B;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--ink); color: var(--text);
    font: 400 13px/1.7 -apple-system, BlinkMacSystemFont, "Hiragino Sans", sans-serif;
    font-variant-numeric: tabular-nums;
  }
  header { padding: 12px 20px; border-bottom: 1px solid var(--rule); }
  header h1 { margin: 0; font-size: 15px; font-weight: 600; }
  header .meta { color: var(--muted); font-size: 12px; margin-top: 2px; }

  .controls {
    display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
    padding: 10px 20px; border-bottom: 1px solid var(--rule); background: var(--panel);
  }
  .group { display: flex; align-items: center; gap: 6px; }
  .group > label { color: var(--muted); font-size: 12px; }
  select, button {
    background: var(--ink); color: var(--text); border: 1px solid var(--rule);
    border-radius: 4px; padding: 4px 8px; font: inherit;
  }
  button { cursor: pointer; }
  button:hover { border-color: var(--muted); }
  input[type=range] { width: 110px; accent-color: var(--a); }
  input[type=checkbox] { accent-color: var(--a); }

  main { display: flex; flex-wrap: wrap; gap: 0; align-items: flex-start; }
  .stage { flex: 1 1 560px; min-width: 320px; padding: 14px 20px; }
  .canvases { display: flex; gap: 12px; flex-wrap: wrap; }
  .canvas-box { flex: 1 1 260px; min-width: 240px; }
  .canvas-box h2 {
    margin: 0 0 4px; font-size: 12px; font-weight: 600; letter-spacing: .02em;
  }
  canvas { width: 100%; background: #0A1013; border: 1px solid var(--rule); border-radius: 6px; }

  .timeline { margin-top: 14px; }
  .timeline { padding: 0 28px; }
  .timeline input[type=range] { width: 100%; }
  .ticks { position: relative; height: 16px; margin-top: -2px; }
  .tick { position: absolute; transform: translateX(-50%); font-size: 10px; color: var(--muted); white-space: nowrap; }
  .tick span { display: block; text-align: center; }

  aside { flex: 0 1 380px; min-width: 300px; padding: 14px 20px; border-left: 1px solid var(--rule); }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { padding: 4px 6px; border-bottom: 1px solid var(--rule); text-align: right; }
  th:first-child, td:first-child { text-align: left; color: var(--muted); }
  thead th { color: var(--muted); font-weight: 500; }
  .col-a { color: var(--a); }
  .col-b { color: var(--b); }
  .note { color: var(--muted); font-size: 11px; margin-top: 12px; line-height: 1.8; }
  .note strong { color: var(--text); font-weight: 600; }
  .warn { color: var(--warn); }
</style>

<header>
  <h1>__TITLE__</h1>
  <div class="meta">__META__</div>
</header>

<div class="controls">
  <div class="group">
    <button id="play">▶ 再生</button>
    <button id="prev">◀</button>
    <button id="next">▶</button>
  </div>
  <div class="group">
    <label for="speed">速度</label>
    <select id="speed">
      <option value="0.25">0.25x</option>
      <option value="0.5" selected>0.5x</option>
      <option value="1">1x</option>
    </select>
  </div>
  <div class="group">
    <label for="sync">そろえ方</label>
    <select id="sync">
      <option value="progress">進行率 0〜100%</option>
      <option value="release">リリース基準</option>
      <option value="frame">フレーム番号</option>
    </select>
  </div>
  <div class="group">
    <label for="layout">表示</label>
    <select id="layout">
      <option value="side">並べて</option>
      <option value="overlay">重ねて</option>
    </select>
  </div>
  <div class="group">
    <label for="vector">ベクトル</label>
    <select id="vector">
      <option value="none">なし</option>
      <option value="velocity" selected>速度（2フレーム差）</option>
      <option value="accel">加速度（力の向き）</option>
    </select>
  </div>
  <div class="group">
    <label for="target">対象</label>
    <select id="target"></select>
  </div>
  <div class="group">
    <label for="scale">矢印倍率</label>
    <input type="range" id="scale" min="0.1" max="3" step="0.05" value="1">
  </div>
  <div class="group">
    <label><input type="checkbox" id="trail" checked> 軌跡</label>
  </div>
</div>

<main>
  <div class="stage">
    <div class="canvases" id="canvases"></div>
    <div class="timeline">
      <input type="range" id="cursor" min="0" max="100" step="1" value="0">
      <div class="ticks" id="ticks"></div>
    </div>
  </div>
  <aside>
    <table id="readout">
      <thead><tr><th>項目</th><th class="col-a" id="head-a">A</th><th class="col-b" id="head-b">B</th><th>差</th></tr></thead>
      <tbody></tbody>
    </table>
    <div class="note" id="note"></div>
  </aside>
</main>

<script>
const DATA = __DATA__;

// ---- 座標系 -------------------------------------------------------------
// 埋め込み座標は「身体サイズを1」とした値で、Xは打者方向が正、Yは上が正。
// 原点は基準イベント（足接地）の股関節中点。画面に出すときだけYを反転する。

const PITCHES = DATA.pitches;
const EDGES = DATA.edges;
const COLORS = ['#BFE9F2', '#FFB23E'];
// ベクトルは骨格と別の色にする。骨格と同じ色だと、線なのか矢印なのか見分けにくい。
const VECTOR_COLORS = ['#6EE7A0', '#C77DFF'];
const EVENT_LABELS = {
  foot_contact: '足接地', max_elbow_flexion: '最大屈曲',
  extension_start: '伸展開始', release: 'リリース',
  pitch_start: '開始', pitch_end: '終了',
};

const ui = {
  play: document.getElementById('play'),
  prev: document.getElementById('prev'),
  next: document.getElementById('next'),
  speed: document.getElementById('speed'),
  sync: document.getElementById('sync'),
  layout: document.getElementById('layout'),
  vector: document.getElementById('vector'),
  target: document.getElementById('target'),
  scale: document.getElementById('scale'),
  trail: document.getElementById('trail'),
  cursor: document.getElementById('cursor'),
  ticks: document.getElementById('ticks'),
  canvases: document.getElementById('canvases'),
  readout: document.querySelector('#readout tbody'),
  note: document.getElementById('note'),
};

// 進行率でそろえられない投球があるなら、既定を切り替えておく
if (PITCHES.some(p => !p.normalized)) ui.sync.value = 'release';

// ベクトルの対象になる関節
const JOINT_NAMES = (() => {
  const names = new Set();
  PITCHES.forEach(p => p.frames.forEach(f => Object.keys(f.k).forEach(n => names.add(n))));
  return [...names].sort();
})();
ui.target.innerHTML =
  '<option value="__all__">全関節</option><option value="__arm__" selected>投球腕</option>' +
  JOINT_NAMES.map(n => `<option value="${n}">${n}</option>`).join('');

// ---- 位置の取り出し -----------------------------------------------------
// そろえ方ごとに「カーソル位置 → 各投球のどのコマか」を決める。

function sequence(pitch) {
  return ui.sync.value === 'progress' && pitch.normalized ? pitch.normalized : pitch.frames;
}

function cursorRange() {
  if (ui.sync.value === 'progress') return { min: 0, max: DATA.normalized_samples - 1 };
  if (ui.sync.value === 'release') {
    let before = 0, after = 0;
    PITCHES.forEach(p => {
      const anchor = anchorIndex(p);
      before = Math.max(before, anchor);
      after = Math.max(after, p.frames.length - 1 - anchor);
    });
    return { min: -before, max: after };
  }
  return { min: 0, max: Math.max(...PITCHES.map(p => p.frames.length)) - 1 };
}

function anchorIndex(pitch) {
  const frame = pitch.events.release && pitch.events.release.frame;
  if (frame === null || frame === undefined) return 0;
  const index = pitch.frames.findIndex(f => f.f === frame);
  return index < 0 ? 0 : index;
}

function indexFor(pitch, cursor) {
  if (ui.sync.value === 'progress') return pitch.normalized ? cursor : null;
  if (ui.sync.value === 'release') return anchorIndex(pitch) + cursor;
  return cursor;
}

function frameAt(pitch, cursor) {
  const index = indexFor(pitch, cursor);
  const list = sequence(pitch);
  if (index === null || index < 0 || index >= list.length) return null;
  return list[index];
}

/** 進行率モードでは実フレームが無いので、進行率が最も近い実フレームを探す。 */
function realIndex(pitch, cursor) {
  const index = indexFor(pitch, cursor);
  if (index === null) return null;
  if (ui.sync.value !== 'progress') {
    return index >= 0 && index < pitch.frames.length ? index : null;
  }
  const wanted = pitch.normalized[index].p;
  let best = null, bestGap = Infinity;
  pitch.frames.forEach((f, i) => {
    if (f.p === null) return;
    const gap = Math.abs(f.p - wanted);
    if (gap < bestGap) { bestGap = gap; best = i; }
  });
  return best;
}

// ---- ベクトル -----------------------------------------------------------
// 速度は2フレーム差、加速度は3フレームの2階差分。加速度の向きが力の向きに当たる。
// 質量が分からないので大きさは N ではなく「身体長/秒^2」の相対値。

function vectorAt(pitch, cursor, name) {
  const mode = ui.vector.value;
  if (mode === 'none') return null;

  const list = sequence(pitch);
  const index = indexFor(pitch, cursor);
  if (index === null) return null;

  const point = i => (list[i] && list[i].k[name]) || null;
  const step = ui.sync.value === 'progress' ? 1 : pitch.fps;  // 1% きざみ or 1/fps 秒

  if (mode === 'velocity') {
    const here = point(index), next = point(index + 1);
    if (!here || !next) return null;
    return [(next[0] - here[0]) * step, (next[1] - here[1]) * step];
  }
  const prev = point(index - 1), here = point(index), next = point(index + 1);
  if (!prev || !here || !next) return null;
  return [
    (next[0] - 2 * here[0] + prev[0]) * step * step,
    (next[1] - 2 * here[1] + prev[1]) * step * step,
  ];
}

/** 矢印の基準になる大きさ。全フレーム・全関節の上位10%を代表値にする。 */
const referenceCache = new Map();
function referenceMagnitude() {
  const key = `${ui.vector.value}:${ui.sync.value}`;
  if (referenceCache.has(key)) return referenceCache.get(key);

  const magnitudes = [];
  const range = cursorRange();
  PITCHES.forEach(pitch => {
    for (let cursor = range.min; cursor <= range.max; cursor += 1) {
      JOINT_NAMES.forEach(name => {
        const vector = vectorAt(pitch, cursor, name);
        if (vector) magnitudes.push(Math.hypot(vector[0], vector[1]));
      });
    }
  });
  magnitudes.sort((a, b) => a - b);
  const reference = magnitudes.length
    ? magnitudes[Math.floor(magnitudes.length * 0.9)] || magnitudes[magnitudes.length - 1]
    : 1;
  referenceCache.set(key, reference || 1);
  return reference || 1;
}

function targetNames(pitch) {
  const choice = ui.target.value;
  if (choice === '__all__') return JOINT_NAMES;
  if (choice === '__arm__') {
    const side = pitch.throwing_side;
    return [`${side}_shoulder`, `${side}_elbow`, `${side}_wrist`];
  }
  return [choice];
}

// ---- 描画 ---------------------------------------------------------------

const bounds = (() => {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  PITCHES.forEach(p => p.frames.forEach(f => Object.values(f.k).forEach(([x, y]) => {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  })));
  const padX = (maxX - minX) * 0.15 + 0.2, padY = (maxY - minY) * 0.15 + 0.2;
  return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
})();

function makeCanvas(title, color, isHtml = false) {
  const box = document.createElement('div');
  box.className = 'canvas-box';
  const heading = document.createElement('h2');
  if (isHtml) heading.innerHTML = title; else heading.textContent = title;
  heading.style.color = color || 'var(--muted)';
  const canvas = document.createElement('canvas');
  box.append(heading, canvas);
  ui.canvases.append(box);
  return canvas;
}

let canvases = [];
function layoutCanvases() {
  ui.canvases.innerHTML = '';
  if (ui.layout.value === 'overlay') {
    const legend = PITCHES.map((p, i) =>
      `<span style="color:${COLORS[i]}">■ ${p.display_name}</span>` +
      `<span style="color:${VECTOR_COLORS[i]}"> ➜</span>`).join('　');
    canvases = [{ canvas: makeCanvas(legend, null, true), pitches: PITCHES.map((p, i) => i) }];
  } else {
    canvases = PITCHES.map((pitch, index) => ({
      canvas: makeCanvas(
        `<span style="color:${COLORS[index]}">${pitch.display_name}</span>` +
        `<span style="color:${VECTOR_COLORS[index]}"> ➜ ベクトル</span>`,
        null,
        true,
      ),
      pitches: [index],
    }));
  }
  resize();
}

// 人は縦長なので、素直に縦横比を取ると画面から溢れる。高さで頭打ちにして、
// 余った幅は projector 側の余白で吸収する。
const MAX_CANVAS_HEIGHT = 520;

function resize() {
  canvases.forEach(({ canvas }) => {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 300;
    const aspect = (bounds.maxY - bounds.minY) / (bounds.maxX - bounds.minX);
    const height = Math.min(width * aspect, MAX_CANVAS_HEIGHT);
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.height = `${height}px`;
  });
  draw();
}

function projector(canvas) {
  const scaleX = canvas.width / (bounds.maxX - bounds.minX);
  const scaleY = canvas.height / (bounds.maxY - bounds.minY);
  const scale = Math.min(scaleX, scaleY);
  const offsetX = (canvas.width - (bounds.maxX - bounds.minX) * scale) / 2;
  const offsetY = (canvas.height - (bounds.maxY - bounds.minY) * scale) / 2;
  return {
    scale,
    to: ([x, y]) => [
      offsetX + (x - bounds.minX) * scale,
      canvas.height - offsetY - (y - bounds.minY) * scale,  // 画面は下が正なので反転
    ],
  };
}

function drawStickFigure(context, project, frame, color, alpha) {
  context.globalAlpha = alpha;
  context.strokeStyle = color;
  context.lineWidth = Math.max(1.5, project.scale * 0.018);
  context.lineCap = 'round';

  EDGES.forEach(([a, b]) => {
    const first = frame.k[a], second = frame.k[b];
    if (!first || !second) return;
    const [x1, y1] = project.to(first), [x2, y2] = project.to(second);
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();
  });

  context.fillStyle = color;
  Object.values(frame.k).forEach(point => {
    const [x, y] = project.to(point);
    context.beginPath();
    context.arc(x, y, Math.max(1.5, project.scale * 0.012), 0, Math.PI * 2);
    context.fill();
  });
  context.globalAlpha = 1;
}

function drawTrail(context, project, pitch, cursor, color) {
  const side = pitch.throwing_side;
  const list = sequence(pitch);
  const index = indexFor(pitch, cursor);
  if (index === null) return;

  context.globalAlpha = 0.45;
  context.strokeStyle = color;
  context.lineWidth = 1;
  [`${side}_wrist`, `${side}_elbow`].forEach(name => {
    context.beginPath();
    let started = false;
    for (let i = 0; i <= Math.min(index, list.length - 1); i += 1) {
      const point = list[i] && list[i].k[name];
      if (!point) { started = false; continue; }
      const [x, y] = project.to(point);
      if (started) context.lineTo(x, y); else { context.moveTo(x, y); started = true; }
    }
    context.stroke();
  });
  context.globalAlpha = 1;
}

// 矢印の最大長（身体サイズ単位）
const MAX_ARROW_LENGTH = 1.2;

function drawArrow(context, from, to, color, clipped = false) {
  const [x1, y1] = from, [x2, y2] = to;
  const length = Math.hypot(x2 - x1, y2 - y1);
  if (length < 2) return;
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = Math.min(10, length * 0.35);

  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.stroke();
  context.beginPath();
  context.moveTo(x2, y2);
  context.lineTo(x2 - head * Math.cos(angle - 0.4), y2 - head * Math.sin(angle - 0.4));
  context.lineTo(x2 - head * Math.cos(angle + 0.4), y2 - head * Math.sin(angle + 0.4));
  context.closePath();
  context.fill();
  if (clipped) {
    context.strokeStyle = '#E8EFF2';
    context.lineWidth = 1;
    context.stroke();
  }
}

function drawVectors(context, project, pitch, cursor, color) {
  if (ui.vector.value === 'none') return;
  const frame = frameAt(pitch, cursor);
  if (!frame) return;
  // 代表値が身体長 0.5 になるようそろえてから、スライダの倍率を掛ける
  const gain = (Number(ui.scale.value) * 0.5) / referenceMagnitude();

  targetNames(pitch).forEach(name => {
    const point = frame.k[name];
    const vector = vectorAt(pitch, cursor, name);
    if (!point || !vector) return;

    // 外れ値1つで画面外まで伸びると他が読めなくなるので、長さを頭打ちにする。
    // 打ち切った矢印は先端を白く縁取って「この長さは実際より短い」と示す。
    let [dx, dy] = [vector[0] * gain, vector[1] * gain];
    const length = Math.hypot(dx, dy);
    const clipped = length > MAX_ARROW_LENGTH;
    if (clipped) {
      dx *= MAX_ARROW_LENGTH / length;
      dy *= MAX_ARROW_LENGTH / length;
    }
    const from = project.to(point);
    const to = project.to([point[0] + dx, point[1] + dy]);
    drawArrow(context, from, to, color, clipped);
  });
}

function draw() {
  const cursor = Number(ui.cursor.value);
  canvases.forEach(({ canvas, pitches }) => {
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    const project = projector(canvas);

    // 原点（股関節中点）の高さに床線を引くと、上下の比較がしやすい
    context.strokeStyle = '#2A3A42';
    context.lineWidth = 1;
    const [, baseline] = project.to([0, 0]);
    context.beginPath();
    context.moveTo(0, baseline);
    context.lineTo(canvas.width, baseline);
    context.stroke();

    pitches.forEach(index => {
      const pitch = PITCHES[index];
      const color = COLORS[index];
      const frame = frameAt(pitch, cursor);
      if (!frame) return;
      if (ui.trail.checked) drawTrail(context, project, pitch, cursor, color);
      drawStickFigure(context, project, frame, color, pitches.length > 1 ? 0.85 : 1);
      drawVectors(context, project, pitch, cursor, VECTOR_COLORS[index]);
    });
  });
  updateReadout();
}

// ---- 数値パネル ---------------------------------------------------------

function formatNumber(value, digits = 1) {
  return value === null || value === undefined || Number.isNaN(value) ? '—' : value.toFixed(digits);
}

/** 大きさが桁違いになる量（速度・加速度）は有効数字3桁で出す。 */
function formatMagnitude(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (value === 0) return '0';
  const digits = Math.max(0, 2 - Math.floor(Math.log10(Math.abs(value))));
  return value.toFixed(Math.min(6, digits));
}

function updateReadout() {
  const cursor = Number(ui.cursor.value);
  const rows = [];

  // 1投球のときもB列を空けておく。列がずれると別の投球の値に見えてしまう。
  const row = (head, values, diff = '') => rows.push([head, values[0] ?? '—', values[1] ?? '—', diff]);

  const positions = PITCHES.map(pitch => realIndex(pitch, cursor));
  row('フレーム', PITCHES.map((p, i) => {
    const index = positions[i];
    return index === null ? '—' : String(p.frames[index].f);
  }));
  row('進行率 %', PITCHES.map((p, i) => {
    const index = positions[i];
    return index === null || p.frames[index].p === null ? '—' : formatNumber(p.frames[index].p, 1);
  }));

  Object.entries(DATA.panel_series).forEach(([key, label]) => {
    const values = PITCHES.map((pitch, i) => {
      const index = positions[i];
      const series = pitch.series[key];
      if (index === null || !series) return null;
      return series[index];
    });
    row(label, values.map(v => formatNumber(v)), diffText(values));
  });

  const unit = ui.sync.value === 'progress' ? '/1%' : '/秒';
  const power = ui.vector.value === 'accel' ? 2 : 1;
  if (ui.vector.value !== 'none') {
    const magnitudes = PITCHES.map(pitch => {
      const names = targetNames(pitch);
      const vectors = names.map(name => vectorAt(pitch, cursor, name)).filter(Boolean);
      if (!vectors.length) return null;
      return Math.max(...vectors.map(([x, y]) => Math.hypot(x, y)));
    });
    const suffix = power === 2 ? `身体長${unit}²` : `身体長${unit}`;
    row(`最大 ${ui.vector.value === 'accel' ? '加速度' : '速度'}（${suffix}）`,
        magnitudes.map(formatMagnitude), diffText(magnitudes, formatMagnitude));
  }

  ui.readout.innerHTML = rows.map(([head, a, b, diff]) =>
    `<tr><td>${head}</td><td class="col-a">${a}</td><td class="col-b">${b}</td><td>${diff}</td></tr>`
  ).join('');
}

function diffText(values, format = value => value.toFixed(1)) {
  if (values.length < 2 || values[0] === null || values[1] === null) return '';
  const difference = values[0] - values[1];
  return (difference > 0 ? '+' : '') + format(difference);
}

// ---- タイムライン -------------------------------------------------------

function refreshCursorRange() {
  const range = cursorRange();
  ui.cursor.min = range.min;
  ui.cursor.max = range.max;
  if (Number(ui.cursor.value) < range.min) ui.cursor.value = range.min;
  if (Number(ui.cursor.value) > range.max) ui.cursor.value = range.max;
  renderTicks(range);
}

function renderTicks(range) {
  const span = range.max - range.min || 1;
  const marks = [];
  PITCHES.forEach((pitch, index) => {
    Object.entries(pitch.events).forEach(([name, event]) => {
      if (event.frame === null || event.frame === undefined) return;
      if (name === 'pitch_start' || name === 'pitch_end') return;
      const position = cursorOfFrame(pitch, event.frame);
      if (position === null) return;
      marks.push({
        left: ((position - range.min) / span) * 100,
        text: `${EVENT_LABELS[name] || name}${PITCHES.length > 1 ? (index === 0 ? ' A' : ' B') : ''}`,
        color: COLORS[index],
      });
    });
  });
  ui.ticks.innerHTML = marks.map(mark =>
    `<div class="tick" style="left:${mark.left}%;color:${mark.color}"><span>│</span><span>${mark.text}</span></div>`
  ).join('');
}

function cursorOfFrame(pitch, frame) {
  const index = pitch.frames.findIndex(f => f.f === frame);
  if (index < 0) return null;
  if (ui.sync.value === 'release') return index - anchorIndex(pitch);
  if (ui.sync.value === 'frame') return index;
  const progress = pitch.frames[index].p;
  if (progress === null || !pitch.normalized) return null;
  return Math.round((progress / 100) * (DATA.normalized_samples - 1));
}

// ---- 再生 ---------------------------------------------------------------

let timer = null;
function togglePlay() {
  if (timer) { stop(); return; }
  const fps = PITCHES[0].fps * Number(ui.speed.value);
  const step = ui.sync.value === 'progress' ? 1 : 1;
  timer = setInterval(() => {
    const range = cursorRange();
    let next = Number(ui.cursor.value) + step;
    if (next > range.max) next = range.min;
    ui.cursor.value = next;
    draw();
  }, Math.max(20, 1000 / fps));
  ui.play.textContent = '■ 停止';
}

function stop() {
  clearInterval(timer);
  timer = null;
  ui.play.textContent = '▶ 再生';
}

function nudge(delta) {
  stop();
  const range = cursorRange();
  ui.cursor.value = Math.min(range.max, Math.max(range.min, Number(ui.cursor.value) + delta));
  draw();
}

// ---- 配線 ---------------------------------------------------------------

ui.play.addEventListener('click', togglePlay);
ui.prev.addEventListener('click', () => nudge(-1));
ui.next.addEventListener('click', () => nudge(1));
ui.cursor.addEventListener('input', () => { stop(); draw(); });
ui.sync.addEventListener('change', () => { stop(); refreshCursorRange(); draw(); });
ui.layout.addEventListener('change', layoutCanvases);
[ui.vector, ui.target, ui.scale, ui.trail].forEach(element =>
  element.addEventListener('input', draw));
window.addEventListener('resize', resize);
document.addEventListener('keydown', event => {
  if (event.key === 'ArrowLeft') nudge(-1);
  if (event.key === 'ArrowRight') nudge(1);
  if (event.key === ' ') { event.preventDefault(); togglePlay(); }
});

document.getElementById('head-a').textContent = PITCHES[0].pitch_id;
document.getElementById('head-b').textContent = PITCHES[1] ? PITCHES[1].pitch_id : '—';

ui.note.innerHTML = `
  <strong>読み方</strong><br>
  座標は身体サイズ（${PITCHES[0].scale_mode}）を1とした相対値。原点は足接地時の股関節中点。
  Xは打者方向が正、Yは上が正。<br>
  <span style="color:${VECTOR_COLORS[0]}">➜ 矢印</span>は骨格と別の色で描く。
  <span class="warn">矢印は力そのものではない。</span>
  速度は連続2フレームの変位、加速度はその変化（3フレームの2階差分）で、
  向きが力の向きに相当する。質量が不明なので大きさはニュートンではなく相対値。<br>
  2D画像から得た見かけの動きなので、奥行き方向の成分は含まれない。<br>
  1球対1球の差は観測された差であり、因果関係ではない。
`;

layoutCanvases();
refreshCursorRange();
draw();
</script>
</html>
'''
