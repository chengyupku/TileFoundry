"""Self-contained interactive SVG rendering for warpgroup schedules."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from typing import Protocol


class _LaneLike(Protocol):
    @property
    def operations(self) -> Sequence[str]: ...


class _SyncLike(Protocol):
    @property
    def after(self) -> str: ...

    @property
    def before(self) -> str: ...

    @property
    def distance(self) -> int: ...


class _TimedLike(Protocol):
    @property
    def iteration(self) -> int: ...

    @property
    def operation_id(self) -> str: ...

    @property
    def start(self) -> int: ...

    @property
    def issue_end(self) -> int: ...

    @property
    def completion(self) -> int: ...


class _ScheduleLike(Protocol):
    @property
    def format(self) -> str: ...

    @property
    def lanes(self) -> Sequence[_LaneLike]: ...

    @property
    def sync(self) -> Sequence[_SyncLike]: ...

    @property
    def times(self) -> Sequence[_TimedLike]: ...


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root { color-scheme: light; --ink:#17202a; --muted:#66717d; --line:#d8dee5; --paper:#f6f8fa; --panel:#fff; --accent:#1769aa; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:var(--paper); font:14px/1.45 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    header { display:flex; align-items:flex-start; justify-content:space-between; gap:24px; padding:24px 30px 18px; border-bottom:1px solid var(--line); background:var(--panel); }
    .eyebrow { margin:0 0 4px; color:var(--accent); font-size:11px; font-weight:750; letter-spacing:.08em; text-transform:uppercase; }
    h1 { margin:0; font-size:24px; line-height:1.2; letter-spacing:0; }
    .subtitle { margin:6px 0 0; color:var(--muted); }
    .stats { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:8px; }
    .stat { min-width:86px; padding:9px 12px; border:1px solid var(--line); border-radius:6px; background:#fbfcfd; }
    .stat strong { display:block; font-size:17px; line-height:1.1; }
    .stat span { color:var(--muted); font-size:11px; }
    .toolbar { display:flex; align-items:center; flex-wrap:wrap; gap:12px 18px; padding:13px 30px; border-bottom:1px solid var(--line); background:var(--panel); }
    label { color:var(--muted); font-size:12px; font-weight:650; }
    select, input[type="search"] { height:32px; margin-left:7px; padding:0 9px; border:1px solid #c5cdd5; border-radius:5px; color:var(--ink); background:#fff; font:inherit; }
    input[type="search"] { width:210px; }
    input[type="range"] { width:135px; vertical-align:middle; accent-color:var(--accent); }
    input[type="checkbox"] { margin:0 6px 0 0; accent-color:var(--accent); vertical-align:-1px; }
    button { height:32px; padding:0 11px; border:1px solid #b9c4ce; border-radius:5px; color:var(--ink); background:#fff; cursor:pointer; font:650 12px inherit; }
    button:hover { border-color:var(--accent); color:var(--accent); }
    .scale-value { display:inline-block; min-width:150px; color:var(--muted); font-variant-numeric:tabular-nums; }
    .legend { display:flex; align-items:center; flex-wrap:wrap; gap:8px 15px; padding:10px 30px; border-bottom:1px solid var(--line); color:var(--muted); font-size:11px; }
    .legend-item { display:inline-flex; align-items:center; gap:5px; }
    .swatch { width:11px; height:11px; border-radius:3px; }
    .swatch-issue { background:#52606d; }
    .swatch-tail { border:1px dashed #7b8793; background:#d8dee5; opacity:.6; }
    .swatch-sync0 { border:1px dashed #8995a1; background:#fff; }
    .swatch-sync1 { border:1px dashed #c14646; background:#fff; }
    .swatch-iteration-start { border-left:2px solid #1769aa; }
    .swatch-iteration-end { border-left:2px solid #c14646; }
    .canvas { overflow:auto; padding:20px 30px 28px; background:var(--paper); }
    svg { display:block; min-height:190px; }
    .bar { cursor:pointer; }
    .bar:hover, .bar:focus { filter:brightness(.88); stroke:#17202a; stroke-width:1.4; outline:none; }
    .bar-tail { pointer-events:none; }
    .bar-label { pointer-events:none; fill:#17202a; font-size:10px; font-weight:650; }
    .lane-label { fill:#34404b; font-size:12px; font-weight:750; }
    .lane-meta { fill:#87919b; font-size:10px; }
    .axis-label { fill:#66717d; font-size:10px; font-variant-numeric:tabular-nums; }
    .iteration-label { fill:#34404b; font-size:10px; font-weight:750; }
    .iteration-start-label { fill:#1769aa; font-size:9px; font-variant-numeric:tabular-nums; }
    .iteration-end-label { fill:#a83b3b; font-size:9px; font-variant-numeric:tabular-nums; }
    .empty { fill:#66717d; font-size:13px; }
    .details { min-height:76px; padding:12px 30px 16px; border-top:1px solid var(--line); background:var(--panel); color:var(--muted); }
    .details strong { color:var(--ink); }
    .details code { color:#34404b; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
    @media (max-width:720px) {
      header { display:block; padding:18px; }
      .stats { justify-content:flex-start; margin-top:16px; }
      .toolbar, .legend, .canvas, .details { padding-left:18px; padding-right:18px; }
      input[type="search"] { width:min(210px,52vw); }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <p class="eyebrow">TileFoundry / warpgroup schedule</p>
      <h1>__TITLE__</h1>
      <p class="subtitle">Issue occupancy is solid; the lighter tail runs until the result is ready.</p>
    </div>
    <div class="stats" id="stats"></div>
  </header>
  <section class="toolbar" aria-label="Schedule controls">
    <label>Iteration<select id="iteration"></select></label>
    <label>Find<input id="filter" type="search" placeholder="operation id"></label>
    <label><input id="show-sync" type="checkbox" checked>Dependencies</label>
    <label>Zoom<input id="zoom" type="range" min="0.5" max="3" step="0.05" value="1"><span class="scale-value" id="zoom-value"></span></label>
    <button id="reset" type="button">Reset view</button>
  </section>
  <div class="legend">
    <span class="legend-item"><i class="swatch swatch-issue"></i>issue interval</span>
    <span class="legend-item"><i class="swatch swatch-tail"></i>completion tail</span>
    <span class="legend-item"><i class="swatch swatch-sync0"></i>distance-0 sync</span>
    <span class="legend-item"><i class="swatch swatch-sync1"></i>distance-1 sync</span>
    <span class="legend-item"><i class="swatch swatch-iteration-start"></i>iteration start</span>
    <span class="legend-item"><i class="swatch swatch-iteration-end"></i>iteration end</span>
  </div>
  <main class="canvas"><svg id="chart" role="img" aria-label="Warpgroup schedule timeline"></svg></main>
  <div class="details" id="details">Hover or focus a block to inspect its timing witness.</div>
  <script>
    const DATA = /*__SCHEDULE_DATA__*/;
    const svg = document.getElementById('chart');
    const canvas = document.querySelector('.canvas');
    const iteration = document.getElementById('iteration');
    const filter = document.getElementById('filter');
    const showSync = document.getElementById('show-sync');
    const zoom = document.getElementById('zoom');
    const zoomValue = document.getElementById('zoom-value');
    const stats = document.getElementById('stats');
    const details = document.getElementById('details');
    const NS = 'http://www.w3.org/2000/svg';
    const times = DATA.times.map(row => ({ iteration: row[0], id: row[1], start: row[2], issueEnd: row[3], completion: row[4] }));
    const iterations = [...new Set(times.map(item => item.iteration))].sort((a, b) => a - b);
    const scheduleMakespan = Math.max(0, ...times.map(item => item.completion));
    const byKey = new Map(times.map(item => [`${item.iteration}:${item.id}`, item]));
    const laneOf = new Map(DATA.lanes.flatMap((lane, index) => lane.map(id => [id, index])));
    const laneColor = index => `hsl(${(index * 137.508) % 360} 58% 42%)`;
    const node = (name, attrs = {}, text = null) => {
      const element = document.createElementNS(NS, name);
      for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
      if (text !== null) element.textContent = String(text);
      return element;
    };
    const htmlNode = (name, text) => {
      const element = document.createElement(name);
      element.textContent = String(text);
      return element;
    };
    const makeStat = (value, label) => {
      const item = document.createElement('div');
      item.className = 'stat';
      item.append(htmlNode('strong', value), htmlNode('span', label));
      return item;
    };
    const niceStep = value => {
      if (!Number.isFinite(value) || value <= 0) return 1;
      const power = Math.pow(10, Math.floor(Math.log10(value)));
      const normalized = value / power;
      const base = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
      return base * power;
    };
    const formatNumber = value => Number.isInteger(value) ? String(value) : value.toFixed(2);
    const visible = () => {
      const selected = iteration.value;
      const query = filter.value.trim().toLowerCase();
      return times.filter(item => (selected === 'all' || item.iteration === Number(selected)) && (!query || item.id.toLowerCase().includes(query)));
    };
    const showDetails = item => {
      const issue = item.issueEnd - item.start;
      const tail = item.completion - item.issueEnd;
      details.replaceChildren();
      const heading = document.createElement('strong');
      heading.textContent = item.id;
      const code = document.createElement('code');
      code.textContent = `start=${item.start}  issue_end=${item.issueEnd}  completion=${item.completion}  issue=${issue}  tail=${tail}`;
      details.append(heading, document.createTextNode(`  |  iteration ${item.iteration}  |  lane ${laneOf.get(item.id) ?? -1}`), document.createElement('br'), code);
    };
    function render() {
      const items = visible();
      const selected = iteration.value;
      const scaleItems = selected === 'all'
        ? times
        : times.filter(item => item.iteration === Number(selected));
      const visibleStart = scaleItems.length ? Math.min(...scaleItems.map(item => item.start)) : 0;
      const visibleCompletion = scaleItems.length ? Math.max(...scaleItems.map(item => item.completion)) : 1;
      const visibleSpan = Math.max(1, visibleCompletion - visibleStart);
      const left = 145;
      const right = 36;
      const markerIterations = selected === 'all'
        ? iterations
        : iterations.filter(value => value === Number(selected));
      const iterationBounds = markerIterations.map(value => {
        const rows = times.filter(item => item.iteration === value);
        return {
          iteration: value,
          start: Math.min(...rows.map(item => item.start)),
          end: Math.max(...rows.map(item => item.completion)),
        };
      });
      const iterationRowHeight = 25;
      const iterationTrackHeight = Math.max(1, iterationBounds.length) * iterationRowHeight;
      const top = 20 + iterationTrackHeight + 12;
      const laneHeight = 72;
      const axisHeight = 28;
      const viewportWidth = Math.max(720, window.innerWidth - 60);
      const fitPxPerTime = Math.max(0.04, (viewportWidth - left - right) / visibleSpan);
      const pxPerTime = fitPxPerTime * Number(zoom.value);
      const width = Math.max(viewportWidth, left + visibleSpan * pxPerTime + right);
      const height = top + DATA.lanes.length * laneHeight + axisHeight;
      const relative = time => (time - visibleStart) * pxPerTime;
      const iterationMarkers = [];
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      svg.style.width = `${width}px`;
      svg.style.height = `${height}px`;
      svg.replaceChildren();
      zoomValue.textContent = `${Number(zoom.value).toFixed(2)}x fit (${formatNumber(pxPerTime)} px/t)`;
      svg.appendChild(node('rect', { x:0, y:0, width, height, rx:6, fill:'#fff', stroke:'#d8dee5' }));
      svg.appendChild(node('text', { x:10, y:15, class:'lane-meta' }, 'iteration timeline'));
      iterationBounds.forEach((bound, index) => {
        const rowY = 20 + index * iterationRowHeight;
        const startX = left + relative(bound.start);
        const endX = left + relative(bound.end);
        const selectedRow = selected !== 'all' && Number(selected) === bound.iteration;
        const trackFill = selectedRow ? '#e5f0f8' : index % 2 ? '#f5f8fa' : '#eef3f6';
        svg.appendChild(node('text', { x:10, y:rowY + 16, class:'iteration-label' }, `iteration ${bound.iteration}`));
        svg.appendChild(node('rect', { x:startX, y:rowY + 5, width:Math.max(1, endX - startX), height:14, rx:3, fill:trackFill, stroke:'#cdd6de' }));
        iterationMarkers.push({ startX, endX, rowY, selectedRow });
        svg.appendChild(node('text', { x:startX + 3, y:rowY + 16, class:'iteration-start-label' }, `start ${formatNumber(bound.start)}`));
        svg.appendChild(node('text', { x:endX - 3, y:rowY + 16, 'text-anchor':'end', class:'iteration-end-label' }, `end ${formatNumber(bound.end)}`));
      });
      DATA.lanes.forEach((lane, laneIndex) => {
        const y = top + laneIndex * laneHeight;
        svg.appendChild(node('rect', { x:0, y, width, height:laneHeight, fill:laneIndex % 2 ? '#fbfcfd' : '#fff' }));
        svg.appendChild(node('line', { x1:0, y1:y + laneHeight, x2:width, y2:y + laneHeight, stroke:'#d8dee5' }));
        svg.appendChild(node('rect', { x:10, y:y + 20, width:4, height:24, rx:2, fill:laneColor(laneIndex) }));
        svg.appendChild(node('text', { x:22, y:y + 29, class:'lane-label' }, `lane ${laneIndex}`));
        svg.appendChild(node('text', { x:22, y:y + 47, class:'lane-meta' }, `${lane.length} operations`));
      });
      const step = niceStep(visibleSpan / 8);
      const ticks = [];
      for (let tick = 0; tick <= visibleSpan + step / 2; tick += step) ticks.push(Math.min(tick, visibleSpan));
      if (ticks[ticks.length - 1] !== visibleSpan) ticks.push(visibleSpan);
      for (const tick of [...new Set(ticks)]) {
        const x = left + tick * pxPerTime;
        svg.appendChild(node('line', { x1:x, y1:top - 18, x2:x, y2:height - axisHeight, stroke:'#e9edf1' }));
        svg.appendChild(node('text', { x, y:height - 8, 'text-anchor':'middle', class:'axis-label' }, formatNumber(visibleStart + tick)));
      }
      iterationMarkers.forEach(({ startX, endX, rowY, selectedRow }) => {
        svg.appendChild(node('line', { x1:startX, y1:rowY + 1, x2:startX, y2:height - axisHeight, stroke:'#1769aa', 'stroke-width':selectedRow ? 1.5 : 1, opacity:selectedRow ? .9 : .6 }));
        svg.appendChild(node('line', { x1:endX, y1:rowY + 1, x2:endX, y2:height - axisHeight, stroke:'#c14646', 'stroke-width':selectedRow ? 1.5 : 1, opacity:selectedRow ? .9 : .6 }));
      });
      if (showSync.checked) {
        const visibleKeys = new Set(items.map(item => `${item.iteration}:${item.id}`));
        const defs = node('defs');
        const marker = node('marker', { id:'arrow', viewBox:'0 0 8 8', refX:7, refY:4, markerWidth:5, markerHeight:5, orient:'auto-start-reverse' });
        marker.appendChild(node('path', { d:'M 0 0 L 8 4 L 0 8 z', fill:'#8995a1' }));
        defs.appendChild(marker);
        svg.appendChild(defs);
        for (const edge of DATA.sync) {
          for (const sourceIteration of iterations) {
            const targetIteration = sourceIteration + edge.distance;
            const from = byKey.get(`${sourceIteration}:${edge.after}`);
            const to = byKey.get(`${targetIteration}:${edge.before}`);
            if (!from || !to || !visibleKeys.has(`${sourceIteration}:${edge.after}`) || !visibleKeys.has(`${targetIteration}:${edge.before}`)) continue;
            const y1 = top + (laneOf.get(from.id) ?? 0) * laneHeight + laneHeight / 2;
            const y2 = top + (laneOf.get(to.id) ?? 0) * laneHeight + laneHeight / 2;
            const x1 = left + relative(from.completion);
            const x2 = left + relative(to.start);
            const bend = Math.max(12, Math.abs(x2 - x1) * .18);
            svg.appendChild(node('path', { d:`M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`, fill:'none', stroke:edge.distance ? '#c14646' : '#8995a1', 'stroke-width':edge.distance ? 1.2 : .85, 'stroke-dasharray':edge.distance ? '4 3' : '2 3', opacity:edge.distance ? .55 : .35, 'marker-end':'url(#arrow)' }));
          }
        }
      }
      for (const item of items) {
        const laneIndex = laneOf.get(item.id);
        if (!Number.isInteger(laneIndex)) continue;
        const y = top + laneIndex * laneHeight + 16;
        const x = left + relative(item.start);
        const issueWidth = Math.max(2, (item.issueEnd - item.start) * pxPerTime);
        const tailWidth = Math.max(0, (item.completion - item.issueEnd) * pxPerTime);
        const bar = node('rect', { x, y, width:issueWidth, height:30, rx:4, fill:laneColor(laneIndex), class:'bar', tabindex:0 });
        bar.appendChild(node('title', {}, `${item.id} | iteration ${item.iteration} | lane ${laneIndex}; start ${item.start}, issue_end ${item.issueEnd}, completion ${item.completion}`));
        bar.addEventListener('mouseenter', () => showDetails(item));
        bar.addEventListener('focus', () => showDetails(item));
        svg.appendChild(bar);
        if (tailWidth > 0) svg.appendChild(node('rect', { x:x + issueWidth, y, width:tailWidth, height:30, rx:2, fill:laneColor(laneIndex), opacity:.22, class:'bar-tail' }));
        if (issueWidth >= 48) svg.appendChild(node('text', { x:x + 7, y:y + 19, class:'bar-label' }, item.id));
      }
      if (!items.length) svg.appendChild(node('text', { x:left, y:top + 36, class:'empty' }, 'No operations match the current filter.'));
      stats.replaceChildren(
        makeStat(DATA.lanes.length, 'lanes'),
        makeStat(items.length, 'visible operations'),
        makeStat(new Set(items.map(item => item.iteration)).size, 'iterations'),
        makeStat(scheduleMakespan, 'makespan'),
        makeStat(DATA.sync.length, 'sync edges'),
      );
    }
    for (const value of iterations) {
      const option = htmlNode('option', `Iteration ${value}`);
      option.value = String(value);
      iteration.appendChild(option);
    }
    const allIterations = htmlNode('option', 'All iterations');
    allIterations.value = 'all';
    iteration.insertBefore(allIterations, iteration.firstChild);
    iteration.value = 'all';
    [iteration, filter, showSync, zoom].forEach(control => {
      control.addEventListener('input', render);
      control.addEventListener('change', render);
    });
    window.addEventListener('resize', render);
    document.getElementById('reset').addEventListener('click', () => { iteration.value = 'all'; filter.value = ''; showSync.checked = true; zoom.value = '1'; render(); });
    render();
  </script>
</body>
</html>
"""


def _schedule_payload(schedule: _ScheduleLike) -> dict[str, object]:
    """Copy the validated schedule boundary into renderer-owned plain data."""
    lanes = tuple(tuple(str(operation) for operation in lane.operations) for lane in schedule.lanes)
    sync = tuple(
        {"after": edge.after, "before": edge.before, "distance": edge.distance}
        for edge in schedule.sync
    )
    times: list[list[int | str]] = []
    for timed in schedule.times:
        times.append(
            [timed.iteration, timed.operation_id, timed.start, timed.issue_end, timed.completion]
        )
    format_value = str(schedule.format)
    if format_value != "tilefoundry.warpgroup_schedule":
        raise ValueError(f"unsupported schedule format {format_value!r}")
    return {"format": format_value, "lanes": lanes, "sync": sync, "times": tuple(times)}


def render_warpgroup_schedule_html(
    schedule: _ScheduleLike, *, title: str = "Warpgroup schedule"
) -> str:
    """Render a validated warpgroup schedule as deterministic standalone HTML."""
    payload = json.dumps(_schedule_payload(schedule), ensure_ascii=True, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")
    safe_title = html.escape(title, quote=True)
    return _HTML_TEMPLATE.replace("__TITLE__", safe_title).replace("/*__SCHEDULE_DATA__*/", payload)


__all__ = ["render_warpgroup_schedule_html"]
