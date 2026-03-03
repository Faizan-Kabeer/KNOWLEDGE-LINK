/**
 * KnowledgeLink — app.js
 * D3.js force-directed graph + entity search + predictions + attention heatmap
 */

const API = '';   // same origin; change to 'http://localhost:8000' if serving separately

// ── Global State ──────────────────────────────────────────────────────────
const state = {
  selectedEntity:   null,   // { id, name }
  selectedRelation: null,   // { id, name }
  allRelations:     [],
  graphData:        { nodes: [], links: [] },
  predictions:      [],
  explainData:      null,
  activePredict:    null,   // index in predictions
};

// ── DOM references ────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const searchInput    = $('entity-search');
const searchResults  = $('search-results');
const relationSection = $('relation-section');
const relationSearch = $('relation-search');
const relationList   = $('relation-list');
const predictBtn     = $('predict-btn');
const selectionDisp  = $('selection-display');
const chipHead       = $('chip-head');
const chipRel        = $('chip-rel');
const graphTitle     = $('graph-title');
const predictSection = $('predict-section');
const predictPlaceh  = $('predict-placeholder');
const predictList    = $('predict-list');
const explainPlaceh  = $('explain-placeholder');
const explainContent = $('explain-content');
const explainSummary = $('explain-summary');
const influenceList  = $('influence-list');
const statusDot      = $('status-dot');
const statusText     = $('status-text');
const tooltip        = $('graph-tooltip');

// ── D3 Graph setup ────────────────────────────────────────────────────────
const svg = d3.select('#graph-svg');
let   svgWidth = 0, svgHeight = 0;
const g = svg.append('g');              // zoomable container

// Zoom behaviour
const zoom = d3.zoom()
  .scaleExtent([0.2, 4])
  .on('zoom', e => g.attr('transform', e.transform));
svg.call(zoom);

// Force simulation
let simulation = d3.forceSimulation()
  .force('link',   d3.forceLink().id(d => d.id).distance(90).strength(0.4))
  .force('charge', d3.forceManyBody().strength(-220))
  .force('center', d3.forceCenter())
  .force('collide',d3.forceCollide(28));

// Layer containers (order matters for z-index)
const linkLayer = g.append('g').attr('class', 'links');
const nodeLayer = g.append('g').attr('class', 'nodes');

// ── Utility ───────────────────────────────────────────────────────────────
function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function setStatus(mode, text) {
  statusDot.className  = `status-dot ${mode}`;
  statusText.textContent = text;
}

function nodeColor(type) {
  return {
    center:    'var(--node-center)',
    neighbor:  'var(--node-neighbor)',
    predicted: 'var(--node-predicted)',
    explained: 'var(--node-explained)',
  }[type] || 'var(--node-neighbor)';
}

function nodeRadius(type) {
  return { center: 14, neighbor: 8, predicted: 10, explained: 10 }[type] || 8;
}

function truncate(str, n = 22) {
  return str.length > n ? str.slice(0, n) + '…' : str;
}

async function apiFetch(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

// ── Startup: health check + load relations ────────────────────────────────
async function init() {
  setStatus('loading', 'Connecting…');
  try {
    const health = await apiFetch('/health');
    if (health.model_loaded) {
      setStatus('ok', `Ready · ${health.num_entities.toLocaleString()} entities`);
    } else {
      setStatus('warn', `Data loaded — no model checkpoint found`);
    }
    state.allRelations = await apiFetch('/relations');
  } catch (e) {
    setStatus('idle', 'Backend unreachable');
    console.error(e);
  }
}

// ── Entity Search ─────────────────────────────────────────────────────────
async function doEntitySearch(q) {
  if (!q.trim()) { searchResults.innerHTML = ''; return; }
  try {
    const results = await apiFetch(`/entities/search?q=${encodeURIComponent(q)}&topk=12`);
    renderSearchResults(results);
  } catch {}
}

function renderSearchResults(results) {
  searchResults.innerHTML = '';
  results.forEach(r => {
    const li = document.createElement('li');
    li.textContent = r.name;
    li.title = r.name;
    li.addEventListener('click', () => selectEntity(r));
    searchResults.appendChild(li);
  });
}

async function selectEntity(entity) {
  state.selectedEntity   = entity;
  state.selectedRelation = null;
  state.predictions      = [];
  state.explainData      = null;

  // Update UI chips
  chipHead.textContent = '⬡ ' + entity.name;
  chipRel.textContent  = 'relation: —';
  selectionDisp.classList.remove('hidden');
  relationSection.classList.remove('hidden');
  predictBtn.disabled = true;

  // Highlight in search list
  [...searchResults.querySelectorAll('li')].forEach(li => {
    li.classList.toggle('active', li.textContent === entity.name);
  });

  // Clear predictions + explain
  predictList.innerHTML  = '';
  predictList.classList.add('hidden');
  predictPlaceh.classList.remove('hidden');
  explainContent.classList.add('hidden');
  explainPlaceh.classList.remove('hidden');

  renderRelationList('');
  graphTitle.textContent = `Neighborhood: ${truncate(entity.name, 30)}`;

  // Load graph
  await loadGraph(entity.id);
}

// ── Relation List ─────────────────────────────────────────────────────────
function renderRelationList(filter) {
  const f = filter.toLowerCase();
  const filtered = f
    ? state.allRelations.filter(r => r.name.toLowerCase().includes(f))
    : state.allRelations;

  relationList.innerHTML = '';
  filtered.slice(0, 60).forEach(r => {
    const li = document.createElement('li');
    li.textContent = r.name;
    li.title = r.name;
    li.addEventListener('click', () => selectRelation(r));
    relationList.appendChild(li);
  });
}

function selectRelation(rel) {
  state.selectedRelation = rel;
  chipRel.textContent    = '⟶ ' + rel.name;
  predictBtn.disabled    = false;

  [...relationList.querySelectorAll('li')].forEach(li =>
    li.classList.toggle('active', li.textContent === rel.name)
  );
}

// ── Graph ─────────────────────────────────────────────────────────────────
async function loadGraph(entityId) {
  setStatus('loading', 'Loading graph…');
  try {
    const data = await apiFetch(`/graph/${entityId}?max_neighbors=40`);
    state.graphData = data;
    renderGraph(data);
    setStatus('ok', `Graph: ${data.nodes.length} nodes · ${data.links.length} edges`);
  } catch (e) {
    setStatus('warn', 'Graph load failed');
    console.error(e);
  }
}

function renderGraph(data, predictedLinks = [], explainedIds = new Set()) {
  // Resize to current SVG dimensions
  const rect = document.getElementById('graph-svg').getBoundingClientRect();
  svgWidth  = rect.width;
  svgHeight = rect.height;

  // Tag node types
  const nodeMap = new Map(data.nodes.map(n => [n.id, n]));
  if (explainedIds.size > 0) {
    data.nodes.forEach(n => {
      if (explainedIds.has(n.id)) n.type = 'explained';
    });
  }
  predictedLinks.forEach(p => {
    if (!nodeMap.has(p.entity_id)) {
      nodeMap.set(p.entity_id, { id: p.entity_id, name: p.entity_name, type: 'predicted' });
    } else if (nodeMap.get(p.entity_id).type !== 'center') {
      nodeMap.get(p.entity_id).type = 'predicted';
    }
  });

  const allNodes = [...nodeMap.values()];
  const allLinks = [
    ...data.links,
    ...predictedLinks.map(p => ({
      source: state.selectedEntity.id,
      target: p.entity_id,
      relation: state.selectedRelation ? state.selectedRelation.name : '',
      type: 'predicted',
      score: p.score,
    })),
  ];

  // ── Links ─────────────────────────────────────────────────────────────
  linkLayer.selectAll('*').remove();

  const link = linkLayer.selectAll('line')
    .data(allLinks)
    .join('line')
    .attr('class', d => `link-${d.type || 'known'}`)
    .style('stroke-width', d => {
      if (d.type === 'predicted') return 2;
      return 1.5;
    });

  // ── Link labels (relation name on hover via tooltip) ──────────────────
  // handled via tooltip on nodes

  // ── Nodes ─────────────────────────────────────────────────────────────
  nodeLayer.selectAll('*').remove();

  const node = nodeLayer.selectAll('g.node')
    .data(allNodes, d => d.id)
    .join('g')
    .attr('class', 'node')
    .call(
      d3.drag()
        .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end',   (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
    );

  node.append('circle')
    .attr('r',    d => nodeRadius(d.type))
    .attr('fill', d => nodeColor(d.type))
    .style('filter', d => `drop-shadow(0 0 6px ${nodeColor(d.type)})`)
    .on('mouseover', (e, d) => {
      tooltip.classList.remove('hidden');
      tooltip.textContent = d.name;
    })
    .on('mousemove', e => {
      const rect = document.getElementById('panel-center').getBoundingClientRect();
      tooltip.style.left = (e.clientX - rect.left + 14) + 'px';
      tooltip.style.top  = (e.clientY - rect.top  - 10) + 'px';
    })
    .on('mouseleave', () => tooltip.classList.add('hidden'));

  node.append('text')
    .attr('dy', d => nodeRadius(d.type) + 12)
    .text(d => truncate(d.name, 18))
    .style('opacity', d => d.type === 'center' ? 1 : 0.6);

  // ── Simulation ────────────────────────────────────────────────────────
  simulation.nodes(allNodes);
  simulation.force('link').links(allLinks);
  simulation.force('center').x(svgWidth / 2).y(svgHeight / 2);
  simulation.alpha(0.8).restart();

  simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);

    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  // Auto-fit after a moment
  setTimeout(() => {
    svg.transition().duration(600).call(
      zoom.transform,
      d3.zoomIdentity
        .translate(svgWidth / 2, svgHeight / 2)
        .scale(Math.min(svgWidth, svgHeight) / 500)
        .translate(-svgWidth / 2, -svgHeight / 2)
    );
  }, 400);
}

// ── Predict ───────────────────────────────────────────────────────────────
async function doPrediction() {
  if (!state.selectedEntity || !state.selectedRelation) return;

  setStatus('loading', 'Running prediction…');
  predictBtn.disabled = true;

  try {
    const results = await apiFetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        head_id: state.selectedEntity.id,
        rel_id:  state.selectedRelation.id,
        topk:    10,
      }),
    });

    state.predictions = results;
    state.activePredict = null;

    renderPredictions(results);

    // Overlay predicted nodes/links on graph
    renderGraph(state.graphData, results);

    setStatus('ok', `${results.length} predictions ready`);
  } catch (e) {
    setStatus('warn', e.message);
  } finally {
    predictBtn.disabled = false;
  }
}

function renderPredictions(predictions) {
  predictPlaceh.classList.add('hidden');
  predictList.classList.remove('hidden');
  predictList.innerHTML = '';

  const maxScore = Math.max(...predictions.map(p => p.score), 0.001);

  predictions.forEach((p, i) => {
    const li = document.createElement('li');
    li.className = 'predict-item';
    li.innerHTML = `
      <div class="predict-item-header">
        <span class="predict-name" title="${p.entity_name}">${truncate(p.entity_name, 24)}</span>
        <span class="predict-score">${(p.score * 100).toFixed(1)}%</span>
        ${p.is_known ? '<span class="predict-known-badge">known</span>' : ''}
      </div>
      <div class="predict-bar-wrap">
        <div class="predict-bar" style="width:${(p.score / maxScore * 100).toFixed(1)}%"></div>
      </div>`;

    li.addEventListener('click', () => {
      state.activePredict = i;
      [...predictList.querySelectorAll('.predict-item')].forEach((el, j) =>
        el.classList.toggle('active', j === i)
      );
      doExplain(p);
    });

    predictList.appendChild(li);
  });
}

// ── Explain ───────────────────────────────────────────────────────────────
async function doExplain(prediction) {
  if (!state.selectedEntity || !state.selectedRelation) return;

  setStatus('loading', 'Explaining…');

  try {
    const data = await apiFetch('/explain', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        head_id: state.selectedEntity.id,
        rel_id:  state.selectedRelation.id,
        tail_id: prediction.entity_id,
        max_neighbors: 20,
      }),
    });

    state.explainData = data;

    // Summary
    explainPlaceh.classList.add('hidden');
    explainContent.classList.remove('hidden');

    explainSummary.innerHTML =
      `<strong>${truncate(data.head.name, 28)}</strong> —[${truncate(data.relation, 24)}]→ ` +
      `<strong>${truncate(data.predicted_tail.name, 28)}</strong><br>` +
      `Based on <strong>${data.num_neighbors}</strong> neighbors across ` +
      `<strong>${data.num_layers}</strong> attention layers.`;

    // Heatmap
    renderHeatmap(data);

    // Influence list
    renderInfluenceList(data.neighbors);

    // Highlight explained nodes on graph
    const explainedIds = new Set(data.neighbors.map(n => n.entity_id));
    renderGraph(state.graphData, state.predictions, explainedIds);

    setStatus('ok', 'Explanation ready');
  } catch (e) {
    setStatus('warn', e.message);
    console.error(e);
  }
}

function renderHeatmap(data) {
  const canvas  = $('heatmap-canvas');
  const ctx     = canvas.getContext('2d');
  const labels  = $('heatmap-labels');

  const rows    = data.num_layers;          // y = layers
  const cols    = data.num_neighbors;       // x = neighbors
  if (cols === 0 || rows === 0) return;

  const cellW   = Math.max(18, Math.min(32, Math.floor(240 / cols)));
  const cellH   = 24;

  canvas.width  = cols * cellW;
  canvas.height = rows * cellH;

  // Find global max for normalisation
  let maxVal = 0;
  data.attention_layers.forEach(layer =>
    layer.forEach(v => { if (v > maxVal) maxVal = v; })
  );
  if (maxVal === 0) maxVal = 1;

  // Draw cells
  data.attention_layers.forEach((layer, ly) => {
    layer.forEach((val, col) => {
      const t   = val / maxVal;              // 0–1
      const r   = Math.round(249 * t);
      const g2  = Math.round(115 * t);
      const b   = Math.round(22  * t);
      const a   = 0.15 + 0.85 * t;

      ctx.fillStyle = `rgba(${r},${g2},${b},${a})`;
      ctx.fillRect(col * cellW, ly * cellH, cellW - 1, cellH - 1);

      // Value text
      if (cellW >= 26) {
        ctx.fillStyle = t > 0.5 ? '#fff' : 'rgba(255,255,255,0.4)';
        ctx.font      = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(val.toFixed(2), col * cellW + cellW / 2, ly * cellH + cellH / 2 + 3);
      }
    });
  });

  // Row labels (L0, L1, …) — drawn on left side inside canvas (first column area is fine)
  // We overlay them as CSS labels instead
  labels.innerHTML = '';
  for (let l = 0; l < rows; l++) {
    const s = document.createElement('span');
    s.textContent = `L${l}`;
    s.style.width = canvas.width + 'px';
    s.style.display = 'block';
    s.style.textAlign = 'left';
    s.style.height = cellH + 'px';
    s.style.lineHeight = cellH + 'px';
  }
  // Layer labels on y-axis: overlay text at left of each row
  ctx.save();
  for (let l = 0; l < rows; l++) {
    ctx.fillStyle = 'rgba(148,163,184,0.7)';
    ctx.font = 'bold 9px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`L${l}`, 2, l * cellH + cellH / 2 + 3);
  }
  ctx.restore();
}

function renderInfluenceList(neighbors) {
  influenceList.innerHTML = '';
  const maxImp = Math.max(...neighbors.map(n => n.importance), 0.001);

  neighbors.slice(0, 12).forEach((n, i) => {
    const li = document.createElement('li');
    li.className = 'influence-item';
    li.innerHTML = `
      <span class="influence-rank">#${i + 1}</span>
      <span class="influence-name" title="${n.entity_name}">${truncate(n.entity_name, 22)}</span>
      <div class="influence-bar-wrap">
        <div class="influence-bar" style="width:${(n.importance / maxImp * 100).toFixed(1)}%"></div>
      </div>
      <span class="influence-val">${(n.importance * 100).toFixed(1)}%</span>`;
    influenceList.appendChild(li);
  });
}

// ── Event Bindings ────────────────────────────────────────────────────────
searchInput.addEventListener('input',
  debounce(e => doEntitySearch(e.target.value), 300)
);

relationSearch.addEventListener('input', e =>
  renderRelationList(e.target.value)
);

predictBtn.addEventListener('click', doPrediction);

// Resize: re-centre simulation
window.addEventListener('resize', () => {
  if (state.graphData.nodes.length) {
    const r = document.getElementById('graph-svg').getBoundingClientRect();
    svgWidth  = r.width;
    svgHeight = r.height;
    simulation.force('center').x(svgWidth / 2).y(svgHeight / 2);
    simulation.alpha(0.1).restart();
  }
});

// ── Boot ──────────────────────────────────────────────────────────────────
init();
