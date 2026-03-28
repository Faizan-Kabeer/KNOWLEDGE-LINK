import { apiFetch } from '../core/api.js';
import { dom } from '../core/dom.js';
import { state } from '../core/state.js';
import { setStatus, truncate } from '../core/utils.js';
import { renderGraph } from './graph.js';
import { openDrawer } from './ui.js';

export async function doExplain(prediction) {
  if (!state.selectedEntity || !state.selectedRelation) return;

  // Open drawer immediately with loading state
  dom.explainContent.classList.add('hidden');
  dom.explainPlaceh.classList.remove('hidden');
  dom.explainPlaceh.textContent = 'Fetching explanation…';
  openDrawer();

  setStatus('loading', 'Explaining... (This may take a few seconds)');

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

    // Show content inside drawer
    dom.explainPlaceh.classList.add('hidden');
    dom.explainContent.classList.remove('hidden');

    dom.explainSummary.innerHTML =
      `<strong>${truncate(data.head.name, 28)}</strong> —[${truncate(data.relation, 24)}]→ ` +
      `<strong>${truncate(data.predicted_tail.name, 28)}</strong>`;

    // Render Heatmap
    renderHeatmap(data.attention, data.neighbors);

    // Render layer contribution chart
    renderLayerChart(data.attention);

    // Influence list + histogram
    renderInfluenceList(data.neighbors);
    renderImportanceHistogram(data.neighbors);

    // Highlight explained nodes on graph
    const explainedIds = new Set(data.neighbors.map(n => n.entity_id));
    renderGraph(state.graphData, state.predictions, explainedIds);

    // Scroll to the bottom to reveal freshly loaded content
    setTimeout(() => {
      dom.explorerView.scrollTo({ top: dom.explorerView.scrollHeight, behavior: 'smooth' });
    }, 150);

    setStatus('ok', 'Explanation ready');
  } catch (e) {
    setStatus('warn', e.message);
    console.error(e);
  }
}

export function renderHeatmap(attention, neighbors) {
  const container = document.getElementById('heatmap-matrix');
  if (!container || !attention || !neighbors) return;

  container.innerHTML = '';
  
  const numCols = Math.min(attention.length, 8); // Display up to 8 neighbors
  if (!numCols) return;

  // Add the Y-axis labels
  const yLabels = document.createElement('div');
  yLabels.className = 'hm-y-labels';
  yLabels.innerHTML = `
    <div class="hm-y-label" title="Layer 0: 2-Hop Path Influence">L0</div>
    <div class="hm-y-label" title="Layer 1: 1-Hop Direct Influence">L1</div>
  `;
  container.appendChild(yLabels);

  const colsWrap = document.createElement('div');
  colsWrap.className = 'hm-cols-wrap';
  container.appendChild(colsWrap);

  for (let i = 0; i < numCols; i++) {
    const [a1, a2] = attention[i];
    const neighbor = neighbors[i];

    const col = document.createElement('div');
    col.className = 'hm-col';

    const title = document.createElement('div');
    title.className = 'hm-col-name';
    title.textContent = truncate(neighbor.entity_name, 10);
    title.title = neighbor.entity_name;
    col.appendChild(title);

    // L1 cell
    const cell1 = document.createElement('div');
    cell1.className = 'hm-cell';
    const bg1 = document.createElement('div');
    bg1.className = 'hm-cell-bg' + (a1 > 0.4 ? ' glow' : '');
    bg1.style.opacity = Math.min(1, a1 * 2); // Boost visibility
    const val1 = document.createElement('div');
    val1.className = 'hm-cell-val';
    val1.textContent = `${(a1 * 100).toFixed(0)}%`;
    
    // Add tooltip
    cell1.title = `L0 Contribution: ${(a1 * 100).toFixed(1)}%`;
    
    cell1.appendChild(bg1);
    cell1.appendChild(val1);
    col.appendChild(cell1);

    // L2 cell
    const cell2 = document.createElement('div');
    cell2.className = 'hm-cell';
    const bg2 = document.createElement('div');
    bg2.className = 'hm-cell-bg' + (a2 > 0.4 ? ' glow' : '');
    bg2.style.opacity = Math.min(1, a2 * 2);
    const val2 = document.createElement('div');
    val2.className = 'hm-cell-val';
    val2.textContent = `${(a2 * 100).toFixed(0)}%`;
    
    // Add tooltip
    cell2.title = `L1 Contribution: ${(a2 * 100).toFixed(1)}%`;
    
    cell2.appendChild(bg2);
    cell2.appendChild(val2);
    col.appendChild(cell2);

    colsWrap.appendChild(col);
  }
  
  // Trigger animations
  setTimeout(() => {
    container.querySelectorAll('.hm-cell-bg').forEach(bg => {
      bg.style.opacity = bg.style.opacity; // Force reflow
    });
  }, 10);
}

export function renderLayerChart(attention) {
  const svgEl = document.getElementById('layer-chart');
  if (!svgEl || !attention || !attention.length) return;

  // Compute averages
  const avgL0 = attention.reduce((s, [a]) => s + a, 0) / attention.length;
  const avgL1 = attention.reduce((s, [, b]) => s + b, 0) / attention.length;
  const total = avgL0 + avgL1 || 1;

  const layers = [
    { label: 'L0  (2-hop)', value: avgL0, pct: avgL0 / total },
    { label: 'L1  (1-hop)', value: avgL1, pct: avgL1 / total },
  ];

  // Draw with D3
  const svg    = d3.select('#layer-chart');
  svg.selectAll('*').remove();

  const W      = svgEl.getBoundingClientRect().width || 300;
  const H      = 80;
  const ml     = 80;   // left margin for labels
  const mr     = 52;   // right margin for pct text
  const barH   = 22;
  const gap    = 14;
  const barW   = W - ml - mr;

  const g = svg.append('g').attr('transform', `translate(${ml},${(H - (barH * 2 + gap)) / 2})`);

  layers.forEach(({ label, value, pct }, i) => {
    const y = i * (barH + gap);

    // Background track
    g.append('rect')
      .attr('x', 0).attr('y', y)
      .attr('width', barW).attr('height', barH)
      .attr('rx', 3)
      .attr('fill', 'rgba(255,255,255,0.04)')
      .attr('stroke', 'rgba(255,255,255,0.06)');

    // Foreground fill — animated
    g.append('rect')
      .attr('x', 0).attr('y', y)
      .attr('width', 0).attr('height', barH)
      .attr('rx', 3)
      .attr('fill', i === 0 ? 'var(--neon)' : 'rgba(57,255,20,0.45)')
      .attr('filter', i === 0 ? 'drop-shadow(0 0 6px var(--neon))' : 'none')
      .transition().duration(600).ease(d3.easeCubicOut)
      .attr('width', barW * pct);

    // Label (left)
    g.append('text')
      .attr('x', -8).attr('y', y + barH / 2 + 1)
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'middle')
      .attr('fill', 'rgba(255,255,255,0.5)')
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('font-size', 10)
      .text(label);

    // Percentage (right)
    g.append('text')
      .attr('x', barW + 8).attr('y', y + barH / 2 + 1)
      .attr('dominant-baseline', 'middle')
      .attr('fill', i === 0 ? 'var(--neon)' : 'rgba(57,255,20,0.7)')
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('font-size', 11)
      .attr('font-weight', 700)
      .text(`${(pct * 100).toFixed(1)}%`);
  });
}

export function renderImportanceHistogram(neighbors) {
  const svgEl    = document.getElementById('importance-hist');
  const labelEl  = document.getElementById('hist-label');
  if (!svgEl || !neighbors || !neighbors.length) return;

  const values = neighbors.map(n => n.importance);
  const max    = Math.max(...values, 0.0001);
  const norm   = values.map(v => v / max);   // normalise to 0-1

  // Bin into 5 equal buckets
  const NUM_BINS = 5;
  const bins = Array.from({ length: NUM_BINS }, () => 0);
  norm.forEach(v => {
    const idx = Math.min(Math.floor(v * NUM_BINS), NUM_BINS - 1);
    bins[idx]++;
  });

  // Gini-based concentration label
  const total = values.reduce((a, b) => a + b, 0) || 1;
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const gini = sorted.reduce((acc, v, i) => acc + (2 * (i + 1) - n - 1) * v, 0) / (n * total);
  const label = gini > 0.55 ? 'Focused' : gini > 0.3 ? 'Moderate' : 'Distributed';
  const labelColor = gini > 0.55 ? 'var(--neon)' : gini > 0.3 ? '#FFD700' : '#00D7FF';
  if (labelEl) {
    labelEl.textContent = label;
    labelEl.style.color = labelColor;
    labelEl.style.borderColor = labelColor;
  }

  const svg  = d3.select('#importance-hist');
  svg.selectAll('*').remove();

  const W    = svgEl.getBoundingClientRect().width || 300;
  const H    = 115;
  const mb   = 36;   
  const mt   = 16;   
  const ml   = 32;   
  const mr   = 8;
  const barAreaW = W - ml - mr;
  const barAreaH = H - mb - mt;
  const maxCount = Math.max(...bins, 1);
  const barW  = barAreaW / NUM_BINS;
  const pad   = 3;

  const g = svg.append('g').attr('transform', `translate(${ml},${mt})`);

  // Y-axis title
  svg.append('text')
    .attr('transform', 'rotate(-90)')
    .attr('y', 10)
    .attr('x', 0 - (H / 2))
    .attr('text-anchor', 'middle')
    .attr('fill', 'rgba(255,255,255,0.4)')
    .attr('font-family', 'JetBrains Mono, monospace')
    .attr('font-size', 11)
    .text('Neighbors');

  // X-axis title
  svg.append('text')
    .attr('x', ml + barAreaW / 2)
    .attr('y', H - 4)
    .attr('text-anchor', 'middle')
    .attr('fill', 'rgba(255,255,255,0.4)')
    .attr('font-family', 'JetBrains Mono, monospace')
    .attr('font-size', 11)
    .text('Relative Importance');

  const xLabels = ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%'];

  bins.forEach((count, i) => {
    const barH  = barAreaH * (count / maxCount);
    const x     = i * barW;
    const y     = barAreaH - barH;
    const alpha = 0.3 + 0.7 * (i / (NUM_BINS - 1)); 
    const fill  = i === NUM_BINS - 1 ? 'var(--neon)' : `rgba(57,255,20,${alpha.toFixed(2)})`;

    // Background track
    g.append('rect')
      .attr('x', x + pad).attr('y', 0)
      .attr('width', barW - pad * 2).attr('height', barAreaH)
      .attr('rx', 2)
      .attr('fill', 'rgba(255,255,255,0.03)');

    // Animated bar
    g.append('rect')
      .attr('x', x + pad).attr('y', barAreaH)
      .attr('width', barW - pad * 2).attr('height', 0)
      .attr('rx', 2)
      .attr('fill', fill)
      .attr('filter', i === NUM_BINS - 1 ? 'drop-shadow(0 0 4px var(--neon))' : 'none')
      .transition().duration(550).ease(d3.easeCubicOut)
      .attr('y', y)
      .attr('height', barH || 1);

    // Count label above bar
    if (count > 0) {
      g.append('text')
        .attr('x', x + barW / 2).attr('y', y - 4)
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.65)')
        .attr('font-family', 'JetBrains Mono, monospace')
        .attr('font-size', 11)
        .text(count);
    }

    // X-axis label
    g.append('text')
      .attr('x', x + barW / 2).attr('y', barAreaH + 15)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.35)')
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('font-size', 10)
      .text(xLabels[i]);
  });
}

export function renderInfluenceList(neighbors) {
  dom.influenceList.innerHTML = '';
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
    dom.influenceList.appendChild(li);
  });
}
