import { apiFetch } from '../core/api.js';
import { dom } from '../core/dom.js';
import { state } from '../core/state.js';
import { setStatus, nodeColor, nodeGlowColor, nodeRadius, truncate } from '../core/utils.js';

const svg = d3.select('#graph-svg');
let svgWidth = 0, svgHeight = 0;
const g = svg.append('g');              // zoomable container

// Zoom behaviour
const zoom = d3.zoom()
  .scaleExtent([0.2, 4])
  .on('zoom', e => g.attr('transform', e.transform));
svg.call(zoom);

// Force simulation
export const simulation = d3.forceSimulation()
  .force('link',   d3.forceLink().id(d => d.id).distance(90).strength(0.4))
  .force('charge', d3.forceManyBody().strength(-220))
  .force('center', d3.forceCenter())
  .force('collide',d3.forceCollide(28));

const linkLayer = g.append('g').attr('class', 'links');
const nodeLayer = g.append('g').attr('class', 'nodes');

export function clearGraph() {
  linkLayer.selectAll('*').remove();
  nodeLayer.selectAll('*').remove();
  simulation.nodes([]);
  simulation.force('link').links([]);
  simulation.stop();
}

export function initGraphResize() {
  window.addEventListener('resize', () => {
    if (state.graphData.nodes.length) {
      const r = document.getElementById('graph-svg').getBoundingClientRect();
      svgWidth  = r.width;
      svgHeight = r.height;
      simulation.force('center').x(svgWidth / 2).y(svgHeight / 2);
      simulation.alpha(0.1).restart();
    }
  });
}

export async function loadGraph(entityId) {
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

export function renderGraph(data, predictedLinks = [], explainedIds = new Set()) {
  const svgEl = document.getElementById('graph-svg');
  if (!svgEl) return;
  
  // Resize to current SVG dimensions or fallback
  const rect = svgEl.getBoundingClientRect();
  svgWidth  = rect.width  || 800;
  svgHeight = rect.height || 600;
  
  // Tag node types
  const nodeMap = new Map(data.nodes.map(n => {
    if (n.id === state.selectedEntity?.id) n.type = 'selected';
    return [n.id, n];
  }));

  if (explainedIds.size > 0) {
    data.nodes.forEach(n => {
      if (explainedIds.has(n.id)) n.type = 'influential';
    });
  }
  predictedLinks.forEach(p => {
    if (!nodeMap.has(p.entity_id)) {
      nodeMap.set(p.entity_id, { id: p.entity_id, name: p.entity_name, type: 'predicted' });
    } else if (nodeMap.get(p.entity_id).type !== 'selected') {
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

  // Links
  const link = linkLayer.selectAll('line')
    .data(allLinks)
    .join('line')
    .attr('class', d => `link-${d.type || 'known'}`)
    .style('stroke-width', d => (d.type === 'predicted' ? 2.5 : 1.5));

  // Nodes
  const node = nodeLayer.selectAll('g.node')
    .data(allNodes, d => d.id)
    .join(
      enter => {
        const g = enter.append('g').attr('class', 'node');
        g.append('circle');
        g.append('text');
        return g;
      }
    )
    .call(
      d3.drag()
        .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end',   (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
    );

  // Update circles
  node.select('circle')
    .attr('r',    d => nodeRadius(d.type))
    .attr('fill', d => nodeColor(d.type))
    .style('filter', d => `drop-shadow(0 0 6px ${nodeGlowColor(d.type)})`)
    .on('mouseover', (e, d) => {
      dom.tooltip.classList.remove('hidden');
      dom.tooltip.textContent = d.name;
    })
    .on('mousemove', e => {
      const r = svgEl.getBoundingClientRect();
      dom.tooltip.style.left = (e.clientX - r.left + 14) + 'px';
      dom.tooltip.style.top  = (e.clientY - r.top  - 10) + 'px';
    })
    .on('mouseleave', () => dom.tooltip.classList.add('hidden'));

  // Update labels
  node.select('text')
    .attr('dy', d => nodeRadius(d.type) + 12)
    .text(d => truncate(d.name, 18))
    .style('opacity', d => d.type === 'selected' ? 1 : 0.6);

  // Simulation
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
