import { apiFetch } from '../core/api.js';
import { dom } from '../core/dom.js';
import { state } from '../core/state.js';
import { setStatus, truncate } from '../core/utils.js';
import { renderGraph } from './graph.js';
import { doExplain } from './explain.js';

export async function doPrediction() {
  if (!state.selectedEntity || !state.selectedRelation) return;

  setStatus('loading', 'Running prediction…');
  dom.predictBtn.disabled = true;

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
    dom.predictBtn.disabled = false;
  }
}

export function renderPredictions(predictions) {
  dom.predictPlaceh.classList.add('hidden');
  dom.predictList.classList.remove('hidden');
  dom.predictList.innerHTML = '';
  dom.resetBtn.classList.remove('hidden');

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
      [...dom.predictList.querySelectorAll('.predict-item')].forEach((el, j) =>
        el.classList.toggle('active', j === i)
      );
      doExplain(p);
    });

    dom.predictList.appendChild(li);
  });
}

export function initPredictListeners() {
  dom.predictBtn.addEventListener('click', doPrediction);
}
