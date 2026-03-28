import { apiFetch } from '../core/api.js';
import { dom } from '../core/dom.js';
import { state } from '../core/state.js';
import { setStatus } from '../core/utils.js';
import { renderRelationList, selectEntity } from './search.js';

export function populateModelSelect(models, active) {
  dom.modelSelect.innerHTML = '';
  models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m === 'fb15k-237' ? 'FB15k-237' : m;
    if (m === active) opt.selected = true;
    dom.modelSelect.appendChild(opt);
  });
  // Show delete button only for non-default models
  const selected = dom.modelSelect.value;
  if (selected && selected !== 'fb15k-237') {
    dom.deleteModelBtn.classList.remove('hidden');
  } else {
    dom.deleteModelBtn.classList.add('hidden');
  }
}

export function initModelListeners() {
  dom.modelSelect.addEventListener('change', async (e) => {
    const modelName = e.target.value;
    const displayName = modelName === 'fb15k-237' ? 'FB15k-237' : modelName;

    setStatus('loading', `Switching to ${displayName}…`);

    dom.modelSelect.disabled = true;
    dom.deleteModelBtn.disabled = true;

    if (modelName === 'fb15k-237') {
      dom.deleteModelBtn.classList.add('hidden');
    } else {
      dom.deleteModelBtn.classList.remove('hidden');
    }

    try {
      await apiFetch('/models/switch', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName }),
      });
      
      const health = await apiFetch('/health');
      setStatus('ok', `Switched to ${displayName} · ${health.num_entities.toLocaleString()} entities`);
      state.allRelations = await apiFetch('/relations');
      
      if (state.selectedEntity) {
        selectEntity(state.selectedEntity);
      } else {
        renderRelationList('');
      }
    } catch (err) {
      setStatus('warn', `Failed to switch: ${err.message}`);
    } finally {
      dom.modelSelect.disabled = false;
      dom.deleteModelBtn.disabled = false;
    }
  });

  dom.deleteModelBtn.addEventListener('click', async () => {
    const modelName = dom.modelSelect.value;
    if (!modelName || modelName === 'fb15k-237') return;

    if (!confirm(`Delete model "${modelName}"? This cannot be undone.`)) return;

    dom.modelSelect.disabled = true;
    dom.deleteModelBtn.disabled = true;

    const optToRemove = dom.modelSelect.querySelector(`option[value="${CSS.escape(modelName)}"]`);
    if (optToRemove) optToRemove.remove();

    dom.modelSelect.value = 'fb15k-237';
    setStatus('loading', `Switching to FB15k-237…`);
    
    try {
      const res = await apiFetch(`/models/${encodeURIComponent(modelName)}`, { method: 'DELETE' });

      const activeAfterDelete = res.active_model || 'fb15k-237';
      const displayName = activeAfterDelete === 'fb15k-237' ? 'FB15k-237' : activeAfterDelete;

      populateModelSelect(res.models, activeAfterDelete);

      const health = await apiFetch('/health');
      state.allRelations = await apiFetch('/relations');

      setStatus('ok', `Switched to ${displayName} · ${health.num_entities.toLocaleString()} entities`);
    } catch (err) {
      apiFetch('/models').then(r => populateModelSelect(r.models, r.active_model)).catch(() => {});
      setStatus('warn', `Delete failed: ${err.message}`);
    } finally {
      dom.modelSelect.disabled = false;
      dom.deleteModelBtn.disabled = false;
    }
  });
}
