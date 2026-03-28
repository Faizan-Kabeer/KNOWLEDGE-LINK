import { apiFetch } from '../core/api.js';
import { dom } from '../core/dom.js';
import { state } from '../core/state.js';
import { debounce, truncate } from '../core/utils.js';
import { loadGraph } from './graph.js';
import { closeDrawer } from './ui.js';

export async function doEntitySearch(q) {
  if (!q.trim()) { dom.searchResults.innerHTML = ''; return; }
  try {
    const results = await apiFetch(`/entities/search?q=${encodeURIComponent(q)}&topk=12`);
    renderSearchResults(results);
  } catch {}
}

export function renderSearchResults(results) {
  dom.searchResults.innerHTML = '';
  results.forEach(r => {
    const li = document.createElement('li');
    li.textContent = r.name;
    li.title = r.name;
    li.addEventListener('click', () => selectEntity(r));
    dom.searchResults.appendChild(li);
  });
}

export async function selectEntity(entity) {
  state.selectedEntity   = entity;
  state.selectedRelation = null;
  state.predictions      = [];
  state.explainData      = null;

  dom.searchInput.value      = entity.name;

  // Update UI chips
  dom.chipHead.textContent = '⬡ ' + entity.name;
  dom.chipRel.textContent  = 'relation: —';
  dom.selectionDisp.classList.remove('hidden');
  dom.relationSection.classList.remove('hidden');
  dom.predictBtn.disabled = true;

  // Clear the search list to hide dropdown
  dom.searchResults.innerHTML = '';

  // Clear predictions + explain; close drawer
  dom.predictList.innerHTML  = '';
  dom.predictList.classList.add('hidden');
  dom.predictPlaceh.classList.remove('hidden');
  dom.explainContent.classList.add('hidden');
  dom.explainPlaceh.classList.remove('hidden');
  dom.explainPlaceh.textContent = 'Fetching explanation…';
  closeDrawer();

  // Do not prepopulate relationList until user clicks or types
  dom.relationList.innerHTML = '';
  dom.graphTitle.textContent = `Neighborhood: ${truncate(entity.name, 30)}`;

  // Load graph
  await loadGraph(entity.id);
}

export function renderRelationList(filter) {
  const f = filter.toLowerCase();
  
  let filtered;
  if (f) {
    filtered = state.allRelations.filter(r => r.name.toLowerCase().includes(f));
    filtered.sort((a, b) => {
      const aStarts = a.name.toLowerCase().startsWith(f);
      const bStarts = b.name.toLowerCase().startsWith(f);
      if (aStarts && !bStarts) return -1;
      if (!aStarts && bStarts) return 1;
      return a.name.localeCompare(b.name);
    });
  } else {
    filtered = state.allRelations;
  }

  dom.relationList.innerHTML = '';
  filtered.slice(0, 60).forEach(r => {
    const li = document.createElement('li');
    li.textContent = r.name;
    li.title = r.name;
    li.addEventListener('click', () => selectRelation(r));
    dom.relationList.appendChild(li);
  });
}

export function selectRelation(rel) {
  state.selectedRelation = rel;
  dom.relationSearch.value   = rel.name;
  dom.chipRel.textContent    = '⟶ ' + rel.name;
  dom.predictBtn.disabled    = false;

  // Clear the relation list to hide dropdown
  dom.relationList.innerHTML = '';
}

export function initSearchListeners() {
  dom.searchInput.addEventListener('input',
    debounce(e => doEntitySearch(e.target.value), 300)
  );

  dom.searchInput.addEventListener('focus', e => {
    if (e.target.value.trim() && !dom.searchResults.innerHTML) {
      doEntitySearch(e.target.value);
    }
  });

  dom.relationSearch.addEventListener('input', e =>
    renderRelationList(e.target.value)
  );

  dom.relationSearch.addEventListener('focus', e => {
    if (state.selectedEntity) {
      renderRelationList(e.target.value);
    }
  });
}
