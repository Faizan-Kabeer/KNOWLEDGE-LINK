import { dom } from '../core/dom.js';
import { state } from '../core/state.js';
import { setStatus } from '../core/utils.js';
import { clearGraph } from './graph.js';

export function openDrawer() {
  dom.explainBottom.classList.add('open');
  // Scroll only the explorer-view container (not the outer document)
  // so the topbar never moves
  setTimeout(() => {
    dom.explorerView.scrollTo({ top: dom.explorerView.scrollHeight, behavior: 'smooth' });
  }, 420);
}

export function closeDrawer() {
  dom.explainBottom.classList.remove('open');
  dom.explainContent.classList.add('hidden');
  dom.explainPlaceh.classList.remove('hidden');
  dom.explainPlaceh.textContent = 'Fetching explanation…';
  // Pan back to top
  dom.explorerView.scrollTo({ top: 0, behavior: 'smooth' });
}

export function clearAll() {
  // Reset state
  state.selectedEntity   = null;
  state.selectedRelation = null;
  state.predictions      = [];
  state.explainData      = null;
  state.activePredict    = null;
  state.graphData        = { nodes: [], links: [] };

  // Left panel
  dom.searchInput.value        = '';
  dom.searchResults.innerHTML  = '';
  dom.relationSearch.value     = '';
  dom.relationList.innerHTML   = '';
  dom.relationSection.classList.add('hidden');
  dom.selectionDisp.classList.add('hidden');
  dom.chipHead.textContent     = '';
  dom.chipRel.textContent      = '';
  dom.predictBtn.disabled      = true;

  // Right panel
  dom.predictList.innerHTML    = '';
  dom.predictList.classList.add('hidden');
  dom.predictPlaceh.classList.remove('hidden');
  dom.resetBtn.classList.add('hidden');

  // Explainability
  closeDrawer();

  // Graph
  dom.graphTitle.textContent = 'Select an entity to explore';
  clearGraph();

  setStatus('ok', 'Reset — select an entity to start');
}

export function initUiListeners() {
  dom.resetBtn.addEventListener('click', clearAll);
  dom.explainCloseBtn.addEventListener('click', closeDrawer);

  // UI Navigation
  dom.navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      dom.navBtns.forEach(b => b.classList.remove('active'));
      dom.views.forEach(v => v.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.target).classList.add('active');
      
      // Auto-fit graph if returning to explorer
      if (btn.dataset.target === 'explorer-view' && state.graphData.nodes.length) {
        window.dispatchEvent(new Event('resize'));
      }
    });
  });
}
