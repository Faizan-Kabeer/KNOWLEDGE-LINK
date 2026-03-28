import { initUiListeners } from './features/ui.js';
import { initSearchListeners } from './features/search.js';
import { initPredictListeners } from './features/predict.js';
import { initModelListeners, populateModelSelect } from './features/models.js';
import { initTrainingListeners } from './features/training.js';
import { initGraphResize } from './features/graph.js';
import { apiFetch } from './core/api.js';
import { setStatus } from './core/utils.js';
import { state } from './core/state.js';

async function init() {
  console.log("%c KnowledgeLink: Cyber-Minimalist Theme v1.1 Loaded ", "background: #000; color: #39FF14; font-weight: bold;");
  setStatus('loading', 'Connecting…');
  try {
    const health = await apiFetch('/health');
    if (health.model_loaded) {
      setStatus('ok', `Ready · ${health.num_entities.toLocaleString()} entities`);
    } else {
      setStatus('warn', `Data loaded — no model checkpoint found`);
    }
    
    // Load relations and models
    const [rels, modelsData] = await Promise.all([
      apiFetch('/relations'),
      apiFetch('/models')
    ]);
    
    state.allRelations = rels;
    populateModelSelect(modelsData.models, modelsData.active_model);
    
  } catch (e) {
    setStatus('idle', 'Backend unreachable');
    console.error(e);
  }
}

// Attach all listeners
initUiListeners();
initSearchListeners();
initPredictListeners();
initModelListeners();
initTrainingListeners();
initGraphResize();

// Boot the app
init();
