import { dom } from './dom.js';

export function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

export function setStatus(mode, text) {
  dom.statusDot.className  = `status-dot ${mode}`;
  dom.statusText.textContent = text;
}

export function nodeColor(type) {
  return {
    selected:    'url(#grad-selected)',
    neighbor:    'url(#grad-neighbor)',
    predicted:   'url(#grad-predicted)',
    influential: 'url(#grad-influential)',
  }[type] || 'url(#grad-neighbor)';
}

export function nodeGlowColor(type) {
  return {
    selected:    'var(--node-selected)',
    neighbor:    'var(--node-neighbor)',
    predicted:   'var(--node-predicted)',
    influential: 'var(--node-influential)',
  }[type] || 'var(--node-neighbor)';
}

export function nodeRadius(type) {
  return { selected: 14, neighbor: 8, predicted: 10, influential: 10 }[type] || 8;
}

export function truncate(str, n = 22) {
  return str.length > n ? str.slice(0, n) + '…' : str;
}
