import { assignDomElements, dom } from './state.js';
import { setupEventListeners, resetWorkflow, closeModal } from './workflow.js';
import { loadStickers } from './stickers.js';
import { animationLoop } from './memeEditor.js';

document.addEventListener('DOMContentLoaded', () => {
  assignDomElements();
  setupEventListeners();
  loadStickers();
  resetWorkflow();
  animationLoop();
  window.closeModal = closeModal;
});
