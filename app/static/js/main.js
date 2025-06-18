import { assignDomElements, dom } from './state.js';
import { initTheme } from './theme.js';
import { setupEventListeners, resetWorkflow, closeModal, loadAvailableModels } from './workflow.js';
import { initFaceBoxObservers } from './facebox.js';
import { loadStickers } from './stickers.js';
import { animationLoop } from './memeEditor.js';
import { loadGallery, setupGalleryInteraction, initSidebarToggle } from './gallery.js';

document.addEventListener('DOMContentLoaded', () => {
  assignDomElements();
  initFaceBoxObservers();
  loadAvailableModels();
  initTheme(dom.themeToggle, dom.themeIcon);
  initSidebarToggle(dom.sidebar, dom.sidebarToggle);
  setupEventListeners();
  loadStickers();
  if (dom.galleryContainer) {
    loadGallery(dom.galleryContainer).then(() => setupGalleryInteraction(dom.galleryContainer));
  }
  resetWorkflow();
  animationLoop();
  window.closeModal = closeModal;
});
