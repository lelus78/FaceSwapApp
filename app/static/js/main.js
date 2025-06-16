import { assignDomElements, dom } from './state.js';
import { setupEventListeners, resetWorkflow, closeModal, openModal } from './workflow.js';
import { initFaceBoxObservers } from './facebox.js';
import { initTheme } from './theme.js';

/**
 * Funzione principale eseguita al caricamento del DOM.
 * Inizializza tutti i componenti dell'applicazione nell'ordine corretto.
 */
document.addEventListener('DOMContentLoaded', () => {
  // 1. Raccoglie tutti gli elementi del DOM e li rende disponibili nell'oggetto 'dom'
  assignDomElements();
  
  // 2. Inizializza la logica per il cambio tema (dark/light),
  //    PASSANDO gli elementi corretti che abbiamo raccolto al punto 1.
  initTheme(dom.themeToggle, dom.themeIcon);

  // 3. Imposta tutti gli event listener per i pulsanti, input, etc.
  setupEventListeners();

  // 4. Inizializza gli observer che gestiscono il ridisegno dei box al resize
  initFaceBoxObservers();

  // 5. Esegue un reset iniziale per portare l'app allo stato di partenza
  resetWorkflow();

  window.closeModal = closeModal;
  window.openModal = openModal;
});