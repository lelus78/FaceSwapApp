import * as api from './api.js';
import { state, dom } from './state.js';
import { drawFaceBoxes, updateSelectionHighlights } from './facebox.js';

/**
 * Visualizza un'immagine (da File o Blob) in un elemento <img>.
 * Esegue un callback solo dopo che l'immagine è stata completamente caricata nel browser.
 * @param {File|Blob} blobOrFile - Il file immagine da visualizzare.
 * @param {HTMLImageElement} imageElement - L'elemento <img> dove visualizzare l'immagine.
 * @param {Function} [callback=null] - La funzione da eseguire dopo il caricamento.
 */
export function displayImage(blobOrFile, imageElement, callback = null) {
    if (!imageElement || !blobOrFile) return;

    // Pulisce il listener 'onload' precedente per evitare esecuzioni multiple
    const newImageElement = imageElement.cloneNode(true);
    imageElement.parentNode.replaceChild(newImageElement, imageElement);
    
    // Assegna il callback all'evento 'onload' del nuovo elemento
    newImageElement.onload = callback;
    
    // Crea e assegna l'URL per l'immagine
    const oldUrl = newImageElement.src;
    if (oldUrl && oldUrl.startsWith('blob:')) {
        URL.revokeObjectURL(oldUrl);
    }
    newImageElement.src = URL.createObjectURL(blobOrFile);
    newImageElement.classList.remove('hidden');

    // Aggiorna il riferimento nel nostro oggetto DOM globale
    dom[newImageElement.id] = newImageElement;

    // Logica UI specifica per diversi elementi immagine
    if (newImageElement.id === 'imagePreview') {
        dom.subjectUploadPrompt.style.display = 'none';
    } else if (newImageElement.id === 'source-img-preview') {
        dom.sourceUploadPrompt.style.opacity = '0';
    } else if (newImageElement.id === 'result-image-display') {
        dom.resultPlaceholder.classList.add('hidden');
    }
}


/**
 * Chiama l'API per rilevare i volti e poi invoca la funzione per disegnarli.
 * @param {File|Blob} blob - L'immagine su cui rilevare i volti.
 * @param {HTMLImageElement} imageElement - L'elemento <img> di riferimento.
 * @param {HTMLElement} boxesContainer - Il contenitore dove disegnare i box.
 * @param {Array} faceArray - L'array di stato (es. state.sourceFaces) da aggiornare.
 * @param {string} type - 'source' o 'target'.
 */
async function detectFacesAndDraw(blob, imageElement, boxesContainer, faceArray, type) {
    try {
        const data = await api.detectFaces(blob);
        faceArray.length = 0; // Pulisce l'array
        faceArray.push(...data.faces); // Riempie con i nuovi dati
        drawFaceBoxes(boxesContainer, imageElement, faceArray, type);
    } catch (err) {
        console.error(`Errore durante il rilevamento dei volti (${type}):`, err);
    }
}


/**
 * Gestisce la selezione e deselezione di un volto.
 * @param {number} index - L'indice del box cliccato.
 * @param {string} type - 'source' o 'target'.
 */
export function handleFaceSelection(index, type) {
    if (type === 'source') {
        // Se clicco sullo stesso box, lo deseleziono (-1), altrimenti seleziono il nuovo.
        state.selectedSourceIndex = (state.selectedSourceIndex === index) ? -1 : index;
        updateSelectionHighlights(dom.sourceFaceBoxesContainer, state.selectedSourceIndex);
    } else {
        state.selectedTargetIndex = (state.selectedTargetIndex === index) ? -1 : index;
        updateSelectionHighlights(dom.targetFaceBoxesContainer, state.selectedTargetIndex);
    }
    
    // Aggiorna il testo di stato
    dom.selectedSourceId.textContent = state.selectedSourceIndex > -1 ? state.selectedSourceIndex + 1 : 'Nessuno';
    dom.selectedTargetId.textContent = state.selectedTargetIndex > -1 ? state.selectedTargetIndex + 1 : 'Nessuno';
    
    // Abilita/disabilita il pulsante di swap
    dom.swapBtn.disabled = state.selectedSourceIndex < 0 || state.selectedTargetIndex < 0;
}


/**
 * Gestisce il caricamento del file IMMAGINE SOGGETTO (Step 1).
 * @param {File} file - Il file selezionato dall'utente.
 */
function handleSubjectFile(file) {
    if (!file) return;
    state.subjectFile = file;
    // Mostra solo l'anteprima. Il rilevamento volti avverrà dopo.
    displayImage(file, dom.imagePreview);
    dom.prepareSubjectBtn.disabled = false;
    dom.skipToSwapBtn.disabled = false;
}

/**
 * Gestisce il caricamento del file IMMAGINE SORGENTE (Step 4).
 * @param {File} file - Il file selezionato dall'utente.
 */
function handleSourceFile(file) {
    if (!file) return;
    state.sourceImageFile = file;
    // Mostra l'immagine e, solo al termine, rileva i volti.
    displayImage(file, dom.sourceImgPreview, () => {
        detectFacesAndDraw(file, dom.sourceImgPreview, dom.sourceFaceBoxesContainer, state.sourceFaces, 'source');
    });
}


/**
 * Imposta tutti gli event listener principali dell'applicazione.
 */
export function setupEventListeners() {
    // Listener per il caricamento dell'immagine iniziale
    dom.fileInput.addEventListener('change', (e) => handleSubjectFile(e.target.files[0]));

    // Listener per il caricamento dell'immagine sorgente nello step 4
    dom.sourceImgInput.addEventListener('change', (e) => handleSourceFile(e.target.files[0]));
    
    // Aggiungi qui gli altri listener per i pulsanti del workflow
    // Esempio: dom.prepareSubjectBtn.addEventListener('click', handlePrepareSubject);
    // Esempio: dom.swapBtn.addEventListener('click', handlePerformSwap);
    // Esempio: dom.resetAllBtn.addEventListener('click', resetWorkflow);

    // Listener per il toggle che mostra/nasconde i box
    dom.toggleFaceBoxes.addEventListener('change', (e) => {
        const display = e.target.checked ? 'block' : 'none';
        if (dom.sourceFaceBoxesContainer) dom.sourceFaceBoxesContainer.style.display = display;
        if (dom.targetFaceBoxesContainer) dom.targetFaceBoxesContainer.style.display = display;
    });
}

/**
 * Resetta l'intero stato dell'applicazione all'inizio.
 */
export function resetWorkflow() {
    // Qui andrà la logica completa per resettare tutti gli stati e l'UI
    console.log("Workflow resettato.");
    // Esempio parziale:
    state.selectedSourceIndex = -1;
    state.selectedTargetIndex = -1;
    if (dom.sourceFaceBoxesContainer) dom.sourceFaceBoxesContainer.innerHTML = '';
    if (dom.targetFaceBoxesContainer) dom.targetFaceBoxesContainer.innerHTML = '';
    if (dom.swapBtn) dom.swapBtn.disabled = true;
    if (dom.selectedSourceId) dom.selectedSourceId.textContent = 'Nessuno';
    if (dom.selectedTargetId) dom.selectedTargetId.textContent = 'Nessuno';
}