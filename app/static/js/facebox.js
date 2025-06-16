import { state, dom } from './state.js';
import { handleFaceSelection } from './workflow.js';

/**
 * Disegna i box di selezione sui volti rilevati.
 * Usa getBoundingClientRect per calcolare l'offset tra l'immagine e il suo contenitore.
 * @param {HTMLElement} container - Il contenitore dove disegnare i box.
 * @param {HTMLImageElement} image - L'elemento <img> di riferimento.
 * @param {Array} faces - L'array dei volti rilevati.
 * @param {string} type - 'source' o 'target'.
 */
export function drawFaceBoxes(container, image, faces, type) {
    // Se non abbiamo gli elementi necessari o l'immagine non è caricata, puliamo e usciamo.
    if (!container || !image || !image.complete || image.naturalWidth === 0) {
        if (container) container.innerHTML = '';
        return;
    }
    container.innerHTML = '';

    const imageRect = image.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    // Se l'immagine non è ancora visibile, usciamo per evitare calcoli errati.
    if (imageRect.width === 0) return;

    // Calcoliamo la scala e l'offset basandoci sulla posizione e dimensione reale degli elementi.
    const scaleX = imageRect.width / image.naturalWidth;
    const scaleY = imageRect.height / image.naturalHeight;
    const offsetX = imageRect.left - containerRect.left;
    const offsetY = imageRect.top - containerRect.top;

    faces.forEach((face, idx) => {
        const [x1, y1, x2, y2] = face.bbox;
        const box = document.createElement('div');
        box.className = 'face-box';

        // Applichiamo scala e offset per un posizionamento preciso.
        box.style.left = `${offsetX + (x1 * scaleX)}px`;
        box.style.top = `${offsetY + (y1 * scaleY)}px`;
        box.style.width = `${(x2 - x1) * scaleX}px`;
        box.style.height = `${(y2 - y1) * scaleY}px`;
        box.style.pointerEvents = 'auto'; // Assicura che i box siano sempre cliccabili.

        const label = document.createElement('span');
        label.className = 'face-box-label';
        label.textContent = idx + 1;
        box.appendChild(label);

        box.onclick = (e) => {
            e.stopPropagation();
            handleFaceSelection(idx, type);
        };
        container.appendChild(box);
    });
}

/**
 * Aggiorna l'evidenziazione visiva del box selezionato.
 * @param {HTMLElement} container - Il contenitore dei box.
 * @param {number} selectedIndex - L'indice del box da evidenziare.
 */
export function updateSelectionHighlights(container, selectedIndex) {
    if (!container) return;
    container.querySelectorAll('.face-box').forEach((box, i) => {
        box.classList.toggle('selected', i === selectedIndex);
    });
}

/**
 * Funzione di utility per ridisegnare i box, utile per il resize della finestra.
 */
export function refreshFaceBoxes() {
    if (dom.sourceImgPreview && dom.sourceImgPreview.src) {
        drawFaceBoxes(dom.sourceFaceBoxesContainer, dom.sourceImgPreview, state.sourceFaces, 'source');
    }
    if (dom.resultImageDisplay && dom.resultImageDisplay.src) {
        drawFaceBoxes(dom.targetFaceBoxesContainer, dom.resultImageDisplay, state.targetFaces, 'target');
    }
}

/**
 * Inizializza gli observer che ridisegnano i box quando la finestra o le immagini vengono ridimensionate.
 */
export function initFaceBoxObservers() {
    if (window.ResizeObserver) {
        const ro = new ResizeObserver(refreshFaceBoxes);
        if (dom.resultImageDisplay) ro.observe(dom.resultImageDisplay);
        if (dom.sourceImgPreview) ro.observe(dom.sourceImgPreview);
    }
    // Aggiungiamo un listener anche per il resize della finestra come fallback.
    window.addEventListener('resize', refreshFaceBoxes);
}