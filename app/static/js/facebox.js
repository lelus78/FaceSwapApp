import { state, dom } from './state.js';
import * as api from './api.js';
import { showError } from './workflow.js';

/**
 * LOGICA MATEMATICA CORRETTA - IGNORA COMPLETAMENTE LE DIMENSIONI DELL'ELEMENTO IMG
 * E CALCOLA TUTTO BASANDOSI SULLE PROPORZIONI, RISOLVENDO IL PROBLEMA.
 */
function drawFaceBoxes(container, image, faces) {
  // --- LOG INIZIALE DETTAGLIATO E CONTROLLI ROBUSTI ---
  console.log(`[${container?.id || 'container-sconosciuto'}] drawFaceBoxes CALLED. Image src: ${image?.src?.substring(0,50)}...`);

  if (!container || !image || !image.complete ||
      !image.naturalWidth || image.naturalWidth === 0 ||
      !image.naturalHeight || image.naturalHeight === 0 ||
      !container.offsetWidth || container.offsetWidth === 0 ||
      !container.offsetHeight || container.offsetHeight === 0) {
    console.warn(`[${container?.id || 'container-sconosciuto'}] drawFaceBoxes: Prerequisiti (incluse dimensioni > 0) non soddisfatti.`, {
      containerExists: !!container,
      imageExists: !!image,
      imageComplete: image?.complete,
      imageNaturalWidth: image?.naturalWidth,
      imageNaturalHeight: image?.naturalHeight,
      containerOffsetWidth: container?.offsetWidth,
      containerOffsetHeight: container?.offsetHeight,
    });
    if (container) container.innerHTML = ''; // Pulisce comunque se il container esiste
    return;
  }
  // --- FINE LOG E CONTROLLI ---

  // Ora possiamo leggere le dimensioni con più sicurezza
  const currentContainerWidth = container.offsetWidth;
  const currentContainerHeight = container.offsetHeight;
  const currentNaturalWidth = image.naturalWidth;
  const currentNaturalHeight = image.naturalHeight;

  console.log(`[${container.id}] INPUT DIMS VALIDATED: containerWidth=${currentContainerWidth}, containerHeight=${currentContainerHeight}, naturalWidth=${currentNaturalWidth}, naturalHeight=${currentNaturalHeight}`);

  container.innerHTML = ''; // Pulisce il container da box precedenti e dal rettangolo di debug

  if (!faces || faces.length === 0) {
    console.log(`[${container.id}] drawFaceBoxes: Nessun volto da disegnare.`);
    return;
  }

  // Usa le dimensioni correnti per i calcoli
  const imageRatio = currentNaturalWidth / currentNaturalHeight;
  const containerRatio = currentContainerWidth / currentContainerHeight;

  let finalWidth, finalHeight, scale;

  console.log(`[${container.id}] --- RATIO CHECK START ---`);
  // console.log(`[${container.id}] image.naturalWidth: ${currentNaturalWidth}, image.naturalHeight: ${currentNaturalHeight}`); // Già loggato sopra
  // console.log(`[${container.id}] container.offsetWidth: ${currentContainerWidth}, container.offsetHeight: ${currentContainerHeight}`); // Già loggato sopra
  console.log(`[${container.id}] imageRatio: ${imageRatio}`);
  console.log(`[${container.id}] containerRatio: ${containerRatio}`);
  console.log(`[${container.id}] CONFRONTO: imageRatio (${imageRatio}) > containerRatio (${containerRatio}) ? Risultato: ${imageRatio > containerRatio}`);

  if (imageRatio > containerRatio) {
    console.log(`[${container.id}] [Entered IF Block] - Logica Letterbox`);
    finalWidth = currentContainerWidth;
    finalHeight = currentContainerWidth / imageRatio;
  } else {
    console.log(`[${container.id}] [Entered ELSE Block] - Logica Pillarbox`);
    finalHeight = currentContainerHeight;
    finalWidth = currentContainerHeight * imageRatio;
  }
  console.log(`[${container.id}] [CALCULATED DIMS] finalWidth: ${finalWidth}, finalHeight: ${finalHeight}`);

  scale = finalWidth / currentNaturalWidth;

  const offsetX = (currentContainerWidth - finalWidth) / 2;
  const offsetY = (currentContainerHeight - finalHeight) / 2;

  const debugRect = document.createElement('div');
  debugRect.style.position = 'absolute';
  debugRect.style.left = `${offsetX}px`;
  debugRect.style.top = `${offsetY}px`;
  debugRect.style.width = `${finalWidth}px`;
  debugRect.style.height = `${finalHeight}px`;
  debugRect.style.border = '3px dashed limegreen';
  debugRect.style.zIndex = '1';
  debugRect.style.boxSizing = 'border-box';
  debugRect.id = `debug-rect-${container.id}`;
  container.appendChild(debugRect);

  console.log(`[${container.id}] Debug Info Post-Calc:`, {
    scale, offsetX, offsetY
  });

  faces.forEach((face, idx) => {
    const [x1, y1, x2, y2] = face.bbox;
    console.log(`[${container.id}] Face ${idx} RAW BBOX: [${x1}, ${y1}, ${x2}, ${y2}]`);

    const box = document.createElement('div');
    box.className = 'face-box';
    // CSS: .face-box { position: absolute; box-sizing: border-box; }

    box.style.left = `${(x1 * scale) + offsetX}px`;
    box.style.top = `${(y1 * scale) + offsetY}px`;
    box.style.width = `${(x2 - x1) * scale}px`;
    box.style.height = `${(y2 - y1) * scale}px`;

    const label = document.createElement('span');
    label.className = 'face-box-label';
    label.textContent = idx + 1;
    box.appendChild(label);

    box.onclick = (e) => {
      e.stopPropagation();
      const type = container.id.includes('source') ? 'source' : 'target';
      handleFaceSelection(idx, type);
    };
    container.appendChild(box);

    console.log(`[${container.id}] Face ${idx} Applied Styles: left=${box.style.left}, top=${box.style.top}, width=${box.style.width}, height=${box.style.height}`);
  });
  console.log(`[${container.id}] --- RATIO CHECK END ---`);
}

export function updateSelectionHighlights(container, selectedIndex) {
  if (!container) return;
  container.querySelectorAll('.face-box').forEach((b, i) => b.classList.toggle('selected', i === selectedIndex));
}

export function refreshFaceBoxes() {
  // Utilizza requestAnimationFrame per sincronizzare con il ciclo di rendering del browser
  requestAnimationFrame(() => {
    console.log('requestAnimationFrame: refreshFaceBoxes triggered');

    // Source Image
    if (dom.sourceImgPreview && dom.sourceImgPreview.complete &&
        dom.sourceImgPreview.naturalWidth > 0 && dom.sourceImgPreview.naturalHeight > 0 && // Controlla anche naturalHeight
        dom.sourceFaceBoxesContainer &&
        dom.sourceFaceBoxesContainer.offsetWidth > 0 && dom.sourceFaceBoxesContainer.offsetHeight > 0) { // Controlla anche dimensioni container
      console.log('Refreshing source faces for:', dom.sourceImgPreview.src.substring(0,100));
      drawFaceBoxes(dom.sourceFaceBoxesContainer, dom.sourceImgPreview, state.sourceFaces);
      updateSelectionHighlights(dom.sourceFaceBoxesContainer, state.selectedSourceIndex);
    } else {
      console.warn('Cannot refresh source faces, image or container not ready or has zero dimensions', {
        imgSrc: dom.sourceImgPreview?.src.substring(0,50),
        complete: dom.sourceImgPreview?.complete,
        naturalWidth: dom.sourceImgPreview?.naturalWidth,
        naturalHeight: dom.sourceImgPreview?.naturalHeight,
        containerExists: !!dom.sourceFaceBoxesContainer,
        containerOffsetWidth: dom.sourceFaceBoxesContainer?.offsetWidth,
        containerOffsetHeight: dom.sourceFaceBoxesContainer?.offsetHeight,
      });
      // Pulisci i box se non possiamo disegnare, per evitare box vecchi
      if (dom.sourceFaceBoxesContainer) dom.sourceFaceBoxesContainer.innerHTML = '';
    }

    // Target/Result Image
    if (dom.resultImageDisplay && dom.resultImageDisplay.complete &&
        dom.resultImageDisplay.naturalWidth > 0 && dom.resultImageDisplay.naturalHeight > 0 && // Controlla anche naturalHeight
        dom.targetFaceBoxesContainer &&
        dom.targetFaceBoxesContainer.offsetWidth > 0 && dom.targetFaceBoxesContainer.offsetHeight > 0) { // Controlla anche dimensioni container
      console.log('Refreshing target faces for:', dom.resultImageDisplay.src.substring(0,100));
      drawFaceBoxes(dom.targetFaceBoxesContainer, dom.resultImageDisplay, state.targetFaces);
      updateSelectionHighlights(dom.targetFaceBoxesContainer, state.selectedTargetIndex);
    } else {
      console.warn('Cannot refresh target faces, image or container not ready or has zero dimensions', {
        imgSrc: dom.resultImageDisplay?.src.substring(0,50),
        complete: dom.resultImageDisplay?.complete,
        naturalWidth: dom.resultImageDisplay?.naturalWidth,
        naturalHeight: dom.resultImageDisplay?.naturalHeight,
        containerExists: !!dom.targetFaceBoxesContainer,
        containerOffsetWidth: dom.targetFaceBoxesContainer?.offsetWidth,
        containerOffsetHeight: dom.targetFaceBoxesContainer?.offsetHeight,
      });
      // Pulisci i box se non possiamo disegnare
      if (dom.targetFaceBoxesContainer) dom.targetFaceBoxesContainer.innerHTML = '';
    }
  });
}

export async function detectAndDrawFaces(blob, imageElement, faceBoxesContainer, facesArray, type) {
  if (!imageElement || !faceBoxesContainer) {
    showError('Errore Interno', `Elemento immagine (${type}) o contenitore non fornito a detectAndDrawFaces.`);
    console.error(`detectAndDrawFaces (${type}): imageElement o faceBoxesContainer mancanti`, {imageElement, faceBoxesContainer});
    return;
  }

  const onImageLoadOrReady = async () => {
    // Controlli aggiuntivi prima di procedere, simili a refreshFaceBoxes
    if (!imageElement.complete || imageElement.naturalWidth === 0 || imageElement.naturalHeight === 0) {
      console.warn(`Immagine ${type} (${imageElement.src.substring(0,50)}) non ancora pronta per il rilevamento volti (complete: ${imageElement.complete}, naturalWidth: ${imageElement.naturalWidth}, naturalHeight: ${imageElement.naturalHeight}). Attendo onload se non già attaccato.`);
      if (!imageElement.dataset.onloadAttached) { // Evita di riattaccare onload se è già in attesa
          imageElement.onload = () => { // Assicurati di rimuovere il flag e l'handler dopo l'esecuzione
            delete imageElement.dataset.onloadAttached;
            imageElement.onload = null; 
            imageElement.onerror = null;
            onImageLoadOrReady(); // Richiama per processare
          };
          imageElement.onerror = () => {
            console.error(`Errore caricamento immagine ${type}: ${imageElement.src}`);
            showError(`Errore caricamento Immagine (${type})`, `Impossibile caricare: ${imageElement.src.substring(0,100)}...`);
            delete imageElement.dataset.onloadAttached;
            imageElement.onload = null;
            imageElement.onerror = null;
          };
          imageElement.dataset.onloadAttached = 'true';
      }
      return;
    }
    // Se siamo qui, l'immagine è 'complete' e ha dimensioni > 0
    delete imageElement.dataset.onloadAttached; // Rimuovi il flag se l'immagine è già pronta
    imageElement.onload = null; // Rimuovi handler per evitare chiamate multiple
    imageElement.onerror = null;

    console.log(`Rilevamento volti per immagine ${type}:`, imageElement.src.substring(0, 100) + "...");
    try {
      const data = await api.detectFaces(blob);
      const faceData = data && Array.isArray(data.faces) ? data.faces : [];
      facesArray.splice(0, facesArray.length, ...faceData);
      console.log(`Volti rilevati per ${type}: ${faceData.length}`);
      refreshFaceBoxes(); // Questo chiamerà drawFaceBoxes
    } catch (err) {
      console.error(`Errore API Rilevamento Volti (${type})`, err);
      showError(`Errore Rilevamento Volti (${type})`, err.message || JSON.stringify(err));
      facesArray.length = 0;
      refreshFaceBoxes(); // Aggiorna per rimuovere i box
    }
  };

  // Se l'immagine è già caricata e ha dimensioni, procedi.
  // Altrimenti, imposta l'handler onload.
  if (imageElement.complete && imageElement.naturalWidth > 0 && imageElement.naturalHeight > 0) {
    console.log(`Immagine ${type} (${imageElement.src.substring(0,50)}) già completa e con dimensioni, avvio rilevamento.`);
    // Un piccolo timeout può aiutare se questa funzione è chiamata immediatamente dopo aver impostato src
    // ma con requestAnimationFrame in refreshFaceBoxes, potrebbe non essere necessario.
    // Prova a rimuoverlo o a tenerlo a 0 per vedere se cambia qualcosa.
    setTimeout(onImageLoadOrReady, 0); 
  } else {
    console.log(`Immagine ${type} (${imageElement.src.substring(0,50)}) non completa o senza dimensioni valide, imposto onload/onerror.`);
    imageElement.onload = () => {
        delete imageElement.dataset.onloadAttached;
        imageElement.onload = null; 
        imageElement.onerror = null;
        onImageLoadOrReady();
    };
    imageElement.onerror = () => {
        console.error(`Errore caricamento immagine ${type}: ${imageElement.src}`);
        showError(`Errore caricamento Immagine (${type})`, `Impossibile caricare: ${imageElement.src.substring(0,100)}...`);
        delete imageElement.dataset.onloadAttached;
        imageElement.onload = null;
        imageElement.onerror = null;
    };
    imageElement.dataset.onloadAttached = 'true'; // Imposta il flag per indicare che stiamo aspettando onload
  }
}

function handleFaceSelection(index, type) {
  if (type === 'source') {
    state.selectedSourceIndex = index === state.selectedSourceIndex ? -1 : index;
  } else {
    state.selectedTargetIndex = index === state.selectedTargetIndex ? -1 : index;
  }

  if (dom.sourceFaceBoxesContainer) {
    updateSelectionHighlights(dom.sourceFaceBoxesContainer, state.selectedSourceIndex);
  }
  if (dom.targetFaceBoxesContainer) {
    updateSelectionHighlights(dom.targetFaceBoxesContainer, state.selectedTargetIndex);
  }

  if(dom.selectedSourceId) dom.selectedSourceId.textContent = state.selectedSourceIndex > -1 ? (state.selectedSourceIndex + 1).toString() : 'Nessuno';
  if(dom.selectedTargetId) dom.selectedTargetId.textContent = state.selectedTargetIndex > -1 ? (state.selectedTargetIndex + 1).toString() : 'Nessuno';
  if(dom.swapBtn) dom.swapBtn.disabled = state.selectedSourceIndex < 0 || state.selectedTargetIndex < 0;
}

export function initFaceBoxObservers() {
  if (window.ResizeObserver) {
    const observedElements = new Set();
    const ro = new ResizeObserver((entries) => {
      console.log('ResizeObserver triggered refreshFaceBoxes per gli elementi:', entries.map(e => e.target.id || e.target.className || e.target.tagName));
      refreshFaceBoxes();
    });

    const elementsToObserve = [
        dom.sourceFaceBoxesContainer, // Osserva direttamente i contenitori dei box
        dom.targetFaceBoxesContainer,
        dom.sourceImgPreview,         // E anche le immagini, se le loro dimensioni cambiano
        dom.resultImageDisplay
    ];

    elementsToObserve.forEach(el => {
      if(el && !observedElements.has(el)) { // Aggiungi controllo per el non nullo
        try {
            ro.observe(el);
            observedElements.add(el);
            console.log('ResizeObserver: observing', el.id || el.className || el.tagName);
        } catch (e) {
            console.error('Errore durante observe con ResizeObserver:', e, el);
        }
      }
    });
  } else {
    window.addEventListener('resize', () => {
      console.log('Window resize triggered refreshFaceBoxes');
      refreshFaceBoxes();
    });
  }
}
