import * as api from './api.js';
import { state, dom } from './state.js';
import { updateMemePreview, handleDownloadAnimation } from './memeEditor.js';
import { getStickerAtPosition } from './stickers.js';
import { addToGallery } from './gallery.js';
// L'importazione corretta deve venire da facebox.js
import { refreshFaceBoxes, detectAndDrawFaces } from "./facebox.js";

export function displayImage(src, imageElement, onLoadCallback = null) {
  if (!src || !imageElement) return;
  const oldUrl = imageElement.dataset.blobUrl;
  if (oldUrl) { URL.revokeObjectURL(oldUrl); imageElement.dataset.blobUrl = ''; }

  const finalize = url => {
    imageElement.onload = () => {
      refreshFaceBoxes(); // Usa la funzione centralizzata per ridisegnare
      if (onLoadCallback) {
        onLoadCallback();
      }
    };
    imageElement.src = url;
    imageElement.classList.remove('hidden');
    if (imageElement.id === 'result-image-display') {
      dom.resultPlaceholder.classList.add('hidden');
      dom.downloadBtn.classList.remove('hidden');
      if(dom.addGalleryBtn) dom.addGalleryBtn.classList.remove('hidden');
      if(dom.shareBtn) dom.shareBtn.classList.remove('hidden');
    } else if (imageElement.id === 'imagePreview') {
      dom.subjectUploadPrompt.style.display = 'none';
    } else if (imageElement.id === 'source-img-preview') {
      dom.sourceUploadPrompt.style.opacity = '0';
    }
  };

  if (src instanceof File) {
    const reader = new FileReader();
    reader.onload = () => finalize(reader.result);
    reader.readAsDataURL(src);
  } else if (src instanceof Blob) {
    const url = URL.createObjectURL(src);
    imageElement.dataset.blobUrl = url;
    finalize(url);
  } else if (typeof src === 'string') {
    finalize(src);
  }
}

export function startProgressBar(title, duration = 30) {
  dom.progressTitle.textContent = title;
  dom.progressModal.style.display = 'flex';
  let progress = 0;
  const interval = setInterval(() => {
    progress = Math.min(progress + 1, 95);
    dom.progressBar.style.width = `${progress}%`;
    dom.progressText.textContent = `${progress}%`;
    if (progress >= 95) clearInterval(interval);
  }, (duration * 1000) / 100);
}

export function finishProgressBar() {
  setTimeout(() => {
    dom.progressBar.style.width = '100%';
    dom.progressText.textContent = '100%';
    setTimeout(() => { dom.progressModal.style.display = 'none'; }, 500);
  }, 200);
}

export function closeModal(id) {
  const modal = document.getElementById(id);
  if (modal) modal.style.display = 'none';
}

export function showError(title, message) {
  if (dom.errorModal) {
    dom.errorModal.style.display = 'flex';
    const errorTitle = document.getElementById('error-title');
    const errorMessage = document.getElementById('error-message');
    if(errorTitle) errorTitle.textContent = title;
    if(errorMessage) errorMessage.textContent = message;
    dom.errorModal.querySelector('button')?.focus();
  } else {
    alert(`${title}: ${message}`);
  }
}

export function showToast(msg) {
  const t = document.getElementById('toast');
  if (!t) return;
  t.textContent = msg;
  t.classList.remove('hidden');
  setTimeout(() => t.classList.add('hidden'), 2000);
}

export function goToStep(stepNumber) {
  state.currentStep = stepNumber;
  document.querySelectorAll('.wizard-step').forEach(step => step.classList.add('hidden'));
  const stepId = `step-${stepNumber}-${['subject', 'scene', 'upscale', 'finalize'][stepNumber - 1]}`;
  document.getElementById(stepId)?.classList.remove('hidden');
}

export async function loadAvailableModels() {
  if (!dom.modelSelector) return;
  try {
    const res = await fetch('/api/models/list');
    const data = await res.json();
    const models = data.models || data;
    const active = data.active;
    dom.modelSelector.innerHTML = '';
    models.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      if (name === active) opt.selected = true;
      dom.modelSelector.appendChild(opt);
    });
    if (!dom.modelSelector.value && models.length > 0) {
      dom.modelSelector.value = models[0];
    }
  } catch (err) {
    console.error('Errore caricamento modelli', err);
  }
}

export function handleSubjectFile(file) {
  if (!file?.type.startsWith('image/')) return;
  state.subjectFile = file;
  displayImage(file, dom.imagePreview, () => {
      detectAndDrawFaces(file, dom.imagePreview, dom.targetFaceBoxesContainer, state.targetFaces, 'target');
  });
  dom.prepareSubjectBtn.disabled = false;
  dom.skipToSwapBtn.disabled = false;
}

export function handleSourceFile(file) {
  if (!file?.type.startsWith('image/')) return;
  state.sourceImageFile = file;
  displayImage(file, dom.sourceImgPreview, () => {
    detectAndDrawFaces(file, dom.sourceImgPreview, dom.sourceFaceBoxesContainer, state.sourceFaces, 'source');
  });
  dom.selectionStatus.classList.remove('hidden');
}

export function handleSkipToSwap() {
  if (!state.subjectFile) {
    showError('File Mancante', "Carica un'immagine soggetto (step 1) prima.");
    return;
  }

  console.log("[handleSkipToSwap] Avviato. state.subjectFile:", state.subjectFile);

  // PUNTO CHIAVE: Usiamo state.subjectFile (che è un oggetto File)
  // per popolare dom.resultImageDisplay e per il rilevamento volti.
  // DisplayImage ora gestisce File e Blob creando un ObjectURL.
  state.upscaledImageBlob = state.subjectFile; // Manteniamo questo per coerenza di nome, ma è ancora il File originale

  state.processedSubjectBlob = null; // Resetta gli altri stati come prima
  state.sceneImageBlob = null;
  state.finalImageWithSwap = null;

  // Chiamiamo displayImage con state.subjectFile.
  // displayImage creerà un URL.createObjectURL(state.subjectFile)
  displayImage(state.subjectFile, dom.resultImageDisplay, () => {
    console.log("[handleSkipToSwap] displayImage onLoad callback eseguito.");
    console.log("  dom.resultImageDisplay naturalWidth:", dom.resultImageDisplay.naturalWidth);
    console.log("  dom.resultImageDisplay naturalHeight:", dom.resultImageDisplay.naturalHeight);
    console.log("  dom.targetFaceBoxesContainer offsetWidth:", dom.targetFaceBoxesContainer.offsetWidth);
    console.log("  dom.targetFaceBoxesContainer offsetHeight:", dom.targetFaceBoxesContainer.offsetHeight);
    
    // Passiamo state.subjectFile a detectAndDrawFaces.
    // L'API (api.detectFaces) riceverà l'oggetto File/Blob.
    detectAndDrawFaces(state.subjectFile, dom.resultImageDisplay, dom.targetFaceBoxesContainer, state.targetFaces, 'target');
  });
  goToStep(4);
}

export async function handlePrepareSubject() {
  if (!state.subjectFile) return;
  startProgressBar('Step 1: Preparazione Soggetto...', 15);
  try {
    state.processedSubjectBlob = await api.prepareSubject(state.subjectFile);
    displayImage(state.processedSubjectBlob, dom.resultImageDisplay, () => {
        detectAndDrawFaces(state.processedSubjectBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, state.targetFaces, 'target');
    });
    goToStep(2);
  } catch (err) {
    showError('Errore Preparazione', err.message);
  } finally {
    finishProgressBar();
  }
}

async function pollTask(taskId, progressTitle = 'Elaborazione AI...') {
  startProgressBar(progressTitle, 60);
  const pollingInterval = 2000;
  try {
    while (true) {
      const status = await api.getTaskStatus(taskId);
      if (status.progress) {
        dom.progressBar.style.width = `${status.progress}%`;
        dom.progressText.textContent = `${status.progress}%`;
      }
      if (status.state === 'SUCCESS') {
        if (status.result && status.result.data) {
          const imageUrl = `data:image/png;base64,${status.result.data}`;
          return await (await fetch(imageUrl)).blob();
        } else {
          throw new Error('Task completato ma senza dati validi.');
        }
      }
      if (status.state === 'FAILURE' || status.state === 'REVOKED') {
        throw new Error(status.error || 'Il task è fallito.');
      }
      await new Promise(resolve => setTimeout(resolve, pollingInterval));
    }
  } finally {
    finishProgressBar();
  }
}

async function pollGenericTask(taskId, progressTitle = 'Operazione in corso...') {
  startProgressBar(progressTitle, 60);
  const pollingInterval = 2000;
  try {
    while (true) {
      const status = await api.getTaskStatus(taskId);
      if (status.progress) {
        dom.progressBar.style.width = `${status.progress}%`;
        dom.progressText.textContent = `${status.progress}%`;
      }
      if (status.state === 'SUCCESS') {
        return status.result;
      }
      if (status.state === 'FAILURE' || status.state === 'REVOKED') {
        throw new Error(status.error || 'Il task è fallito.');
      }
      await new Promise(resolve => setTimeout(resolve, pollingInterval));
    }
  } finally {
    finishProgressBar();
  }
}

export async function handleInstallModel() {
  const url = dom.modelUrlInput.value.trim();
  if (!url) return showError('URL mancante', 'Inserisci un link valido da Civitai.');
  closeModal('model-modal');
  try {
    // La logica della chiamata è ora nascosta dentro api.js
    const data = await api.downloadModel(url); 
    
    if (!data.task_id) throw new Error('Risposta del server non valida.');
    
    await pollGenericTask(data.task_id, '📥 Download modello in corso...');
    await loadAvailableModels();
    showToast('Modello installato');
  } catch (err) {
    showError('Errore Download Modello', err.message);
  }
}

export async function handleCreateScene() {
    if (!dom.bgPromptInput.value || !state.processedSubjectBlob) {
        return showError('Dati Mancanti', 'Assicurati di aver caricato un soggetto e scritto un prompt.');
    }
    try {
        let prompt = dom.bgPromptInput.value;
        if (dom.autoEnhancePromptToggle.checked) {
            startProgressBar('✨ Miglioramento prompt con AI...', 10);
            const enhanced = await api.enhancePrompt(state.processedSubjectBlob, prompt);
            prompt = enhanced.enhanced_prompt;
            dom.bgPromptInput.value = prompt;
            finishProgressBar();
        }
        const model = dom.modelSelector ? dom.modelSelector.value : undefined;
        const taskInfo = await api.createSceneAsync(state.processedSubjectBlob, prompt, model);
        const resultBlob = await pollTask(taskInfo.task_id, '🎨 Creazione Scena AI in corso...');
        state.sceneImageBlob = resultBlob;
        state.finalImageWithSwap = null;
        displayImage(state.sceneImageBlob, dom.resultImageDisplay, () => {
            detectAndDrawFaces(state.sceneImageBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, state.targetFaces, 'target');
        });
        dom.gotoStep3Btn.disabled = false;
    } catch (err) {
        finishProgressBar();
        showError('Errore Creazione Scena', err.message);
    }
}

export async function handleUpscaleAndDetail() {
    if (!state.sceneImageBlob) {
        return showError('Dati Mancanti', 'Genera prima una scena.');
    }
    const enableHires = dom.enableHiresUpscaleToggle.checked;
    const denoising = dom.tileDenoisingSlider.value;
    try {
        const taskInfo = await api.detailAndUpscaleAsync(state.sceneImageBlob, enableHires, denoising);
        const resultBlob = await pollTask(taskInfo.task_id, '✨ Miglioramento e Upscaling in corso...');
        state.upscaledImageBlob = resultBlob;
        state.finalImageWithSwap = null;
        displayImage(state.upscaledImageBlob, dom.resultImageDisplay, () => {
            detectAndDrawFaces(state.upscaledImageBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, state.targetFaces, 'target');
        });
        goToStep(4);
    } catch (err) {
        showError('Errore Upscale', err.message);
    }
}

export async function handlePerformSwap() {
    const targetImg = state.finalImageWithSwap || state.upscaledImageBlob;
    if (state.selectedSourceIndex < 0 || state.selectedTargetIndex < 0 || !targetImg || !state.sourceImageFile) {
        return showError('Dati Mancanti', 'Seleziona un volto sorgente e uno di destinazione.');
    }
    try {
        const taskInfo = await api.finalSwapAsync(targetImg, state.sourceImageFile, state.selectedSourceIndex, state.selectedTargetIndex);
        const resultBlob = await pollTask(taskInfo.task_id, 'Face Swap in corso...');
        state.finalImageWithSwap = resultBlob;
        displayImage(resultBlob, dom.resultImageDisplay, () => {
            detectAndDrawFaces(resultBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, state.targetFaces, 'target');
        });
    } catch (err) {
        showError('Errore Face Swap', err.message);
    }
}

export async function handleAnalyzeParts() {
  const imageToAnalyze = state.finalImageWithSwap || state.upscaledImageBlob || state.sceneImageBlob || state.processedSubjectBlob || state.subjectFile;
  if (!imageToAnalyze) return showError('Immagine Mancante', 'Carica un\'immagine prima di analizzarla.');
  startProgressBar('🤖 Analisi AI delle parti del corpo...', 15);
  dom.analyzePartsBtn.disabled = true;
  try {
    const result = await api.analyzeParts(imageToAnalyze);
    if (result.parts && result.parts.length > 0) {
      renderDynamicPrompts(result.parts);
      dom.generateAllBtn.classList.remove('hidden');
      dom.generateAllBtn.disabled = false;
    } else {
      showError('Nessuna Parte Trovata', 'Il modello non ha rilevato parti umane modificabili.');
    }
  } catch (err) {
    showError('Errore Analisi', err.message);
  } finally {
    finishProgressBar();
    dom.analyzePartsBtn.disabled = false;
  }
}

export function renderDynamicPrompts(parts) {
  const container = dom.dynamicPromptsContainer;
  container.innerHTML = '';
  parts.forEach(part => {
    const pretty = part.charAt(0).toUpperCase() + part.slice(1);
    container.innerHTML += `<div class="flex items-center gap-2">
      <label for="prompt-${part}" class="w-1/4 text-sm text-right text-gray-400">${pretty}:</label>
      <input type="text" id="prompt-${part}" data-part-name="${part}" class="prompt-input w-3/4 bg-gray-900 border border-gray-600 rounded-lg p-2 text-sm text-gray-300" placeholder="Descrivi modifica per ${part}...">
      <button class="enhance-part-btn btn btn-secondary text-white font-bold p-2 rounded-lg" data-part-name="${part}" title="Migliora prompt con AI">✨</button>
    </div>`;
  });
}

export async function handleGenerateAll() {
    const imageToInpaint = state.finalImageWithSwap || state.upscaledImageBlob || state.sceneImageBlob || state.processedSubjectBlob || state.subjectFile;
    if (!imageToInpaint) return showError('Immagine Mancante', 'Non c\'è un\'immagine su cui applicare le modifiche.');
    const prompts = {};
    const inputs = dom.dynamicPromptsContainer.querySelectorAll('.prompt-input');
    for (const input of inputs) {
        const partName = input.dataset.partName;
        const promptText = input.value.trim();
        if (promptText) prompts[partName] = promptText;
    }
    if (Object.keys(prompts).length === 0) return showError('Nessun Prompt', 'Scrivi una descrizione per almeno una parte.');
    dom.generateAllBtn.disabled = true;
    try {
        const taskInfo = await api.generateAllPartsAsync(imageToInpaint, prompts);
        const resultBlob = await pollTask(taskInfo.task_id, '🎨 Generazione Multi-Parte in corso...');
        state.finalImageWithSwap = resultBlob;
        displayImage(resultBlob, dom.resultImageDisplay, () => {
            detectAndDrawFaces(resultBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, state.targetFaces, 'target');
        });
    } catch (err) {
        showError('Errore durante la Generazione', err.message);
    } finally {
        dom.generateAllBtn.disabled = false;
        dom.dynamicPromptsContainer.querySelectorAll('.prompt-input').forEach(inp => inp.value = '');
    }
}

export async function handleGenerateCaption() {
    const imageToCaption = state.finalImageWithSwap || state.upscaledImageBlob || state.sceneImageBlob || state.processedSubjectBlob || state.subjectFile;
    if (!imageToCaption) {
        return showError('Immagine Mancante', 'Nessuna immagine da descrivere.');
    }
    startProgressBar('✨ Generazione Didascalia AI...', 10);
    
    // Selezioniamo il nostro "cartello" per gli avvisi una sola volta
    const notificationDiv = dom.fallbackNotification;

    try {
        const tone = dom.toneButtonsContainer.querySelector('.active')?.dataset.tone || 'scherzoso';
        
        // La chiamata API ora puo' restituire 'caption' e 'fallback_message'
        const result = await api.generateCaption(imageToCaption, tone);

        // --- NUOVA LOGICA PER GESTIRE IL MESSAGGIO DI FALLBACK ---
        if (result.fallback_message) {
            // Se SÌ: scriviamo il messaggio e mostriamo il cartello
            notificationDiv.textContent = result.fallback_message;
            notificationDiv.classList.remove('hidden');
        } else {
            // Se NO: ci assicuriamo che il cartello sia nascosto
            notificationDiv.classList.add('hidden');
        }
        // --- FINE NUOVA LOGICA ---

        // Il resto della logica rimane identico
        dom.captionTextInput.value = result.caption;
        updateMemePreview();

    } catch (err) {
        showError('Errore Generazione Didascalia', err.message);
        // Nascondiamo il box di notifica anche in caso di errore
        if (notificationDiv) notificationDiv.classList.add('hidden');
    } finally {
        finishProgressBar();
    }
}

export async function handleShare() {
  updateMemePreview();
  await new Promise(r => setTimeout(r, 50));
  const url = dom.memeCanvas.classList.contains('hidden') ? dom.resultImageDisplay.src : dom.memeCanvas.toDataURL('image/png');
  if (!navigator.share) return showError('Condivisione non supportata', 'Scarica l\'immagine e condividila manualmente.');
  try {
    let shareData = { title: 'Il mio meme' };
    if (navigator.canShare && url.startsWith('data:')) {
      const blob = await (await fetch(url)).blob();
      const file = new File([blob], 'meme.png', { type: blob.type || 'image/png' });
      if (navigator.canShare({ files: [file] })) shareData.files = [file];
      else shareData.url = url;
    } else {
      shareData.url = url;
    }
    await navigator.share(shareData);
  } catch (err) {
    showError('Errore condivisione', err.message);
  }
}

export function setupEventListeners() {
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      closeModal('error-modal');
      closeModal('progress-modal');
    }
  });
  dom.resetAllBtn.addEventListener('click', resetWorkflow);
  dom.fileInput.addEventListener('change', e => handleSubjectFile(e.target.files[0]));
  dom.prepareSubjectBtn.addEventListener('click', handlePrepareSubject);
  dom.skipToSwapBtn.addEventListener('click', handleSkipToSwap);
  dom.generateSceneBtn.addEventListener('click', handleCreateScene);
  dom.gotoStep3Btn.addEventListener('click', () => goToStep(3));
  dom.addModelBtn.addEventListener('click', () => {
    dom.modelModal.style.display = 'flex';
    dom.modelUrlInput.focus();
  });
  dom.cancelModelBtn.addEventListener('click', () => closeModal('model-modal'));
  dom.installModelBtn.addEventListener('click', handleInstallModel);
  dom.modelUrlInput.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); handleInstallModel(); } });
  dom.startUpscaleBtn.addEventListener('click', handleUpscaleAndDetail);
  dom.captionBtn.addEventListener('click', handleGenerateCaption);
  dom.tileDenoisingSlider.addEventListener('input', e => dom.tileDenoisingValue.textContent = parseFloat(e.target.value).toFixed(2));
  dom.skipUpscaleBtn.addEventListener('click', () => {
    state.upscaledImageBlob = state.sceneImageBlob;
    displayImage(state.upscaledImageBlob, dom.resultImageDisplay, () => {
        detectAndDrawFaces(state.upscaledImageBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, state.targetFaces, 'target');
    });
    goToStep(4);
  });
  dom.sourceImgInput.addEventListener('change', e => handleSourceFile(e.target.files[0]));
  dom.swapBtn.addEventListener('click', handlePerformSwap);
  dom.backToStep3Btn.addEventListener('click', () => goToStep(3));
  dom.toggleFaceBoxes.addEventListener('change', e => {
    const display = e.target.checked ? 'block' : 'none';
    dom.targetFaceBoxesContainer.style.display = display;
    dom.sourceFaceBoxesContainer.style.display = display;
  });
  dom.filterButtonsContainer.addEventListener('click', e => {
    if (e.target.tagName === 'BUTTON') {
      dom.filterButtonsContainer.querySelector('.active')?.classList.remove('active');
      e.target.classList.add('active');
      state.activeFilter = e.target.dataset.filter;
      dom.resultImageDisplay.style.filter = state.activeFilter;
      updateMemePreview();
    }
  });
  dom.analyzePartsBtn.addEventListener('click', handleAnalyzeParts);
  dom.generateAllBtn.addEventListener('click', handleGenerateAll);
  dom.dynamicPromptsContainer.addEventListener('click', async e => {
    if (e.target && e.target.classList.contains('enhance-part-btn')) {
      const button = e.target;
      const partName = button.dataset.partName;
      const input = document.getElementById(`prompt-${partName}`);
      const imageForEnh = state.finalImageWithSwap || state.upscaledImageBlob || state.sceneImageBlob || state.processedSubjectBlob || state.subjectFile;
      if (!input || !input.value) return showError('Prompt Vuoto', "Scrivi prima un'idea da migliorare.");
      if (!imageForEnh) return showError('Immagine Mancante', 'Carica un\'immagine per migliorare il prompt.');
      button.disabled = true;
      startProgressBar(`✨ Miglioramento prompt per ${partName}...`, 10);
      try {
        const result = await api.enhancePartPrompt(partName, input.value, imageForEnh);
        input.value = result.enhanced_prompt;
      } catch (err) {
        showError('Errore Miglioramento Prompt', err.message);
      } finally {
        finishProgressBar();
        button.disabled = false;
      }
    }
  });
  ['captionTextInput', 'fontFamilySelect', 'fontColorInput', 'strokeColorInput'].forEach(id => dom[id] && dom[id].addEventListener('input', updateMemePreview));
  if(dom.fontSizeSlider) dom.fontSizeSlider.addEventListener('input', () => { dom.fontSizeValue.textContent = dom.fontSizeSlider.value; updateMemePreview(); });
  [dom.positionButtons, dom.textBgButtons, dom.toneButtonsContainer].forEach(container => {
    if(container) container.addEventListener('click', e => {
      if (e.target.classList.contains('meme-control-btn') || e.target.classList.contains('tone-btn')) {
        container.querySelector('.active')?.classList.remove('active');
        e.target.classList.add('active');
        if (container !== dom.toneButtonsContainer) updateMemePreview();
      }
    });
  });
  dom.stickerSearchInput.addEventListener('input', e => {
    const term = e.target.value.toLowerCase();
    document.querySelectorAll('#sticker-gallery .category-title').forEach(title => {
      const container = title.nextElementSibling;
      let categoryVisible = false;
      container.querySelectorAll('.sticker-item-wrapper').forEach(wrapper => {
        const match = (wrapper.querySelector('.sticker-item')?.src || '').toLowerCase().includes(term) || title.textContent.toLowerCase().includes(term);
        wrapper.style.display = match ? 'inline-block' : 'none';
        if (match) categoryVisible = true;
      });
      title.style.display = categoryVisible ? 'block' : 'none';
    });
  });
  const stickerControls = [dom.stickerDeleteBtn, dom.stickerFrontBtn, dom.stickerBackBtn];
  stickerControls[0]?.addEventListener('click', () => {
    if (state.selectedSticker) state.stickerStack = state.stickerStack.filter(s => s !== state.selectedSticker);
    state.selectedSticker = null;
  });
  stickerControls[1]?.addEventListener('click', () => {
    if (!state.selectedSticker) return;
    const i = state.stickerStack.indexOf(state.selectedSticker);
    if (i < state.stickerStack.length - 1) { state.stickerStack.splice(i, 1); state.stickerStack.push(state.selectedSticker); }
  });
  stickerControls[2]?.addEventListener('click', () => {
    if (!state.selectedSticker) return;
    const i = state.stickerStack.indexOf(state.selectedSticker);
    if (i > 0) { state.stickerStack.splice(i, 1); state.stickerStack.unshift(state.selectedSticker); }
  });
  dom.downloadBtn.addEventListener('click', async e => {
    e.preventDefault();
    state.selectedSticker = null;
    updateMemePreview();
    await new Promise(r => setTimeout(r, 50));
    const dataUrl = dom.memeCanvas.classList.contains('hidden') ? dom.resultImageDisplay.src : dom.memeCanvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = 'pro-meme-result.png';
    link.click();
  });
  if(dom.addGalleryBtn) dom.addGalleryBtn.addEventListener('click', () => {
    updateMemePreview();
    const src = dom.memeCanvas.classList.contains('hidden') ? dom.resultImageDisplay.src : dom.memeCanvas.toDataURL('image/png');
    addToGallery(`Meme`, src, dom.captionTextInput.value, [], false);
  });
  if(dom.shareBtn) dom.shareBtn.addEventListener('click', handleShare);
  if(dom.downloadAnimBtn) dom.downloadAnimBtn.addEventListener('click', handleDownloadAnimation);
  
  const getCoords = e => {
    const rect = dom.memeCanvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: (clientX - rect.left) * (dom.memeCanvas.width / rect.width), y: (clientY - rect.top) * (dom.memeCanvas.height / rect.height) };
  };
  const onStart = e => {
    const hit = getStickerAtPosition(getCoords(e).x, getCoords(e).y);
    if (hit) {
      e.preventDefault();
      state.selectedSticker = hit.sticker;
      if (hit.corner === 'resize') state.isResizing = true;
      else if (hit.corner === 'rotate') state.isRotating = true;
      else { state.isDragging = true; const c = getCoords(e); state.dragOffsetX = c.x - hit.sticker.x; state.dragOffsetY = c.y - hit.sticker.y; }
    } else {
      state.selectedSticker = null;
    }
    stickerControls.forEach(b => b && (b.disabled = !state.selectedSticker));
  };
  const onMove = e => {
    if (!state.selectedSticker || !(state.isDragging || state.isResizing || state.isRotating)) return;
    e.preventDefault();
    const { x, y } = getCoords(e);
    const s = state.selectedSticker, cx = s.x + s.width / 2, cy = s.y + s.height / 2;
    if (state.isResizing) {
      const newW = Math.sqrt(Math.pow(x - cx, 2) + Math.pow(y - cy, 2)) * Math.sqrt(2);
      if (newW > 20) { s.x += (s.width - newW) / 2; s.y += (s.height - newW * s.aspectRatio) / 2; s.width = newW; s.height = newW * s.aspectRatio; }
    } else if (state.isRotating) {
      s.rotation = Math.atan2(y - cy, x - cx) + Math.PI / 2;
    } else if (state.isDragging) {
      s.x = x - state.dragOffsetX; s.y = y - state.dragOffsetY;
    }
  };
  const onEnd = () => { state.isDragging = state.isResizing = state.isRotating = false; };
  ['mousedown', 'touchstart'].forEach(evt => dom.memeCanvas.addEventListener(evt, onStart, { passive: false }));
  ['mousemove', 'touchmove'].forEach(evt => document.addEventListener(evt, onMove, { passive: false }));
  ['mouseup', 'touchend', 'touchcancel'].forEach(evt => document.addEventListener(evt, onEnd));
}

export function resetWorkflow() {
  state.stickerStack = [];
  state.selectedSticker = null;
  state.isDragging = state.isResizing = state.isRotating = false;
  state.currentStep = 1;
  state.subjectFile = state.processedSubjectBlob = state.sceneImageBlob = state.upscaledImageBlob = state.finalImageWithSwap = null;
  state.activeFilter = 'none';

  if(dom.imagePreview) {
    dom.imagePreview.src = '';
    dom.imagePreview.classList.add('hidden');
  }
  if(dom.subjectUploadPrompt) dom.subjectUploadPrompt.style.display = 'block';

  if(dom.sourceImgPreview) {
    dom.sourceImgPreview.src = '';
    dom.sourceImgPreview.classList.add('hidden');
  }
  if(dom.sourceUploadPrompt) dom.sourceUploadPrompt.style.opacity = '1';
  
  ['fileInput', 'sourceImgInput', 'bgPromptInput', 'captionTextInput'].forEach(id => dom[id] && (dom[id].value = ''));
  
  if(dom.tileDenoisingSlider) dom.tileDenoisingSlider.value = '0.4';
  if(dom.tileDenoisingValue) dom.tileDenoisingValue.textContent = '0.40';
  
  if(dom.memeCanvas) dom.memeCanvas.getContext('2d').clearRect(0, 0, dom.memeCanvas.width, dom.memeCanvas.height);
  if(dom.memeCanvas) dom.memeCanvas.classList.add('hidden');
  
  if(dom.resultImageDisplay) {
    dom.resultImageDisplay.src = '';
    dom.resultImageDisplay.style.filter = 'none';
    dom.resultImageDisplay.classList.remove('hidden');
  }
  if(dom.resultPlaceholder) dom.resultPlaceholder.classList.remove('hidden');
  
  ['downloadBtn', 'addGalleryBtn', 'downloadAnimBtn', 'animFmt', 'shareBtn'].forEach(id => dom[id]?.classList.add('hidden'));
  ['prepareSubjectBtn', 'skipToSwapBtn', 'gotoStep3Btn', 'swapBtn'].forEach(id => dom[id] && (dom[id].disabled = true));
  
  state.sourceFaces = [];
  state.targetFaces = [];
  state.selectedSourceIndex = -1;
  state.selectedTargetIndex = -1;
  state.sourceImageFile = null;

  if (dom.sourceFaceBoxesContainer) dom.sourceFaceBoxesContainer.innerHTML = '';
  if (dom.targetFaceBoxesContainer) dom.targetFaceBoxesContainer.innerHTML = '';
  if (dom.selectionStatus) dom.selectionStatus.classList.add('hidden');
  if (dom.selectedSourceId) dom.selectedSourceId.textContent = 'Nessuno';
  if (dom.selectedTargetId) dom.selectedTargetId.textContent = 'Nessuno';
  
  if(dom.toggleFaceBoxes) dom.toggleFaceBoxes.checked = true;

  if(dom.filterButtonsContainer) {
    dom.filterButtonsContainer.querySelector('.active')?.classList.remove('active');
    dom.filterButtonsContainer.querySelector('[data-filter="none"]')?.classList.add('active');
  }
  
  goToStep(1);
  updateMemePreview();
}