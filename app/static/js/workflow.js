// Contenuto completo e corretto per app/static/js/workflow.js

import * as api from './api.js';
import { state, dom } from './state.js';
import { updateMemePreview, handleDownloadAnimation } from './memeEditor.js';
import { getStickerAtPosition } from './stickers.js';
import { addToGallery } from './gallery.js';
import { drawFaceBoxes, updateSelectionHighlights, refreshFaceBoxes, detectAndDrawFaces } from "./facebox.js";

export function displayImage(src, imageElement, onLoadCallback = null) {
  if (!src || !imageElement) return;
  const oldUrl = imageElement.dataset.blobUrl;
  if (oldUrl) { URL.revokeObjectURL(oldUrl); imageElement.dataset.blobUrl = ''; }

  const finalize = url => {
    imageElement.onload = () => {
      refreshFaceBoxes();
      if (onLoadCallback) {
        onLoadCallback();
      }
    };
    imageElement.src = url;
    imageElement.classList.remove('hidden');
    if (imageElement.id === 'result-image-display') {
      dom.resultPlaceholder.classList.add('hidden');
      dom.downloadBtn.classList.remove('hidden');
      dom.addGalleryBtn.classList.remove('hidden');
      dom.shareBtn.classList.remove('hidden');
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
  dom.errorModal.style.display = 'flex';
  document.getElementById('error-title').textContent = title;
  document.getElementById('error-message').textContent = message;
  dom.errorModal.querySelector('button')?.focus();
}

export function goToStep(stepNumber) {
  state.currentStep = stepNumber;
  document.querySelectorAll('.wizard-step').forEach(step => step.classList.add('hidden'));
  const stepId = `step-${stepNumber}-${['subject', 'scene', 'upscale', 'finalize'][stepNumber - 1]}`;
  document.getElementById(stepId)?.classList.remove('hidden');
}
export function handleSubjectFile(file) {
  if (!file?.type.startsWith('image/')) return;
  state.subjectFile = file;
  displayImage(file, dom.imagePreview, () => {
      // Questa callback viene eseguita solo quando l'immagine è pronta
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
  if (!state.subjectFile) return showError('File Mancante', "Carica un'immagine prima.");
  state.upscaledImageBlob = state.subjectFile;
  state.processedSubjectBlob = state.sceneImageBlob = state.finalImageWithSwap = null;
  displayImage(state.upscaledImageBlob, dom.resultImageDisplay, () => {
    detectAndDrawFaces(state.upscaledImageBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, state.targetFaces, 'target');
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

export async function handleCreateScene() {
  if (!dom.bgPromptInput.value || !state.processedSubjectBlob) return;
  startProgressBar('Step 2: Creazione Scena...', 45);
  try {
    let prompt = dom.bgPromptInput.value;
    const currentImage = state.sceneImageBlob || state.processedSubjectBlob;
    if (dom.autoEnhancePromptToggle.checked && currentImage) {
      prompt = (await api.enhancePrompt(currentImage, prompt)).enhanced_prompt;
      dom.bgPromptInput.value = prompt;
    }
    state.sceneImageBlob = await api.createScene(state.processedSubjectBlob, prompt);
    state.finalImageWithSwap = null;
    displayImage(state.sceneImageBlob, dom.resultImageDisplay, () => {
        detectAndDrawFaces(state.sceneImageBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, state.targetFaces, 'target');
    });
    dom.gotoStep3Btn.disabled = false;
  } catch (err)
 {
    showError('Errore Creazione Scena', err.message);
  } finally {
    finishProgressBar();
  }
}

export async function handleUpscaleAndDetail() {
  if (!state.sceneImageBlob) return;
  startProgressBar('Step 3: Upscale & Detailing...', 90);
  try {
    state.upscaledImageBlob = await api.upscaleAndDetail(state.sceneImageBlob, dom.enableHiresUpscaleToggle.checked, dom.tileDenoisingSlider.value);
    state.finalImageWithSwap = null;
    displayImage(state.upscaledImageBlob, dom.resultImageDisplay, () => {
        detectAndDrawFaces(state.upscaledImageBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, state.targetFaces, 'target');
    });
    goToStep(4);
  } catch (err) {
    showError('Errore Upscale', err.message);
  } finally {
    finishProgressBar();
  }
}

export async function handlePerformSwap() {
  const targetImg = state.finalImageWithSwap || state.upscaledImageBlob;
  if (state.selectedSourceIndex < 0 || state.selectedTargetIndex < 0 || !targetImg) return;
  startProgressBar('Step 4: Face Swap...', 10);
  try {
    state.finalImageWithSwap = await api.performSwap(targetImg, state.sourceImageFile, state.selectedSourceIndex, state.selectedTargetIndex);
    displayImage(state.finalImageWithSwap, dom.resultImageDisplay, () => {
        detectAndDrawFaces(state.finalImageWithSwap, dom.resultImageDisplay, dom.targetFaceBoxesContainer, state.targetFaces, 'target');
    });
    state.selectedTargetIndex = -1;
    dom.selectedTargetId.textContent = 'Nessuno';
    dom.swapBtn.disabled = true;
  } catch (err) {
    showError('Errore Face Swap', err.message);
  } finally {
    finishProgressBar();
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
      showError('Nessuna Parte Trovata', 'Il modello non ha rilevato parti umane modificabili in questa immagine.');
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
    let promptText = input.value.trim();
    if (promptText) {
      if (dom.autoEnhancePromptToggle.checked) {
        startProgressBar(`✨ Miglioramento prompt per ${partName}...`, 5);
        try {
          const enhanced = await api.enhancePartPrompt(partName, promptText, imageToInpaint);
          promptText = enhanced.enhanced_prompt;
          input.value = promptText;
        } catch (err) {
          console.error('Errore nel migliorare il prompt per', partName, err);
        } finally {
          finishProgressBar();
        }
      }
      prompts[partName] = promptText;
    }
  }
  if (Object.keys(prompts).length === 0) return showError('Nessun Prompt', 'Scrivi una descrizione per almeno una parte.');
  startProgressBar('🎨 Generazione Multi-Parte in corso...', 120);
  dom.generateAllBtn.disabled = true;
  try {
    const resultBlob = await api.generateAllParts(imageToInpaint, prompts);
    state.finalImageWithSwap = resultBlob;
    displayImage(resultBlob, dom.resultImageDisplay);
    detectAndDrawFaces(resultBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, state.targetFaces, 'target');
  } catch (err) {
    showError('Errore Generazione', err.message);
  } finally {
    finishProgressBar();
    dom.generateAllBtn.disabled = false;
    dom.dynamicPromptsContainer.querySelectorAll('.prompt-input').forEach(inp => inp.value = '');
  }
}

export async function handleGenerateCaption() {
  const imageToCaption = state.finalImageWithSwap || state.upscaledImageBlob || state.sceneImageBlob || state.processedSubjectBlob || state.subjectFile;
  if (!imageToCaption) return showError('Immagine Mancante', 'Nessuna immagine nell\'anteprima su cui generare una didascalia.');
  startProgressBar('✨ Generazione Didascalia AI...', 10);
  try {
    const tone = dom.toneButtonsContainer.querySelector('.active')?.dataset.tone || 'scherzoso';
    const result = await api.generateCaption(imageToCaption, tone);
    dom.captionTextInput.value = result.caption;
    updateMemePreview();
  } catch (err) {
    showError('Errore Generazione Didascalia', err.message);
  } finally {
    finishProgressBar();
  }
}

export async function handleShare() {
  updateMemePreview();
  await new Promise(r => setTimeout(r, 50));
  const url = dom.memeCanvas.classList.contains('hidden')
    ? dom.resultImageDisplay.src
    : dom.memeCanvas.toDataURL('image/png');
  if (!navigator.share) {
    showError('Condivisione non supportata', 'Scarica l\'immagine e condividila manualmente.');
    return;
  }
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
  dom.startUpscaleBtn.addEventListener('click', handleUpscaleAndDetail);
  dom.captionBtn.addEventListener('click', handleGenerateCaption);
  dom.tileDenoisingSlider.addEventListener('input', e => dom.tileDenoisingValue.textContent = parseFloat(e.target.value).toFixed(2));
  dom.skipUpscaleBtn.addEventListener('click', () => {
    state.upscaledImageBlob = state.sceneImageBlob;
    displayImage(state.upscaledImageBlob, dom.resultImageDisplay);
    detectAndDrawFaces(state.upscaledImageBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, state.targetFaces, 'target');
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
  ['captionTextInput', 'fontFamilySelect', 'fontColorInput', 'strokeColorInput'].forEach(id => dom[id].addEventListener('input', updateMemePreview));
  dom.fontSizeSlider.addEventListener('input', () => { dom.fontSizeValue.textContent = dom.fontSizeSlider.value; updateMemePreview(); });
  [dom.positionButtons, dom.textBgButtons, dom.toneButtonsContainer].forEach(container => {
    container.addEventListener('click', e => {
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
        const match = wrapper.querySelector('.sticker-item').src.toLowerCase().includes(term) || title.textContent.toLowerCase().includes(term);
        wrapper.style.display = match ? 'inline-block' : 'none';
        if (match) categoryVisible = true;
      });
      title.style.display = categoryVisible ? 'block' : 'none';
    });
  });
  const stickerControls = [dom.stickerDeleteBtn, dom.stickerFrontBtn, dom.stickerBackBtn];
  stickerControls[0].addEventListener('click', () => {
    if (state.selectedSticker) state.stickerStack = state.stickerStack.filter(s => s !== state.selectedSticker);
    state.selectedSticker = null;
  });
  stickerControls[1].addEventListener('click', () => {
    if (!state.selectedSticker) return;
    const i = state.stickerStack.indexOf(state.selectedSticker);
    if (i < state.stickerStack.length - 1) { state.stickerStack.splice(i, 1); state.stickerStack.push(state.selectedSticker); }
  });
  stickerControls[2].addEventListener('click', () => {
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
  dom.addGalleryBtn.addEventListener('click', () => {
    updateMemePreview();
    const src = dom.memeCanvas.toDataURL('image/png');
    const data = JSON.parse(localStorage.getItem('galleryData') || '{}');
    const user = localStorage.getItem('username') || 'user';
    let count = 0;
    const u = data[user] || {};
    Object.values(u).forEach(tags => {
      Object.values(tags).forEach(arr => { count += arr.length; });
    });
    const title = `Meme #${count + 1}`;
    addToGallery(title, src, dom.captionTextInput.value, [], false);
  });
  dom.shareBtn.addEventListener('click', handleShare);
  dom.downloadAnimBtn.addEventListener('click', handleDownloadAnimation);
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
    stickerControls.forEach(b => b.disabled = !state.selectedSticker);
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
  window.addEventListener('resize', refreshFaceBoxes);
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(refreshFaceBoxes);
    [dom.resultImageDisplay, dom.sourceImgPreview].forEach(el => el && ro.observe(el));
  }
}

export function resetWorkflow() {
  state.stickerStack = [];
  state.selectedSticker = null;
  state.isDragging = state.isResizing = state.isRotating = false;
  state.currentStep = 1;
  state.subjectFile = state.processedSubjectBlob = state.sceneImageBlob = state.upscaledImageBlob = state.finalImageWithSwap = null;
  state.activeFilter = 'none';
  dom.imagePreview.src = '';
  dom.imagePreview.classList.add('hidden');
  dom.subjectUploadPrompt.style.display = 'block';
  dom.sourceImgPreview.src = '';
  dom.sourceImgPreview.classList.add('hidden');
  dom.sourceUploadPrompt.style.opacity = '1';
  ['fileInput', 'sourceImgInput', 'bgPromptInput', 'captionTextInput'].forEach(id => dom[id] && (dom[id].value = ''));
  dom.tileDenoisingSlider.value = '0.4';
  dom.tileDenoisingValue.textContent = '0.40';
  dom.memeCanvas.getContext('2d').clearRect(0, 0, dom.memeCanvas.width, dom.memeCanvas.height);
  dom.memeCanvas.classList.add('hidden');
  dom.resultImageDisplay.src = '';
  dom.resultImageDisplay.style.filter = 'none';
  dom.resultImageDisplay.classList.remove('hidden');
  dom.resultPlaceholder.classList.remove('hidden');
  ['downloadBtn', 'addGalleryBtn', 'downloadAnimBtn', 'animFmt', 'shareBtn'].forEach(id => dom[id]?.classList.add('hidden'));
  ['prepareSubjectBtn', 'skipToSwapBtn', 'gotoStep3Btn', 'swapBtn'].forEach(id => dom[id] && (dom[id].disabled = true));
  state.sourceFaces = [];
  state.targetFaces = [];
  state.selectedSourceIndex = -1;
  state.selectedTargetIndex = -1;
  state.sourceImageFile = null;
  drawFaceBoxes(dom.sourceFaceBoxesContainer, dom.sourceImgPreview, [], 'source');
  drawFaceBoxes(dom.targetFaceBoxesContainer, dom.resultImageDisplay, [], 'target');
  dom.selectionStatus.classList.add('hidden');
  dom.selectedSourceId.textContent = 'Nessuno';
  dom.selectedTargetId.textContent = 'Nessuno';
  dom.toggleFaceBoxes.checked = true;
  if (dom.sourceFaceBoxesContainer) dom.sourceFaceBoxesContainer.innerHTML = '';
  if (dom.targetFaceBoxesContainer) dom.targetFaceBoxesContainer.innerHTML = '';
  dom.imagePreview.src = '';
  dom.imagePreview.classList.add('hidden');
  dom.targetFaceBoxesContainer.style.display = 'block';
  dom.sourceFaceBoxesContainer.style.display = 'block';
  dom.filterButtonsContainer.querySelector('.active')?.classList.remove('active');
  dom.filterButtonsContainer.querySelector('[data-filter="none"]')?.classList.add('active');
  goToStep(1);
  updateMemePreview();
}
