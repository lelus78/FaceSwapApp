import * as api from './api.js';
import { initTheme } from './theme.js';

let currentStep = 1;
let subjectFile = null, processedSubjectBlob = null, sceneImageBlob = null, upscaledImageBlob = null, finalImageWithSwap = null;
let sourceFaces = [], targetFaces = [], selectedSourceIndex = -1, selectedTargetIndex = -1, sourceImageFile = null;
let activeFilter = 'none';
const dom = {};
let stickerStack = [];
let selectedSticker = null;
let isDragging = false, isResizing = false, isRotating = false;
let dragOffsetX, dragOffsetY;

// --- Funzioni di Utilità e UI ---

function displayImage(imageBlobOrFile, imageElement) {
    if (!imageBlobOrFile || !imageElement) return;
    const oldUrl = imageElement.src;
    if (oldUrl && oldUrl.startsWith('blob:')) {
        URL.revokeObjectURL(oldUrl);
    }
    imageElement.src = URL.createObjectURL(imageBlobOrFile);
    imageElement.classList.remove('hidden');
    if (imageElement.id === 'result-image-display') {
        dom.resultPlaceholder.classList.add('hidden');
        dom.downloadBtn.classList.remove('hidden');
        dom.shareBtn.classList.remove('hidden');
    } else if (imageElement.id === 'subject-img-preview') {
        dom.subjectUploadPrompt.style.display = 'none';
    } else if (imageElement.id === 'source-img-preview') {
        dom.sourceUploadPrompt.style.opacity = '0';
    }
}

function drawFaceBoxes(boxesContainer, imageElement, faceArray, selectionType) {
    if (!boxesContainer || !imageElement || !imageElement.complete || imageElement.naturalWidth === 0) return;
    boxesContainer.innerHTML = '';
    const rect = imageElement.getBoundingClientRect();
    if (rect.width === 0) return;
    const parentRect = boxesContainer.getBoundingClientRect();
    const scaleX = rect.width / imageElement.naturalWidth;
    const scaleY = rect.height / imageElement.naturalHeight;
    const offsetX = rect.left - parentRect.left;
    const offsetY = rect.top - parentRect.top;
    faceArray.forEach((face, index) => {
        const [x1, y1, x2, y2] = face.bbox;
        const box = document.createElement('div');
        box.className = 'face-box';
        box.style.left = `${offsetX + (x1 * scaleX)}px`;
        box.style.top = `${offsetY + (y1 * scaleY)}px`;
        box.style.width = `${(x2 - x1) * scaleX}px`;
        box.style.height = `${(y2 - y1) * scaleY}px`;
        box.style.pointerEvents = 'auto';
        const label = document.createElement('span');
        label.className = 'face-box-label';
        label.textContent = index + 1;
        box.appendChild(label);
        box.onclick = (e) => { e.stopPropagation(); handleFaceSelection(index, selectionType); };
        boxesContainer.appendChild(box);
    });
}

function updateSelectionHighlights(container, selectedIndex) {
    if (!container) return;
    container.querySelectorAll('.face-box').forEach((box, i) => box.classList.toggle('selected', i === selectedIndex));
}

function startProgressBar(title, duration = 30) {
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

function finishProgressBar() {
    setTimeout(() => {
        dom.progressBar.style.width = '100%';
        dom.progressText.textContent = '100%';
        setTimeout(() => { dom.progressModal.style.display = 'none'; }, 500);
    }, 200);
}

function showError(title, message) {
    dom.errorModal.style.display = 'flex';
    document.getElementById('error-title').textContent = title;
    document.getElementById('error-message').textContent = message;
}

window.closeModal = (modalId) => document.getElementById(modalId).style.display = 'none';

function goToStep(stepNumber) {
    currentStep = stepNumber;
    document.querySelectorAll('.wizard-step').forEach(step => step.classList.add('hidden'));
    const stepId = `step-${stepNumber}-${['subject', 'scene', 'upscale', 'finalize'][stepNumber - 1]}`;
    document.getElementById(stepId)?.classList.remove('hidden');
}

// --- Funzioni Principali del Workflow ---

async function detectAndDrawFaces(imageBlob, imageElement, boxesContainer, faceArray, selectionType) {
    const onImageLoad = async () => {
        try {
            const data = await api.detectFaces(imageBlob);
            faceArray.splice(0, faceArray.length, ...data.faces);
            drawFaceBoxes(boxesContainer, imageElement, faceArray, selectionType);
            updateSelectionHighlights(boxesContainer, selectionType === 'source' ? selectedSourceIndex : selectedTargetIndex);
        } catch (err) {
            showError('Errore Rilevamento Volti', err.message);
            faceArray.length = 0;
            drawFaceBoxes(boxesContainer, imageElement, [], selectionType);
        }
    };
    if (imageElement.complete && imageElement.naturalWidth > 0) {
        onImageLoad();
    } else {
        imageElement.onload = onImageLoad;
    }
}

function handleFaceSelection(index, type) {
    if (type === 'source') {
        selectedSourceIndex = (selectedSourceIndex === index) ? -1 : index;
    } else {
        selectedTargetIndex = (selectedTargetIndex === index) ? -1 : index;
    }
    dom.selectedSourceId.textContent = selectedSourceIndex > -1 ? `#${selectedSourceIndex + 1}` : 'Nessuno';
    dom.selectedTargetId.textContent = selectedTargetIndex > -1 ? `#${selectedTargetIndex + 1}` : 'Nessuno';
    updateSelectionHighlights(dom.sourceFaceBoxesContainer, selectedSourceIndex);
    updateSelectionHighlights(dom.targetFaceBoxesContainer, selectedTargetIndex);
    dom.swapBtn.disabled = !(selectedSourceIndex > -1 && selectedTargetIndex > -1);
}

function handleSubjectFile(file) {
    if (!file?.type.startsWith('image/')) return;
    subjectFile = file;
    displayImage(file, dom.subjectImgPreview);
    dom.prepareSubjectBtn.disabled = false;
    dom.skipToSwapBtn.disabled = false;
}

function handleSourceFile(file) {
    if (!file?.type.startsWith('image/')) return;
    sourceImageFile = file;
    displayImage(file, dom.sourceImgPreview);
    detectAndDrawFaces(file, dom.sourceImgPreview, dom.sourceFaceBoxesContainer, sourceFaces, 'source');
    dom.selectionStatus.classList.remove('hidden');
}

function handleSkipToSwap() {
    if (!subjectFile) return showError("File Mancante", "Carica un'immagine prima.");
    upscaledImageBlob = subjectFile;
    processedSubjectBlob = sceneImageBlob = finalImageWithSwap = null;
    displayImage(upscaledImageBlob, dom.resultImageDisplay);
    detectAndDrawFaces(upscaledImageBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, targetFaces, 'target');
    goToStep(4);
}

async function handlePrepareSubject() {
    if (!subjectFile) return;
    startProgressBar("Step 1: Preparazione Soggetto...", 15);
    try {
        processedSubjectBlob = await api.prepareSubject(subjectFile);
        displayImage(processedSubjectBlob, dom.resultImageDisplay);
        detectAndDrawFaces(processedSubjectBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, targetFaces, 'target');
        goToStep(2);
    } catch (err) { showError("Errore Preparazione", err.message); } 
    finally { finishProgressBar(); }
}

async function handleCreateScene() {
    if (!dom.bgPromptInput.value || !processedSubjectBlob) return;
    startProgressBar("Step 2: Creazione Scena...", 45);
    try {
        let prompt = dom.bgPromptInput.value;
        const currentImageForEnhancement = sceneImageBlob || processedSubjectBlob; // Immagine più recente nel workflow
        if (dom.autoEnhancePromptToggle.checked && currentImageForEnhancement) {
            prompt = (await api.enhancePrompt(currentImageForEnhancement, prompt)).enhanced_prompt; // Passa l'immagine
            dom.bgPromptInput.value = prompt;
        }
        sceneImageBlob = await api.createScene(processedSubjectBlob, prompt);
        finalImageWithSwap = null;
        displayImage(sceneImageBlob, dom.resultImageDisplay);
        detectAndDrawFaces(sceneImageBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, targetFaces, 'target');
        dom.gotoStep3Btn.disabled = false;
    } catch (err) { showError("Errore Creazione Scena", err.message); } 
    finally { finishProgressBar(); }
}

async function handleUpscaleAndDetail() {
    if (!sceneImageBlob) return;
    startProgressBar("Step 3: Upscale & Detailing...", 90);
    try {
        upscaledImageBlob = await api.upscaleAndDetail(sceneImageBlob, dom.enableHiresUpscaleToggle.checked, dom.tileDenoisingSlider.value);
        finalImageWithSwap = null;
        displayImage(upscaledImageBlob, dom.resultImageDisplay);
        detectAndDrawFaces(upscaledImageBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, targetFaces, 'target');
        goToStep(4);
    } catch (err) { showError("Errore Upscale", err.message); } 
    finally { finishProgressBar(); }
}

async function handlePerformSwap() {
    const targetImg = finalImageWithSwap || upscaledImageBlob;
    if (selectedSourceIndex < 0 || selectedTargetIndex < 0 || !targetImg) return;
    startProgressBar("Step 4: Face Swap...", 10);
    try {
        finalImageWithSwap = await api.performSwap(targetImg, sourceImageFile, selectedSourceIndex, selectedTargetIndex);
        displayImage(finalImageWithSwap, dom.resultImageDisplay);
        detectAndDrawFaces(finalImageWithSwap, dom.resultImageDisplay, dom.targetFaceBoxesContainer, targetFaces, 'target');
        selectedTargetIndex = -1;
        dom.selectedTargetId.textContent = 'Nessuno';
        dom.swapBtn.disabled = true;
    } catch (err) { showError("Errore Face Swap", err.message); } 
    finally { finishProgressBar(); }
}

/**
 * Gestisce il click sul pulsante "Analizza Parti Corpo".
 * Chiama l'API per ottenere le parti rilevabili e poi le renderizza.
 */
async function handleAnalyzeParts() {
    const imageToAnalyze = finalImageWithSwap || upscaledImageBlob || sceneImageBlob || processedSubjectBlob || subjectFile;
    if (!imageToAnalyze) {
        return showError("Immagine Mancante", "Carica un'immagine prima di analizzarla.");
    }

    startProgressBar("🤖 Analisi AI delle parti del corpo...", 15);
    dom.analyzePartsBtn.disabled = true;

    try {
        const result = await api.analyzeParts(imageToAnalyze);
        if (result.parts && result.parts.length > 0) {
            renderDynamicPrompts(result.parts);
            dom.generateAllBtn.classList.remove('hidden');
            dom.generateAllBtn.disabled = false;
        } else {
            showError("Nessuna Parte Trovata", "Il modello non ha rilevato parti umane modificabili in questa immagine.");
        }
    } catch (err) {
        showError("Errore Analisi", err.message);
    } finally {
        finishProgressBar();
        dom.analyzePartsBtn.disabled = false;
    }
}

/**
 * Crea e visualizza dinamicamente i campi di input per ogni parte rilevata.
 * @param {string[]} parts - Un array di nomi di parti, es. ['hair', 'outfit'].
 */
function renderDynamicPrompts(parts) {
    const container = dom.dynamicPromptsContainer;
    container.innerHTML = ''; // Pulisce il contenitore
    
    parts.forEach(part => {
        // Crea l'HTML per ogni riga di prompt
        const prettyPartName = part.charAt(0).toUpperCase() + part.slice(1); // Es. "hair" -> "Hair"
        const promptRow = `
            <div class="flex items-center gap-2">
                <label for="prompt-${part}" class="w-1/4 text-sm text-right text-gray-400">${prettyPartName}:</label>
                <input type="text" id="prompt-${part}" data-part-name="${part}" class="prompt-input w-3/4 bg-gray-900 border border-gray-600 rounded-lg p-2 text-sm text-gray-300" placeholder="Descrivi modifica per ${part}...">
                <button class="enhance-part-btn btn btn-secondary text-white font-bold p-2 rounded-lg" data-part-name="${part}" title="Migliora prompt con AI">✨</button>
            </div>
        `;
        container.innerHTML += promptRow;
    });
}

/**
 * Raccoglie tutti i prompt inseriti dall'utente e avvia la generazione finale multi-parte.
 */
async function handleGenerateAll() {
    const imageToInpaint = finalImageWithSwap || upscaledImageBlob || sceneImageBlob || processedSubjectBlob || subjectFile;
    if (!imageToInpaint) {
        return showError("Immagine Mancante", "Non c'è un'immagine su cui applicare le modifiche.");
    }

    const prompts = {};
    const inputElements = dom.dynamicPromptsContainer.querySelectorAll('.prompt-input');
    
    // Processa tutti i prompt e applica l'enhancement se il toggle è attivo
    for (const input of inputElements) {
        const partName = input.dataset.partName;
        let promptText = input.value.trim();

        if (promptText) {
            if (dom.autoEnhancePromptToggle.checked) { // Usa lo stesso toggle per l'enhancement di tutte le parti
                startProgressBar(`✨ Miglioramento prompt per ${partName}...`, 5); // Breve progresso per ogni prompt
                try {
                    const enhancedResult = await api.enhancePartPrompt(partName, promptText, imageToInpaint); // Passa l'immagine
                    promptText = enhancedResult.enhanced_prompt;
                    input.value = promptText; // Aggiorna il campo con il prompt migliorato
                } catch (err) {
                    console.error(`Errore nel migliorare il prompt per ${partName}:`, err);
                    // Continua con il prompt originale se l'enhancement fallisce
                } finally {
                    finishProgressBar();
                }
            }
            prompts[partName] = promptText;
        }
    }


    if (Object.keys(prompts).length === 0) {
        return showError("Nessun Prompt", "Scrivi una descrizione per almeno una parte per poter generare le modifiche.");
    }

    startProgressBar("🎨 Generazione Multi-Parte in corso...", 120);
    dom.generateAllBtn.disabled = true;

    try {
        // Assicurati che api.generateAllParts chiami l'endpoint corretto nel server.js
        const resultBlob = await api.generateAllParts(imageToInpaint, prompts); 
        finalImageWithSwap = resultBlob;
        displayImage(resultBlob, dom.resultImageDisplay);
        detectAndDrawFaces(resultBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, targetFaces, 'target');
    } catch (err) {
        showError("Errore Generazione", err.message);
    } finally {
        finishProgressBar();
        dom.generateAllBtn.disabled = false;
        // Pulisce i campi di input dopo la generazione
        dom.dynamicPromptsContainer.querySelectorAll('.prompt-input').forEach(inp => inp.value = '');
    }
}

// Aggiungi questa funzione in script.js
async function handleGenerateCaption() {
    const imageToCaption = finalImageWithSwap || upscaledImageBlob || sceneImageBlob || processedSubjectBlob || subjectFile;
    if (!imageToCaption) {
        return showError("Immagine Mancante", "Nessuna immagine nell'anteprima su cui generare una didascalia.");
    }
    startProgressBar("✨ Generazione Didascalia AI...", 10);
    try {
        const selectedTone = dom.toneButtonsContainer.querySelector('.active')?.dataset.tone || 'scherzoso';
        const result = await api.generateCaption(imageToCaption, selectedTone);
        dom.captionTextInput.value = result.caption;
        updateMemePreview(); // Aggiorna il canvas con il nuovo testo
    } catch (err) {
        showError("Errore Generazione Didascalia", err.message);
    } finally {
        finishProgressBar();
    }
}

async function handleDownloadAnimation() {
    const hasAnimatedStickers = stickerStack.some(s => s.type === 'video' || s.type === 'lottie');
    if (!hasAnimatedStickers) {
        showError("Nessuna Animazione", "Non ci sono sticker animati da registrare.");
        return;
    }

    // --- INIZIO DELLA CORREZIONE ---
    // 1. Deseleziona qualsiasi sticker attivo per nascondere il suo riquadro
    selectedSticker = null;
    // 2. Forza un ridisegno del canvas SENZA il box di selezione
    updateMemePreview();
    // 3. Attendi un istante per garantire che il canvas sia aggiornato prima della registrazione
    await new Promise(resolve => setTimeout(resolve, 50));
    // --- FINE DELLA CORREZIONE ---

    startProgressBar("Registrazione animazione...", 5);
    try {
        const stream = dom.memeCanvas.captureStream(30);
        const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
        const chunks = [];
        recorder.ondataavailable = e => e.data.size > 0 && chunks.push(e.data);
        
        recorder.onstop = async () => {
            const webmBlob = new Blob(chunks, { type: 'video/webm' });
            finishProgressBar();
            startProgressBar("Conversione server...", 20);
            try {
                const format = dom.animFmt.value;
                const result = await api.saveResultVideo(webmBlob, format);
                const link = document.createElement('a');
                link.href = result.url;
                link.download = `pro-meme-result.${format}`;
                link.click();
            } catch (serverErr) {
                showError("Errore Server", serverErr.message);
            } finally {
                finishProgressBar();
            }
        };
        
        recorder.start();
        setTimeout(() => recorder.stop(), 5000); // Registra per 5 secondi

    } catch (err) {
        showError("Errore Registrazione", err.message);
        finishProgressBar();
    }
}

// --- Funzioni per il Meme Editor (Canvas) ---

function updateMemePreview() {
    const imageToDrawOn = dom.resultImageDisplay;
    if (!imageToDrawOn.src || !imageToDrawOn.complete || imageToDrawOn.naturalWidth === 0) return;
    const canvas = dom.memeCanvas, ctx = canvas.getContext('2d');
    canvas.width = imageToDrawOn.naturalWidth;
    canvas.height = imageToDrawOn.naturalHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.filter = activeFilter;
    ctx.drawImage(imageToDrawOn, 0, 0);
    ctx.filter = 'none';
    const text = dom.captionTextInput.value;
    const shouldShowCanvas = text || stickerStack.length > 0;
    dom.resultImageDisplay.classList.toggle('hidden', shouldShowCanvas);
    dom.memeCanvas.classList.toggle('hidden', !shouldShowCanvas);
    if (text) drawMemeText(ctx);
    stickerStack.forEach(sticker => drawSticker(ctx, sticker));
}

function drawMemeText(ctx) {
    const { canvas } = ctx;
    const { value: fontFamily } = dom.fontFamilySelect;
    const fontSize = parseInt(dom.fontSizeSlider.value, 10);
    const { value: fontColor } = dom.fontColorInput;
    const { value: strokeColor } = dom.strokeColorInput;
    const position = dom.positionButtons.querySelector('.active').dataset.position;
    const textBg = dom.textBgButtons.querySelector('.active').dataset.bg;
    ctx.font = `${fontSize}px ${fontFamily}`;
    ctx.fillStyle = fontColor;
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = Math.max(1, fontSize / 12);
    ctx.textAlign = 'center';
    const margin = canvas.width * 0.05;
    const maxWidth = canvas.width - (margin * 2);
    const lineHeight = fontSize * 1.2;
    const x = canvas.width / 2;
    const lines = getWrappedLines(ctx, dom.captionTextInput.value, maxWidth);
    ctx.textBaseline = position === 'top' ? 'top' : 'bottom';
    let startY = position === 'top' ? margin : canvas.height - margin - (lines.length - 1) * lineHeight;
    lines.forEach((line, index) => {
        const y = startY + (index * lineHeight);
        if (textBg !== 'none') {
            const metrics = ctx.measureText(line);
            const bgW = metrics.width + (fontSize * 0.5), bgH = lineHeight;
            const bgX = x - (bgW / 2);
            const bgY = position === 'top' ? y - (bgH - fontSize) / 2 : y - fontSize - (bgH - fontSize) / 2;
            ctx.globalAlpha = 0.7; ctx.fillStyle = textBg;
            ctx.fillRect(bgX, bgY, bgW, bgH);
            ctx.globalAlpha = 1.0; ctx.fillStyle = fontColor;
        }
        ctx.strokeText(line, x, y);
        ctx.fillText(line, x, y);
    });
}

function getWrappedLines(ctx, text, maxWidth) {
    if (!text) return [];
    const words = text.split(' '), lines = [];
    let currentLine = words[0] || '';
    for (let i = 1; i < words.length; i++) {
        if (ctx.measureText(`${currentLine} ${words[i]}`).width < maxWidth) {
            currentLine += ` ${words[i]}`;
        } else {
            lines.push(currentLine);
            currentLine = words[i];
        }
    }
    lines.push(currentLine);
    return lines;
}

function drawSticker(ctx, sticker) {
    if (!sticker.element) return;
    ctx.save();
    ctx.translate(sticker.x + sticker.width / 2, sticker.y + sticker.height / 2);
    ctx.rotate(sticker.rotation);
    ctx.drawImage(sticker.element, -sticker.width / 2, -sticker.height / 2, sticker.width, sticker.height);
    ctx.restore();
    if (sticker === selectedSticker) {
        ctx.save();
        ctx.translate(sticker.x + sticker.width / 2, sticker.y + sticker.height / 2);
        ctx.rotate(sticker.rotation);
        ctx.strokeStyle = '#007bff'; ctx.fillStyle = '#007bff'; ctx.lineWidth = 4;
        const handleSize = 12;
        ctx.strokeRect(-sticker.width / 2, -sticker.height / 2, sticker.width, sticker.height);
        ctx.fillRect(sticker.width / 2 - handleSize / 2, sticker.height / 2 - handleSize / 2, handleSize, handleSize);
        ctx.beginPath(); ctx.moveTo(0, -sticker.height / 2); ctx.lineTo(0, -sticker.height / 2 - 20); ctx.stroke();
        ctx.beginPath(); ctx.arc(0, -sticker.height / 2 - 25, handleSize / 2, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
    }
}

async function addStickerToCanvas(element, isVideo, isLottie, path) {
    let stickerData = { type: isVideo ? 'video' : (isLottie ? 'lottie' : 'image'), x: 20, y: 20, rotation: 0 };
    let stickerElement, naturalW, naturalH;
    if (isLottie) {
        try {
            const animationData = await (await fetch(`/lottie_json/${path.replace('static/', '')}`)).json();
            naturalW = animationData.w; naturalH = animationData.h;
            stickerElement = document.createElement('canvas');
            stickerElement.width = naturalW; stickerElement.height = naturalH;
            lottie.loadAnimation({
                renderer: 'canvas', loop: true, autoplay: true, animationData,
                rendererSettings: { context: stickerElement.getContext('2d'), clearCanvas: true },
            });
        } catch (err) { return showError("Errore Lottie", "Impossibile caricare l'animazione."); }
    } else {
        stickerElement = element;
        naturalW = element.naturalWidth || element.videoWidth;
        naturalH = element.naturalHeight || element.videoHeight;
    }
    if (naturalW > 0) {
        stickerData.element = stickerElement;
        stickerData.aspectRatio = naturalH / naturalW;
        stickerData.width = 150;
        stickerData.height = 150 * stickerData.aspectRatio;
        stickerStack.push(stickerData);
        selectedSticker = stickerData;
        if (isVideo || isLottie) {
            dom.downloadAnimBtn.classList.remove('hidden');
            dom.animFmt.classList.remove('hidden');
        }
    }
}

// --- Funzioni di Setup e Inizializzazione ---

function animationLoop() {
    updateMemePreview();
    requestAnimationFrame(animationLoop);
}

function assignDomElements() {
    const ids = [
        'error-modal', 'progress-modal', 'progress-bar', 'progress-text', 'progress-title',
        'result-image-display', 'result-placeholder', 'meme-canvas',
        'download-btn', 'download-anim-btn', 'anim-fmt', 'share-btn', 'reset-all-btn',
        'step-1-subject', 'subject-img-input', 'subject-img-preview', 'subject-upload-prompt',
        'prepare-subject-btn', 'skip-to-swap-btn',
        'step-2-scene', 'bg-prompt-input', 'auto-enhance-prompt-toggle', 'generate-scene-btn', 'goto-step-3-btn',
        'step-3-upscale', 'enable-hires-upscale-toggle', 'tile-denoising-slider', 'tile-denoising-value',
        'start-upscale-btn', 'skip-upscale-btn',
        'step-4-finalize', 'source-img-input', 'source-img-preview', 'source-upload-prompt',
        'toggle-face-boxes', 'source-face-boxes-container', 'target-face-boxes-container',
        'selection-status', 'selected-source-id', 'selected-target-id', 'swap-btn', 'back-to-step-3-btn',
        'filter-buttons-container',
        'sticker-section', 'sticker-search-input', 'sticker-gallery', 'sticker-delete-btn',
        'sticker-front-btn', 'sticker-back-btn',
        'meme-section', 'caption-text-input', 'caption-btn', 'tone-buttons-container',
        'font-family-select', 'font-size-slider', 'font-size-value',
        'font-color-input', 'stroke-color-input', 'position-buttons', 'text-bg-buttons',
        'inpainting-prompt-input', 'generate-hair-btn',
        'analyze-parts-btn',
        'dynamic-prompts-container',
        'generate-all-btn',
        'theme-toggle', 'theme-icon'

    ];
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (!el) console.warn(`Elemento non trovato: #${id}`);
        dom[id.replace(/-(\w)/g, (_, c) => c.toUpperCase())] = el;
    });
}

function setupEventListeners() {
    dom.resetAllBtn.addEventListener('click', resetWorkflow);
    dom.subjectImgInput.addEventListener('change', (e) => handleSubjectFile(e.target.files[0]));
    dom.prepareSubjectBtn.addEventListener('click', handlePrepareSubject);
    dom.skipToSwapBtn.addEventListener('click', handleSkipToSwap);
    dom.generateSceneBtn.addEventListener('click', handleCreateScene);
    dom.gotoStep3Btn.addEventListener('click', () => goToStep(3));
    dom.startUpscaleBtn.addEventListener('click', handleUpscaleAndDetail);
    dom.captionBtn.addEventListener('click', handleGenerateCaption);
    dom.tileDenoisingSlider.addEventListener('input', e => dom.tileDenoisingValue.textContent = parseFloat(e.target.value).toFixed(2));
    dom.skipUpscaleBtn.addEventListener('click', () => {
        upscaledImageBlob = sceneImageBlob;
        displayImage(upscaledImageBlob, dom.resultImageDisplay);
        detectAndDrawFaces(upscaledImageBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, targetFaces, 'target');
        goToStep(4);
    });
    dom.sourceImgInput.addEventListener('change', (e) => handleSourceFile(e.target.files[0]));
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
            activeFilter = e.target.dataset.filter;
            dom.resultImageDisplay.style.filter = activeFilter; // Applica il filtro anche all'immagine base
            updateMemePreview();
        }
});
    // Listener per il pulsante statico "Analizza Parti"
    dom.analyzePartsBtn.addEventListener('click', handleAnalyzeParts);
    
    // Listener per il pulsante statico "Genera Modifiche"
    dom.generateAllBtn.addEventListener('click', handleGenerateAll);

    // Listener per i pulsanti "Migliora" creati dinamicamente
    // Usiamo l'event delegation per gestire i click su elementi che ancora non esistono.
    dom.dynamicPromptsContainer.addEventListener('click', async (e) => {
        // Controlla se l'elemento cliccato è un pulsante per migliorare il prompt
        if (e.target && e.target.classList.contains('enhance-part-btn')) {
            const button = e.target;
            const partName = button.dataset.partName;
            const input = document.getElementById(`prompt-${partName}`);
            const imageForEnhancement = finalImageWithSwap || upscaledImageBlob || sceneImageBlob || processedSubjectBlob || subjectFile;
            
            if (!input || !input.value) {
                return showError("Prompt Vuoto", "Scrivi prima un'idea da migliorare.");
            }
            if (!imageForEnhancement) {
                return showError("Immagine Mancante", "Carica un'immagine per migliorare il prompt.");
            }
            
            button.disabled = true;
            startProgressBar(`✨ Miglioramento prompt per ${partName}...`, 10);
            
            try {
                const result = await api.enhancePartPrompt(partName, input.value, imageForEnhancement); // Passa l'immagine
                input.value = result.enhanced_prompt;
            } catch (err) {
                showError("Errore Miglioramento Prompt", err.message);
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
        if (selectedSticker) stickerStack = stickerStack.filter(s => s !== selectedSticker);
        selectedSticker = null;
    });
    stickerControls[1].addEventListener('click', () => {
        if (!selectedSticker) return;
        const i = stickerStack.indexOf(selectedSticker);
        if (i < stickerStack.length - 1) { stickerStack.splice(i, 1); stickerStack.push(selectedSticker); }
    });
    stickerControls[2].addEventListener('click', () => {
        if (!selectedSticker) return;
        const i = stickerStack.indexOf(selectedSticker);
        if (i > 0) { stickerStack.splice(i, 1); stickerStack.unshift(selectedSticker); }
    });
    dom.downloadBtn.addEventListener('click', async e => {
        e.preventDefault();
        selectedSticker = null;
        updateMemePreview();
        await new Promise(r => setTimeout(r, 50));
        const dataUrl = dom.memeCanvas.classList.contains('hidden') ? dom.resultImageDisplay.src : dom.memeCanvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = 'pro-meme-result.png';
        link.click();
    });
    dom.downloadAnimBtn.addEventListener('click', handleDownloadAnimation);
    
    const getCoords = e => {
    const rect = dom.memeCanvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    // CORREZIONE: usiamo 'dom.memeCanvas' che è sempre disponibile
    return { x: (clientX - rect.left) * (dom.memeCanvas.width / rect.width), y: (clientY - rect.top) * (dom.memeCanvas.height / rect.height) };
    };
    const onStart = e => {
        const hit = getStickerAtPosition(getCoords(e).x, getCoords(e).y);
        if (hit) {
            e.preventDefault();
            selectedSticker = hit.sticker;
            if (hit.corner === 'resize') isResizing = true;
            else if (hit.corner === 'rotate') isRotating = true;

            else { isDragging = true; const c = getCoords(e); dragOffsetX = c.x - hit.sticker.x; dragOffsetY = c.y - hit.sticker.y; }
        } else {
            selectedSticker = null;
        }
        stickerControls.forEach(b => b.disabled = !selectedSticker);
    };
    const onMove = e => {
        if (!selectedSticker || !(isDragging || isResizing || isRotating)) return;
        e.preventDefault();
        const { x, y } = getCoords(e);
        const s = selectedSticker, cx = s.x + s.width / 2, cy = s.y + s.height / 2;
        if (isResizing) {
            const newW = Math.sqrt(Math.pow(x - cx, 2) + Math.pow(y - cy, 2)) * Math.sqrt(2);
            if (newW > 20) { s.x += (s.width - newW) / 2; s.y += (s.height - newW * s.aspectRatio) / 2; s.width = newW; s.height = newW * s.aspectRatio; }
        } else if (isRotating) {
            s.rotation = Math.atan2(y - cy, x - cx) + Math.PI / 2;
        } else if (isDragging) {
            s.x = x - dragOffsetX; s.y = y - dragOffsetY;
        }
    };
    const onEnd = () => { isDragging = isResizing = isRotating = false; };
    ['mousedown', 'touchstart'].forEach(evt => dom.memeCanvas.addEventListener(evt, onStart, { passive: false }));
    ['mousemove', 'touchmove'].forEach(evt => document.addEventListener(evt, onMove, { passive: false }));
    ['mouseup', 'touchend', 'touchcancel'].forEach(evt => document.addEventListener(evt, onEnd));
}

function resetWorkflow() {
    stickerStack = []; selectedSticker = null; isDragging = isResizing = isRotating = false;
    currentStep = 1; subjectFile = processedSubjectBlob = sceneImageBlob = upscaledImageBlob = finalImageWithSwap = null;
    activeFilter = 'none';
    dom.subjectImgPreview.src = ''; dom.subjectImgPreview.classList.add('hidden');
    dom.subjectUploadPrompt.style.display = 'block';
    dom.sourceImgPreview.src = ''; dom.sourceImgPreview.classList.add('hidden');
    dom.sourceUploadPrompt.style.opacity = '1';
    ['subjectImgInput', 'sourceImgInput', 'bgPromptInput', 'captionTextInput'].forEach(id => dom[id] && (dom[id].value = ''));
    dom.tileDenoisingSlider.value = '0.4';
    dom.tileDenoisingValue.textContent = '0.40';
    dom.memeCanvas.getContext('2d').clearRect(0, 0, dom.memeCanvas.width, dom.memeCanvas.height);
    dom.memeCanvas.classList.add('hidden');
    dom.resultImageDisplay.src = '';
    dom.resultImageDisplay.style.filter = 'none';
    dom.resultImageDisplay.classList.remove('hidden');
    dom.resultPlaceholder.classList.remove('hidden');
    ['downloadBtn', 'downloadAnimBtn', 'animFmt', 'shareBtn'].forEach(id => dom[id]?.classList.add('hidden'));
    ['prepareSubjectBtn', 'skipToSwapBtn', 'gotoStep3Btn', 'swapBtn'].forEach(id => dom[id] && (dom[id].disabled = true));
    sourceFaces = []; targetFaces = []; selectedSourceIndex = -1; selectedTargetIndex = -1; sourceImageFile = null;
    drawFaceBoxes(dom.sourceFaceBoxesContainer, dom.sourceImgPreview, [], 'source');
    drawFaceBoxes(dom.targetFaceBoxesContainer, dom.resultImageDisplay, [], 'target');
    dom.selectionStatus.classList.add('hidden');
    dom.selectedSourceId.textContent = 'Nessuno';
    dom.selectedTargetId.textContent = 'Nessuno';
    dom.toggleFaceBoxes.checked = true;
    if (dom.sourceFaceBoxesContainer) dom.sourceFaceBoxesContainer.innerHTML = '';
    if (dom.targetFaceBoxesContainer) dom.targetFaceBoxesContainer.innerHTML = '';
    dom.subjectImgPreview.src = ''; dom.subjectImgPreview.classList.add('hidden');
    dom.targetFaceBoxesContainer.style.display = 'block';
    dom.sourceFaceBoxesContainer.style.display = 'block';
    dom.filterButtonsContainer.querySelector('.active')?.classList.remove('active');
    dom.filterButtonsContainer.querySelector('[data-filter="none"]')?.classList.add('active');
    goToStep(1);
    updateMemePreview();
}

async function loadStickers() {
    const gallery = dom.stickerGallery;
    gallery.innerHTML = '<p class="text-gray-500">Caricamento...</p>';
    try {
        const categories = await api.getStickers();
        gallery.innerHTML = '';
        if (categories.length === 0) return gallery.innerHTML = '<p class="text-gray-500">Nessuno sticker trovato.</p>';
        categories.forEach(category => {
            const title = document.createElement('h4');
            title.className = 'font-bold text-sm text-blue-400 mt-2 mb-1 px-1 category-title w-full';
            title.textContent = category.category;
            gallery.appendChild(title);
            const container = document.createElement('div');
            container.className = 'flex flex-wrap gap-2 sticker-container';
            gallery.appendChild(container);
            category.stickers.forEach(path => {
                const isVideo = path.endsWith('.webm'), isLottie = path.endsWith('.tgs');
                const wrapper = document.createElement('div');
                wrapper.className = 'sticker-item-wrapper relative';
                let el;
                if (isLottie) {
                    el = document.createElement('div');
                    lottie.loadAnimation({ container: el, renderer: 'svg', loop: true, autoplay: true, path: `/lottie_json/${path.replace('static/', '')}` });
                } else {
                    el = isVideo ? document.createElement('video') : document.createElement('img');
                    if (isVideo) { el.autoplay = el.muted = el.loop = el.playsInline = true; } 
                    else { el.crossOrigin = "anonymous"; }
                    el.src = path;
                }
                el.className = 'h-20 w-auto object-contain p-1 bg-gray-700 rounded-md cursor-pointer hover:bg-blue-600 transition-all sticker-item';
                wrapper.appendChild(el);
                wrapper.addEventListener('click', () => addStickerToCanvas(el, isVideo, isLottie, path));
                container.appendChild(wrapper);
            });
        });
    } catch (err) {
        gallery.innerHTML = '<p class="text-red-500">Errore nel caricamento.</p>';
        showError("Errore Sticker", err.message);
    }
}

function getStickerAtPosition(x, y) {
    const handleRadius = (window.matchMedia?.(`(pointer: coarse)`).matches) ? 40 : 20;
    for (let i = stickerStack.length - 1; i >= 0; i--) {
        const s = stickerStack[i], cx = s.x + s.width / 2, cy = s.y + s.height / 2;
        const rotHandle = rotatePoint(cx, cy - s.height / 2 - 25, cx, cy, s.rotation);
        const resizeHandle = rotatePoint(s.x + s.width, s.y + s.height, cx, cy, s.rotation);
        if (distance(x, y, rotHandle.x, rotHandle.y) < handleRadius) return { sticker: s, corner: 'rotate' };
        if (distance(x, y, resizeHandle.x, resizeHandle.y) < handleRadius) return { sticker: s, corner: 'resize' };
        if (isPointInRotatedRectangle({ x, y }, s)) return { sticker: s, corner: 'drag' };
    }
    return null;
}

function isPointInRotatedRectangle(point, rect) {
    const cx = rect.x + rect.width / 2, cy = rect.y + rect.height / 2;
    const { x, y } = rotatePoint(point.x, point.y, cx, cy, -rect.rotation);
    return x > rect.x && x < rect.x + rect.width && y > rect.y && y < rect.y + rect.height;
}

function rotatePoint(x, y, cx, cy, angle) {
    const cos = Math.cos(angle), sin = Math.sin(angle);
    return { x: (cos * (x - cx)) - (sin * (y - cy)) + cx, y: (sin * (x - cx)) + (cos * (y - cy)) + cy };
}

function distance(x1, y1, x2, y2) {
    return Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2));
}


document.addEventListener('DOMContentLoaded', () => {
    assignDomElements();
    initTheme(dom.themeToggle, dom.themeIcon);
    setupEventListeners();
    loadStickers();
    resetWorkflow();
    animationLoop();
});
