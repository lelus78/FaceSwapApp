import * as api from './api.js';

let currentStep = 1;
let subjectFile = null, processedSubjectBlob = null, sceneImageBlob = null, upscaledImageBlob = null, finalImageWithSwap = null;
let sourceFaces = [], targetFaces = [], selectedSourceIndex = -1, selectedTargetIndex = -1, sourceImageFile = null;
let activeFilter = 'none';
const dom = {};
let stickerStack = [];
let selectedSticker = null;
let isDragging = false;
let isResizing = false;
let isRotating = false;
let dragOffsetX, dragOffsetY;

// ... (Tutte le funzioni da displayImage a handlePerformSwap rimangono invariate) ...
function displayImage(imageBlobOrFile, imageElement) {
    if (!imageBlobOrFile || !imageElement) return;
    const oldUrl = imageElement.src;
    if (oldUrl && oldUrl.startsWith('blob:')) {
        URL.revokeObjectURL(oldUrl);
    }
    const imageUrl = URL.createObjectURL(imageBlobOrFile);
    imageElement.src = imageUrl;
    imageElement.classList.remove('hidden');

    if (imageElement.id === 'result-image-display') {
        dom.resultPlaceholder.classList.add('hidden');
        dom.downloadBtn.classList.remove('hidden');
        if (navigator.share && navigator.canShare) {
            
// --- Gestione share / copia link ---
if (navigator.share && navigator.canShare) {
    // HTTPS / localhost ➜ share nativo
    // --- Gestione share / copia link ---
// --- Gestione share / copia link ---
// Prima rimuoviamo QUALSIASI classe 'hidden' responsive (sm:hidden, md:hidden, ecc.)
dom.shareBtn.className = dom.shareBtn.className.split(' ').filter(c => !c.includes('hidden')).join(' ') || '';
dom.shareBtn.style.display = 'inline-flex';

if (navigator.share && navigator.canShare) {
    dom.shareBtn.textContent = 'Condividi';
    dom.shareBtn.onclick = null;
} else {
    dom.shareBtn.textContent = 'Copia link';
    dom.shareBtn.onclick = () => {
        const url = window.location.href;
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(url)
                .then(() => alert('Link copiato negli appunti!'))
                .catch(() => alert('Copia non riuscita: copia manualmente il link dalla barra indirizzi.'));
        } else {
            prompt('Copia manualmente il link:', url);
        }
    };
}
dom.shareBtn.style.display = 'inline-flex';

if (navigator.share && navigator.canShare) {
    // HTTPS / localhost ➜ share nativo
    dom.shareBtn.textContent = 'Condividi';
    dom.shareBtn.onclick = null; // Usa listener nativo definito in setupEventListeners
} else {
    // HTTP ➜ fallback: copia link
    dom.shareBtn.textContent = 'Copia link';
    dom.shareBtn.onclick = () => {
        const url = window.location.href;
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(url)
                .then(() => alert('Link copiato negli appunti!'))
                .catch(() => alert('Copia non riuscita: copia manualmente il link dalla barra indirizzi.'));
        } else {
            prompt('Copia manualmente il link:', url);
        }
    };
}
    dom.shareBtn.textContent = 'Condividi';
    dom.shareBtn.onclick = null; // usa il listener nativo aggiunto più avanti
} else {
    // HTTP ➜ fallback: copia link
    dom.shareBtn.classList.remove('hidden');
    dom.shareBtn.textContent = 'Copia link';
    dom.shareBtn.onclick = () => {
        const url = window.location.href;
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(url)
                .then(() => alert('Link copiato negli appunti!'))
                .catch(() => alert('Copia non riuscita: copia manualmente il link.'));
        } else {
            prompt('Copia manualmente il link:', url);
        }
    };
}
        }
    } else if (imageElement.id === 'subject-img-preview') {
        dom.subjectUploadPrompt.style.display = 'none';
    } else if (imageElement.id === 'source-img-preview') {
        dom.sourceUploadPrompt.style.opacity = '0';
    }
}

function drawFaceBoxes(boxesContainer, imageElement, faceArray, selectionType) {
    if (!boxesContainer || !imageElement || !imageElement.complete) return;
    boxesContainer.innerHTML = '';
    
    const rect = imageElement.getBoundingClientRect();
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
    const boxes = container.querySelectorAll('.face-box');
    boxes.forEach((box, i) => { box.classList.toggle('selected', i === selectedIndex); });
}

function startProgressBar(title, durationInSeconds = 30) {
    dom.progressTitle.textContent = title; dom.progressModal.style.display = 'flex';
    dom.progressBar.style.width = '0%'; dom.progressText.textContent = '0%'; let progress = 0;
    const interval = setInterval(() => {
        progress++; const progressPercentage = Math.min(progress, 95);
        if (dom.progressBar) dom.progressBar.style.width = `${progressPercentage}%`;
        if (dom.progressText) dom.progressText.textContent = `${progressPercentage}%`;
        if (progress >= 95) clearInterval(interval);
    }, (durationInSeconds * 1000) / 100);
}

function finishProgressBar() {
    setTimeout(() => {
        if (dom.progressBar) { dom.progressBar.style.width = '100%'; dom.progressText.textContent = '100%'; }
        setTimeout(() => { if (dom.progressModal) dom.progressModal.style.display = 'none'; }, 500);
    }, 200);
}

function showError(title, message) {
    if (dom.errorModal) dom.errorModal.style.display = 'flex';
    const errorTitleEl = document.getElementById('error-title');
    const errorMessageEl = document.getElementById('error-message');
    if(errorTitleEl) errorTitleEl.textContent = title;
    if(errorMessageEl) errorMessageEl.textContent = message;
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if(modal) modal.style.display = 'none';
}
window.closeModal = closeModal;

function applyFilters() {
    if(dom.resultImageDisplay) dom.resultImageDisplay.style.filter = activeFilter;
}

function goToStep(stepNumber) {
    currentStep = stepNumber;
    document.querySelectorAll('.wizard-step').forEach(step => step.classList.add('hidden'));
    const activeStepDiv = document.getElementById(`step-${stepNumber}-subject`) || document.getElementById(`step-${stepNumber}-scene`) || document.getElementById(`step-${stepNumber}-upscale`) || document.getElementById(`step-${stepNumber}-finalize`);
    if (activeStepDiv) activeStepDiv.classList.remove('hidden');
}

async function detectAndDrawFaces(imageBlob, imageElement, boxesContainer, faceArray, selectionType) {
    if (!imageElement || !imageElement.complete || imageElement.naturalWidth === 0) {
        imageElement.onload = () => detectAndDrawFaces(imageBlob, imageElement, boxesContainer, faceArray, selectionType);
        return;
    }
    try {
        const data = await api.detectFaces(imageBlob);
        faceArray.splice(0, faceArray.length, ...data.faces);
        drawFaceBoxes(boxesContainer, imageElement, faceArray, selectionType);
    } catch (err) {
        showError('Errore Rilevamento Volti', err.message);
        faceArray.length = 0;
        if (boxesContainer && imageElement) {
           drawFaceBoxes(boxesContainer, imageElement, faceArray, selectionType);
        }
    }
}

function handleFaceSelection(index, type) {
    if (type === 'source') {
        selectedSourceIndex = (selectedSourceIndex === index) ? -1 : index;
    } else {
        selectedTargetIndex = (selectedTargetIndex === index) ? -1 : index;
    }
    if(dom.selectedSourceId) dom.selectedSourceId.textContent = selectedSourceIndex !== -1 ? `#${selectedSourceIndex + 1}` : 'Nessuno';
    if(dom.selectedTargetId) dom.selectedTargetId.textContent = selectedTargetIndex !== -1 ? `#${selectedTargetIndex + 1}` : 'Nessuno';
    updateSelectionHighlights(dom.sourceFaceBoxesContainer, selectedSourceIndex);
    updateSelectionHighlights(dom.targetFaceBoxesContainer, selectedTargetIndex);
    if(dom.swapBtn) dom.swapBtn.disabled = !(selectedSourceIndex !== -1 && selectedTargetIndex !== -1);
}

function handleSubjectFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    subjectFile = file;
    displayImage(file, dom.subjectImgPreview);
    if(dom.prepareSubjectBtn) dom.prepareSubjectBtn.disabled = false;
    if(dom.skipToSwapBtn) dom.skipToSwapBtn.disabled = false;
}

function handleSourceFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    sourceImageFile = file;
    dom.sourceImgPreview.onload = () => {
        detectAndDrawFaces(file, dom.sourceImgPreview, dom.sourceFaceBoxesContainer, sourceFaces, 'source');
    };
    displayImage(file, dom.sourceImgPreview);
    if(dom.selectionStatus) dom.selectionStatus.classList.remove('hidden');
}

function handleSkipToSwap() {
    if (!subjectFile) return showError("File Mancante", "Carica un'immagine prima di saltare allo swap.");
    upscaledImageBlob = subjectFile;
    processedSubjectBlob = null; sceneImageBlob = null; finalImageWithSwap = null;
    dom.resultImageDisplay.onload = () => {
        detectAndDrawFaces(upscaledImageBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, targetFaces, 'target');
    };
    displayImage(upscaledImageBlob, dom.resultImageDisplay);
    goToStep(4);
}

async function handlePrepareSubject() {
    if (!subjectFile) return showError("File Mancante", "Carica un'immagine del soggetto.");
    startProgressBar("Step 1: Preparazione Soggetto...", 15);
    try {
        processedSubjectBlob = await api.prepareSubject(subjectFile);
        dom.resultImageDisplay.onload = () => {
            detectAndDrawFaces(processedSubjectBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, targetFaces, 'target');
        };
        displayImage(processedSubjectBlob, dom.resultImageDisplay);
        goToStep(2);
    } catch (err) { showError("Errore Preparazione", err.message); } 
    finally { finishProgressBar(); }
}

async function handleCreateScene() {
    if (!dom.bgPromptInput.value) return showError("Prompt Mancante", "Descrivi lo sfondo.");
    if (!processedSubjectBlob) return showError("Soggetto Mancante", "Completa prima lo Step 1.");
    startProgressBar("Step 2: Creazione Scena...", 45);
    try {
        let finalPrompt = dom.bgPromptInput.value;
        if (dom.autoEnhancePromptToggle.checked) {
            const result = await api.enhancePrompt(processedSubjectBlob, dom.bgPromptInput.value);
            finalPrompt = result.enhanced_prompt;
            dom.bgPromptInput.value = finalPrompt;
        }
        sceneImageBlob = await api.createScene(processedSubjectBlob, finalPrompt);
        finalImageWithSwap = null;
        displayImage(sceneImageBlob, dom.resultImageDisplay);
        if(dom.gotoStep3Btn) dom.gotoStep3Btn.disabled = false;
    } catch (err) { showError("Errore Creazione Scena", err.message); } 
    finally { finishProgressBar(); }
}

async function handleUpscaleAndDetail() {
    if (!sceneImageBlob) return showError("Scena Mancante", "Completa prima lo Step 2.");
    startProgressBar("Step 3: Upscale & Detailing...", 90);
    try {
        upscaledImageBlob = await api.upscaleAndDetail(sceneImageBlob, dom.enableHiresUpscaleToggle.checked, dom.tileDenoisingSlider.value);
        finalImageWithSwap = null;
        dom.resultImageDisplay.onload = () => {
            detectAndDrawFaces(upscaledImageBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, targetFaces, 'target');
        };
        displayImage(upscaledImageBlob, dom.resultImageDisplay);
        goToStep(4);
    } catch (err) { showError("Errore Upscale", err.message); } 
    finally { finishProgressBar(); }
}

async function handlePerformSwap() {
    if (selectedSourceIndex === -1 || selectedTargetIndex === -1) return showError("Selezione Mancante", "Seleziona un volto sorgente E un volto di destinazione.");
    const targetImageBlob = finalImageWithSwap || upscaledImageBlob;
    if (!targetImageBlob) return showError("Immagine Mancante", "Completa i passaggi precedenti.");
    startProgressBar("Step 4: Face Swap Mirato...", 10);
    try {
        finalImageWithSwap = await api.performSwap(targetImageBlob, sourceImageFile, selectedSourceIndex, selectedTargetIndex);
        dom.resultImageDisplay.onload = () => {
            detectAndDrawFaces(finalImageWithSwap, dom.resultImageDisplay, dom.targetFaceBoxesContainer, targetFaces, 'target');
        };
        displayImage(finalImageWithSwap, dom.resultImageDisplay);
        selectedTargetIndex = -1;
        if(dom.selectedTargetId) dom.selectedTargetId.textContent = 'Nessuno';
        updateSelectionHighlights(dom.targetFaceBoxesContainer, -1);
        if(dom.swapBtn) dom.swapBtn.disabled = true;
    } catch (err) { showError("Errore Face Swap", err.message); } 
    finally { finishProgressBar(); }
}

function getWrappedLines(context, text, maxWidth) {
    if (!text) return [];
    const words = text.split(' ');
    const lines = [];
    let currentLine = words[0] || '';

    for (let i = 1; i < words.length; i++) {
        const word = words[i];
        const width = context.measureText(currentLine + " " + word).width;
        if (width < maxWidth) {
            currentLine += " " + word;
        } else {
            lines.push(currentLine);
            currentLine = word;
        }
    }
    lines.push(currentLine);
    return lines;
}

function updateMemePreview() {
    const imageToDrawOn = dom.resultImageDisplay;
    if (!imageToDrawOn || !imageToDrawOn.complete || imageToDrawOn.naturalWidth === 0) {
        if(dom.memeCanvas) dom.memeCanvas.classList.add('hidden');
        if (imageToDrawOn && imageToDrawOn.src) imageToDrawOn.classList.remove('hidden');
        return;
    }

    const canvas = dom.memeCanvas;
    const ctx = canvas.getContext('2d');
    canvas.width = imageToDrawOn.naturalWidth;
    canvas.height = imageToDrawOn.naturalHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.filter = activeFilter;
    ctx.drawImage(imageToDrawOn, 0, 0);
    ctx.filter = 'none';

    const text = dom.captionTextInput.value;
    const shouldShowCanvas = text || stickerStack.length > 0;

    if (shouldShowCanvas) {
        dom.resultImageDisplay.classList.add('hidden');
        dom.memeCanvas.classList.remove('hidden');
    } else {
        dom.memeCanvas.classList.add('hidden');
        dom.resultImageDisplay.classList.remove('hidden');
    }
// -------------- VISIBILITÀ PULSANTE SHARE / COPIA LINK --------------
if (navigator.share && navigator.canShare) {
    dom.shareBtn.classList.remove('hidden');              // contesto HTTPS / localhost
    dom.shareBtn.textContent = 'Condividi';
} else {
    dom.shareBtn.classList.remove('hidden');              // contesto HTTP
    dom.shareBtn.textContent = 'Copia link';
}
dom.shareBtn.style.cssText += 'display:inline-flex !important; opacity:1 !important; visibility:visible !important;';

    if (text) {
        const fontFamily = dom.fontFamilySelect.value;
        const fontSize = parseInt(dom.fontSizeSlider.value, 10);
        const fontColor = dom.fontColorInput.value;
        const strokeColor = dom.strokeColorInput.value;
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
        const lines = getWrappedLines(ctx, text, maxWidth);

        let startY;
        if (position === 'top') {
            ctx.textBaseline = 'top';
            startY = margin;
        } else {
            ctx.textBaseline = 'bottom';
            startY = canvas.height - margin - (lines.length - 1) * lineHeight;
        }

        lines.forEach((line, index) => {
            const y = startY + (index * lineHeight);
            if (textBg !== 'none') {
                const textMetrics = ctx.measureText(line);
                const bgHeight = lineHeight;
                const bgWidth = textMetrics.width + (fontSize * 0.5);
                const bgX = x - (bgWidth / 2);
                let bgY = (position === 'top') ? y - (lineHeight - fontSize) / 2 : y - fontSize - (lineHeight - fontSize) / 2;
                ctx.globalAlpha = 0.7;
                ctx.fillStyle = textBg;
                ctx.fillRect(bgX, bgY, bgWidth, bgHeight);
                ctx.globalAlpha = 1.0;
                ctx.fillStyle = fontColor;
            }
            ctx.strokeText(line, x, y);
            ctx.fillText(line, x, y);
        });
    }

    stickerStack.forEach(sticker => {
        if (!sticker.element) return; // Salta se l'elemento non è pronto
        ctx.save();
        ctx.translate(sticker.x + sticker.width / 2, sticker.y + sticker.height / 2);
        ctx.rotate(sticker.rotation);
        ctx.drawImage(sticker.element, -sticker.width / 2, -sticker.height / 2, sticker.width, sticker.height);
        ctx.restore();
        
        
        if (sticker === selectedSticker) {
            ctx.save();
            ctx.translate(sticker.x + sticker.width / 2, sticker.y + sticker.height / 2);
            ctx.rotate(sticker.rotation);
            ctx.strokeStyle = '#007bff';
            ctx.lineWidth = 4;
            const handleSize = 12;
            ctx.fillStyle = '#007bff';
            
            ctx.strokeRect(-sticker.width / 2, -sticker.height / 2, sticker.width, sticker.height);
            ctx.fillRect(sticker.width / 2 - handleSize / 2, sticker.height / 2 - handleSize / 2, handleSize, handleSize);
            
            ctx.beginPath();
            ctx.moveTo(0, -sticker.height / 2);
            ctx.lineTo(0, -sticker.height / 2 - 20);
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(0, -sticker.height / 2 - 25, handleSize / 2, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }
    });
}


async function loadStickers() {
    const gallery = dom.stickerGallery;
    if (!gallery) return;
    gallery.innerHTML = '<p class="text-gray-500">Caricamento sticker...</p>';

    try {
        const categories = await api.getStickers();
        gallery.innerHTML = '';

        if (categories.length === 0) {
            gallery.innerHTML = '<p class="text-gray-500">Nessuno sticker trovato in static/stickers.</p>';
            return;
        }

        categories.forEach(category => {
            const categoryTitle = document.createElement('h4');
            categoryTitle.className = 'font-bold text-sm text-blue-400 mt-2 mb-1 px-1 category-title w-full';
            categoryTitle.textContent = category.category;
            gallery.appendChild(categoryTitle);

            const stickerContainer = document.createElement('div');
            stickerContainer.className = 'flex flex-wrap gap-2 sticker-container';
            gallery.appendChild(stickerContainer);

            category.stickers.forEach(stickerPath => {
                const isVideo = stickerPath.endsWith('.webm');
                const isLottie = stickerPath.endsWith('.tgs');
                let galleryElement;

                const stickerWrapper = document.createElement('div');
                stickerWrapper.className = 'sticker-item-wrapper relative';

                if (isLottie) {
                    galleryElement = document.createElement('div');
                    const lottieJsonPath = `/lottie_json/${stickerPath.replace('static/', '')}`;
                    lottie.loadAnimation({
                        container: galleryElement,
                        renderer: 'svg',
                        loop: true,
                        autoplay: true,
                        path: lottieJsonPath
                    });
                } else if (isVideo) {
                    galleryElement = document.createElement('video');
                    galleryElement.autoplay = true;
                    galleryElement.muted = true;
                    galleryElement.loop = true;
                    galleryElement.playsInline = true;
                    galleryElement.src = stickerPath;
                } else {
                    galleryElement = document.createElement('img');
                    galleryElement.src = stickerPath;
                    galleryElement.crossOrigin = "anonymous";
                }
                galleryElement.className = 'h-20 w-auto object-contain p-1 bg-gray-700 rounded-md cursor-pointer hover:bg-blue-600 transition-all sticker-item';
                
                stickerWrapper.appendChild(galleryElement);

                stickerWrapper.addEventListener('click', () => {
                    if (isLottie) {
                        const lottieJsonPath = `/lottie_json/${stickerPath.replace('static/', '')}`;
                        
                        // Step 1: Carichiamo prima i dati dell'animazione per leggerne le dimensioni
                        fetch(lottieJsonPath)
                            .then(response => response.json())
                            .then(animationData => {
                                const naturalWidth = animationData.w;
                                const naturalHeight = animationData.h;

                                if (!naturalWidth || !naturalHeight) {
                                    showError("Errore Sticker Lottie", "Dati dell'animazione non validi o corrotti.");
                                    return;
                                }

                                // Step 2: Creiamo la nostra tela (canvas)
                                const stickerCanvas = document.createElement('canvas');
                                stickerCanvas.width = naturalWidth;
                                stickerCanvas.height = naturalHeight;
                                const stickerCtx = stickerCanvas.getContext('2d');

                                // Step 3: Diciamo a Lottie di disegnare sulla nostra tela
                                const lottieAnim = lottie.loadAnimation({
                                    renderer: 'canvas',
                                    loop: true,
                                    autoplay: true,
                                    animationData: animationData, // Usiamo i dati pre-caricati
                                    rendererSettings: {
                                        context: stickerCtx,
                                        clearCanvas: true, // Lottie pulisce la tela ad ogni frame
                                    },
                                });

                                // Step 4: Aggiungiamo il nostro canvas (ora animato) allo stack degli sticker
                                const aspectRatio = naturalHeight / naturalWidth;
                                const newWidth = 150;
                                const newSticker = {
                                    element: stickerCanvas,
                                    type: 'lottie',
                                    x: 20,
                                    y: 20,
                                    width: newWidth,
                                    height: newWidth * aspectRatio,
                                    rotation: 0,
                                    aspectRatio: aspectRatio
                                };
                                stickerStack.push(newSticker);
                                selectedSticker = newSticker;

                            })
                            .catch(err => {
                                showError("Errore Sticker Lottie", "Impossibile caricare i dati JSON dell'animazione.");
                                console.error("Errore fetch Lottie:", err);
                            });

                    } else { // Logica per immagini e video (invariata)
                        const elementToAdd = galleryElement;
                        const naturalWidth = galleryElement.naturalWidth || galleryElement.videoWidth;
                        const naturalHeight = galleryElement.naturalHeight || galleryElement.videoHeight;
                        if(naturalWidth > 0) {
                            const aspectRatio = naturalHeight / naturalWidth;
                            const newWidth = 150;
                            const newSticker = {
                                element: elementToAdd,
                                type: isVideo ? 'video' : 'image',
                                x: 20, y: 20, width: newWidth, height: newWidth * aspectRatio,
                                rotation: 0, aspectRatio: aspectRatio
                            };
                            stickerStack.push(newSticker);
                            selectedSticker = newSticker;
                        }
                    }
                });
                stickerContainer.appendChild(stickerWrapper);
            });
        });
    } catch (err) {
        gallery.innerHTML = '<p class="text-red-500">Errore nel caricamento degli sticker.</p>';
        showError("Errore Caricamento Sticker", err.message);
    }
}


function getStickerAtPosition(x, y) {
    // Decide handle radius based on pointer type
    const isTouchDevice = window.matchMedia && window.matchMedia('(pointer: coarse)').matches;
    const handleRadius = isTouchDevice ? 60 : 20;

    // Loop top‑down through the sticker stack
    for (let i = stickerStack.length - 1; i >= 0; i--) {
        const sticker = stickerStack[i];
        const cx = sticker.x + sticker.width / 2;
        const cy = sticker.y + sticker.height / 2;

        // PRE‑COMPUTE the two physical handle positions **before** rotation
        const rawRot = { x: cx, y: cy - sticker.height / 2 - 25 };
        const rawResize = { x: sticker.x + sticker.width, y: sticker.y + sticker.height };

        // Rotate them around the sticker centre
        const rotHandle = rotatePoint(rawRot.x, rawRot.y, cx, cy, sticker.rotation);
        const resizeHandle = rotatePoint(rawResize.x, rawResize.y, cx, cy, sticker.rotation);

        // --- 1) HANDLE HIT‑TESTS  ---
        if (distance(x, y, rotHandle.x, rotHandle.y) < handleRadius) {
            return { sticker, corner: 'rotate' };
        }
        if (distance(x, y, resizeHandle.x, resizeHandle.y) < handleRadius) {
            return { sticker, corner: 'resize' };
        }

        // --- 2) STANDARD RECT HIT‑TEST (for drag) ---
        if (isPointInRotatedRectangle({ x, y }, sticker)) {
            return { sticker, corner: 'drag' };
        }
    }
    return null;
}

function isPointInRotatedRectangle(point, rect) {
    const cx = rect.x + rect.width / 2;
    const cy = rect.y + rect.height / 2;
    const rotatedPoint = rotatePoint(point.x, point.y, cx, cy, -rect.rotation);
    return rotatedPoint.x > rect.x && rotatedPoint.x < rect.x + rect.width &&
           rotatedPoint.y > rect.y && rotatedPoint.y < rect.y + rect.height;
}

function rotatePoint(x, y, cx, cy, angle) {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    return {
        x: (cos * (x - cx)) - (sin * (y - cy)) + cx,
        y: (sin * (x - cx)) + (cos * (y - cy)) + cy
    };
}

function distance(x1, y1, x2, y2) {
    return Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2));
}

function animationLoop() {
    updateMemePreview();
    requestAnimationFrame(animationLoop);
}

function assignDomElements() {
    const ids = [
        'error-modal', 'progress-modal', 'progress-bar', 'progress-text', 'progress-title', 'result-image-display',
        'result-placeholder', 'download-btn', 'reset-all-btn', 'step-1-subject', 'subject-img-input',
        'subject-img-preview', 'subject-upload-prompt', 'prepare-subject-btn', 'skip-to-swap-btn', 'step-2-scene',
        'bg-prompt-input', 'auto-enhance-prompt-toggle', 'gemini-api-key-wrapper', 'gemini-api-key-input',
        'generate-scene-btn', 'goto-step-3-btn', 'step-3-upscale', 'tile-denoising-slider', 'tile-denoising-value',
        'start-upscale-btn', 'skip-upscale-btn', 'enable-hires-upscale-toggle', 'step-4-finalize', 'source-img-input',
        'source-img-preview', 'source-upload-prompt', 'swap-btn', 'filter-buttons-container', 'meme-section', 'caption-btn',
        'back-to-step-3-btn', 'source-face-boxes-container', 'target-face-boxes-container', 'selection-status',
        'selected-source-id', 'selected-target-id', 'tone-buttons-container', 'caption-text-input',
        'meme-canvas', 'font-family-select', 'font-size-slider', 'font-size-value',
        'font-color-input', 'stroke-color-input', 'position-buttons', 'text-bg-buttons',
        'sticker-section', 'sticker-gallery', 'sticker-delete-btn', 'sticker-front-btn', 'sticker-back-btn', 'toggle-face-boxes',
        'share-btn', 'sticker-search-input'
    ];

    // --- Stub for removed Gemini API elements to prevent null errors ---
    ['geminiApiKeyWrapper','geminiApiKeyInput'].forEach((key) => {
        if(!dom[key]) {
            dom[key] = {
                classList: { add: ()=>{}, remove: ()=>{}, toggle: ()=>{} },
                value: '',
                focus: ()=>{}
            };
        }
    });
    
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (!el) console.warn(`Elemento non trovato: #${id}`);
        const camelCaseId = id.replace(/-(\w)/g, (_, c) => c.toUpperCase());
        dom[camelCaseId] = el;
    });
}

function setupEventListeners() {
    dom.resetAllBtn.addEventListener('click', resetWorkflow);
    dom.subjectImgInput.addEventListener('change', (e) => handleSubjectFile(e.target.files[0]));
    dom.prepareSubjectBtn.addEventListener('click', handlePrepareSubject);
    dom.skipToSwapBtn.addEventListener('click', handleSkipToSwap);
    if(dom.autoEnhancePromptToggle) dom.autoEnhancePromptToggle.addEventListener('change', (e) => dom.geminiApiKeyWrapper?.classList.toggle('hidden', !e.target.checked));
    if(dom.geminiApiKeyInput) dom.geminiApiKeyInput.addEventListener('blur', () => api.saveApiKey(dom.geminiApiKeyInput.value));
    if(dom.generateSceneBtn) dom.generateSceneBtn.addEventListener('click', handleCreateScene);
    if(dom.gotoStep3Btn) dom.gotoStep3Btn.addEventListener('click', () => goToStep(3));
    if(dom.tileDenoisingSlider) dom.tileDenoisingSlider.addEventListener('input', (e) => {
        if (dom.tileDenoisingValue) dom.tileDenoisingValue.textContent = parseFloat(e.target.value).toFixed(2);
    });
    if(dom.startUpscaleBtn) dom.startUpscaleBtn.addEventListener('click', handleUpscaleAndDetail);
    if(dom.skipUpscaleBtn) dom.skipUpscaleBtn.addEventListener('click', () => {
        upscaledImageBlob = sceneImageBlob; 
        finalImageWithSwap = null;
        dom.resultImageDisplay.onload = () => {
             detectAndDrawFaces(upscaledImageBlob, dom.resultImageDisplay, dom.targetFaceBoxesContainer, targetFaces, 'target');
        };
        displayImage(upscaledImageBlob, dom.resultImageDisplay);
        goToStep(4);
    });
    if(dom.sourceImgInput) dom.sourceImgInput.addEventListener('change', (e) => handleSourceFile(e.target.files[0]));
    if(dom.swapBtn) dom.swapBtn.addEventListener('click', handlePerformSwap);
    if(dom.backToStep3Btn) dom.backToStep3Btn.addEventListener('click', () => goToStep(3));
    if(dom.filterButtonsContainer) dom.filterButtonsContainer.addEventListener('click', (e) => {
        if (e.target.tagName === 'BUTTON') {
            dom.filterButtonsContainer.querySelector('.active')?.classList.remove('active');
            e.target.classList.add('active');
            activeFilter = e.target.dataset.filter;
            applyFilters();
            updateMemePreview();
        }
    });
    if(dom.captionBtn) dom.captionBtn.addEventListener('click', async () => {
        const imageToCaption = finalImageWithSwap || upscaledImageBlob || sceneImageBlob || subjectFile;
        if (!imageToCaption) return showError("Immagine Mancante", "Nessuna immagine nell'anteprima.");
        startProgressBar("Generazione Didascalia Meme...", 10);
        try {
            const selectedTone = dom.toneButtonsContainer.querySelector('.active')?.dataset.tone || 'scherzoso';
            const result = await api.generateCaption(imageToCaption, selectedTone);
            dom.captionTextInput.value = result.caption;
            updateMemePreview();
        } catch (err) { showError("Errore Generazione Didascalia", err.message); } 
        finally { finishProgressBar(); }
    });
    if(dom.toneButtonsContainer) dom.toneButtonsContainer.addEventListener('click', (e) => {
        if (e.target.classList.contains('tone-btn')) {
            dom.toneButtonsContainer.querySelector('.active')?.classList.remove('active');
            e.target.classList.add('active');
        }
    });
    if(dom.captionTextInput) dom.captionTextInput.addEventListener('input', updateMemePreview);
    if(dom.fontFamilySelect) dom.fontFamilySelect.addEventListener('change', updateMemePreview);
    if(dom.fontSizeSlider) dom.fontSizeSlider.addEventListener('input', () => {
        dom.fontSizeValue.textContent = dom.fontSizeSlider.value;
        updateMemePreview();
    });
    if(dom.fontColorInput) dom.fontColorInput.addEventListener('input', updateMemePreview);
    if(dom.strokeColorInput) dom.strokeColorInput.addEventListener('input', updateMemePreview);
    if(dom.positionButtons) dom.positionButtons.addEventListener('click', (e) => {
        if (e.target.classList.contains('meme-control-btn')) {
            dom.positionButtons.querySelector('.active')?.classList.remove('active');
            e.target.classList.add('active');
            updateMemePreview();
        }
    });
    if(dom.textBgButtons) dom.textBgButtons.addEventListener('click', (e) => {
        if (e.target.classList.contains('meme-control-btn')) {
            dom.textBgButtons.querySelector('.active')?.classList.remove('active');
            e.target.classList.add('active');
            updateMemePreview();
        }
    });
    if(dom.downloadBtn) dom.downloadBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        selectedSticker = null;
        updateMemePreview();
        await new Promise(resolve => setTimeout(resolve, 50));
        
        // Se il canvas non è visibile, scarica direttamente l'immagine
        if (dom.memeCanvas.classList.contains('hidden')) {
            const link = document.createElement('a');
            link.href = dom.resultImageDisplay.src;
            link.download = 'pro-meme-result.png';
            link.click();
            return;
        }

        const dataUrl = dom.memeCanvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = 'pro-meme-result.png';
        link.click();
    });

    if (dom.shareBtn) {
        // La logica per mostrare/nascondere il pulsante è in displayImage() e resetWorkflow()
        // Qui aggiungiamo solo il listener se il pulsante è potenzialmente utilizzabile
        if (navigator.share && navigator.canShare) {
             dom.shareBtn.addEventListener('click', async () => {
                selectedSticker = null;
                updateMemePreview();
                await new Promise(resolve => setTimeout(resolve, 50));

                let blob;
                 if (dom.memeCanvas.classList.contains('hidden')) {
                     blob = await fetch(dom.resultImageDisplay.src).then(r => r.blob());
                 } else {
                     blob = await new Promise(resolve => dom.memeCanvas.toBlob(resolve, 'image/png'));
                 }

                const file = new File([blob], 'meme.png', { type: 'image/png' });
                const shareData = {
                    title: 'Il Mio Meme',
                    text: 'Guarda cosa ho creato con AI Face Swap Studio Pro!',
                    files: [file]
                };

                try {
                    if (navigator.canShare(shareData)) {
                        await navigator.share(shareData);
                    } else {
                        showError("Condivisione non possibile", "Il browser non può condividere questo tipo di file.");
                    }
                } catch (err) {
                    if (err.name !== 'AbortError') {
                        showError("Errore di Condivisione", err.toString());
                    }
                }
            });
        }
    }
    
    if(dom.stickerDeleteBtn) dom.stickerDeleteBtn.addEventListener('click', () => {
        if (selectedSticker) {
            stickerStack = stickerStack.filter(s => s !== selectedSticker);
            selectedSticker = null;
        }
    });
    if(dom.stickerFrontBtn) dom.stickerFrontBtn.addEventListener('click', () => {
        if(selectedSticker) {
            const index = stickerStack.indexOf(selectedSticker);
            if (index < stickerStack.length - 1) {
                stickerStack.splice(index, 1);
                stickerStack.push(selectedSticker);
            }
        }
    });
    if(dom.stickerBackBtn) dom.stickerBackBtn.addEventListener('click', () => {
        if(selectedSticker) {
            const index = stickerStack.indexOf(selectedSticker);
            if (index > 0) {
                stickerStack.splice(index, 1);
                stickerStack.unshift(selectedSticker);
            }
        }
    });
    if(dom.toggleFaceBoxes) dom.toggleFaceBoxes.addEventListener('change', (e) => {
        const displayValue = e.target.checked ? 'block' : 'none';
        if (dom.targetFaceBoxesContainer) dom.targetFaceBoxesContainer.style.display = displayValue;
        if (dom.sourceFaceBoxesContainer) dom.sourceFaceBoxesContainer.style.display = displayValue;
    });

    if(dom.stickerSearchInput) {
        dom.stickerSearchInput.addEventListener('input', (e) => {
            const searchTerm = e.target.value.toLowerCase();
            document.querySelectorAll('#sticker-gallery .category-title').forEach(title => {
                const container = title.nextElementSibling;
                let hasVisibleStickers = false;
                container.querySelectorAll('.sticker-item-wrapper').forEach(wrapper => {
                    const stickerEl = wrapper.querySelector('.sticker-item');
                    const stickerPath = (stickerEl.src || '').toLowerCase();
                    const match = stickerPath.includes(searchTerm) || title.textContent.toLowerCase().includes(searchTerm);
                    wrapper.style.display = match ? 'inline-block' : 'none';
                    if(match) hasVisibleStickers = true;
                });
                title.style.display = hasVisibleStickers ? 'block' : 'none';
            });
        });
    }

    const getCoordsFromEvent = (e) => {
        const rect = dom.memeCanvas.getBoundingClientRect();
        const scaleX = dom.memeCanvas.width / rect.width;
        const scaleY = dom.memeCanvas.height / rect.height;
        let clientX, clientY;
        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }
        const mouseX = (clientX - rect.left) * scaleX;
        const mouseY = (clientY - rect.top) * scaleY;
        return { x: mouseX, y: mouseY };
    };

    const handleInteractionStart = (e) => {
        const coords = getCoordsFromEvent(e);
        const hit = getStickerAtPosition(coords.x, coords.y);
        
        if (hit) {
            e.preventDefault(); // Previene comportamenti di default come lo scroll su mobile
            selectedSticker = hit.sticker;
            if (hit.corner === 'resize') {
                isResizing = true; isDragging = false; isRotating = false;
            } else if (hit.corner === 'rotate') {
                isRotating = true; isDragging = false; isResizing = false;
            } else {
                isDragging = true; isResizing = false; isRotating = false;
                dragOffsetX = coords.x - selectedSticker.x;
                dragOffsetY = coords.y - selectedSticker.y;
            }
        } else {
            selectedSticker = null;
        }
        
        if(dom.stickerDeleteBtn) dom.stickerDeleteBtn.disabled = !selectedSticker;
        if(dom.stickerFrontBtn) dom.stickerFrontBtn.disabled = !selectedSticker;
        if(dom.stickerBackBtn) dom.stickerBackBtn.disabled = !selectedSticker;
    };

    const handleInteractionMove = (e) => {
        if (!selectedSticker) return;
        if (isDragging || isResizing || isRotating) {
            e.preventDefault();
        }
        const coords = getCoordsFromEvent(e);
        
        if (isResizing) {
            const stickerCenterX = selectedSticker.x + selectedSticker.width / 2;
            const stickerCenterY = selectedSticker.y + selectedSticker.height / 2;
            const dist = distance(coords.x, coords.y, stickerCenterX, stickerCenterY);
            const newWidth = dist * Math.sqrt(2); // Calcola la nuova larghezza basandosi sulla distanza dal centro

            if (newWidth > 20) {
                const oldWidth = selectedSticker.width;
                selectedSticker.width = newWidth;
                selectedSticker.height = newWidth * selectedSticker.aspectRatio;
                // Mantieni il centro fisso
                selectedSticker.x += (oldWidth - newWidth) / 2;
                selectedSticker.y += ((oldWidth * selectedSticker.aspectRatio) - (newWidth * selectedSticker.aspectRatio)) / 2;
            }
        } else if (isRotating) {
            const cx = selectedSticker.x + selectedSticker.width / 2;
            const cy = selectedSticker.y + selectedSticker.height / 2;
            selectedSticker.rotation = Math.atan2(coords.y - cy, coords.x - cx) + Math.PI / 2;
        } else if (isDragging) {
            selectedSticker.x = coords.x - dragOffsetX;
            selectedSticker.y = coords.y - dragOffsetY;
        }
    };

    const handleInteractionEnd = () => {
        isDragging = false;
        isResizing = false;
        isRotating = false;
    };

    dom.memeCanvas.addEventListener('mousedown', handleInteractionStart);
    dom.memeCanvas.addEventListener('mousemove', handleInteractionMove);
    dom.memeCanvas.addEventListener('mouseup', handleInteractionEnd);
    dom.memeCanvas.addEventListener('mouseout', handleInteractionEnd);
    dom.memeCanvas.addEventListener('touchstart', handleInteractionStart, { passive: false });
    dom.memeCanvas.addEventListener('touchmove', handleInteractionMove, { passive: false });
    dom.memeCanvas.addEventListener('touchend', handleInteractionEnd);
    dom.memeCanvas.addEventListener('touchcancel', handleInteractionEnd);
    
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Delete' || e.key === 'Backspace') {
            if (selectedSticker && document.activeElement.tagName !== 'INPUT') {
                e.preventDefault();
                stickerStack = stickerStack.filter(s => s !== selectedSticker);
                selectedSticker = null;
            }
        }
    });
}

function resetWorkflow() {
    stickerStack = []; selectedSticker = null; isDragging = false; isResizing = false; isRotating = false;
    currentStep = 1; subjectFile = null; processedSubjectBlob = null; sceneImageBlob = null;
    upscaledImageBlob = null; finalImageWithSwap = null; activeFilter = 'none';
    if(dom.subjectImgPreview) { dom.subjectImgPreview.src = ''; dom.subjectImgPreview.classList.add('hidden'); }
    if(dom.subjectUploadPrompt) dom.subjectUploadPrompt.style.display = 'block';
    if(dom.sourceImgPreview) { dom.sourceImgPreview.src = ''; dom.sourceImgPreview.classList.add('hidden'); }
    const sourceUploadPrompt = document.getElementById('source-upload-prompt');
    if(sourceUploadPrompt) sourceUploadPrompt.style.opacity = '1';
    if(dom.subjectImgInput) dom.subjectImgInput.value = '';
    if(dom.sourceImgInput) dom.sourceImgInput.value = '';
    if(dom.bgPromptInput) dom.bgPromptInput.value = '';
    if(dom.tileDenoisingSlider) dom.tileDenoisingSlider.value = '0.4';
    if(dom.tileDenoisingValue) dom.tileDenoisingValue.textContent = '0.40';
    if(dom.memeCanvas) dom.memeCanvas.classList.add('hidden');
    if(dom.resultImageDisplay) { dom.resultImageDisplay.src = ''; dom.resultImageDisplay.style.filter = 'none'; dom.resultImageDisplay.classList.remove('hidden'); }
    if(dom.resultPlaceholder) dom.resultPlaceholder.classList.remove('hidden');
    if(dom.downloadBtn) dom.downloadBtn.classList.add('hidden');
    if(dom.shareBtn) dom.shareBtn.classList.add('hidden');
    if(dom.prepareSubjectBtn) dom.prepareSubjectBtn.disabled = true; 
    if(dom.skipToSwapBtn) dom.skipToSwapBtn.disabled = true;
    if(dom.gotoStep3Btn) dom.gotoStep3Btn.disabled = true;
    if(dom.swapBtn) dom.swapBtn.disabled = true;
    sourceFaces = []; targetFaces = []; selectedSourceIndex = -1; selectedTargetIndex = -1; sourceImageFile = null;
    if(dom.sourceFaceBoxesContainer && dom.sourceImgPreview) drawFaceBoxes(dom.sourceFaceBoxesContainer, dom.sourceImgPreview, [], 'source');
    if(dom.targetFaceBoxesContainer && dom.resultImageDisplay) drawFaceBoxes(dom.targetFaceBoxesContainer, dom.resultImageDisplay, [], 'target');
    if(dom.selectionStatus) dom.selectionStatus.classList.add('hidden');
    if(dom.selectedSourceId) dom.selectedSourceId.textContent = 'Nessuno';
    if(dom.selectedTargetId) dom.selectedTargetId.textContent = 'Nessuno';
    if(dom.captionTextInput) dom.captionTextInput.value = '';
    if(dom.toggleFaceBoxes) dom.toggleFaceBoxes.checked = true;
    if(dom.targetFaceBoxesContainer) dom.targetFaceBoxesContainer.style.display = 'block';
    if(dom.sourceFaceBoxesContainer) dom.sourceFaceBoxesContainer.style.display = 'block';
    
    // Ripristina filtri e altri controlli
    if (dom.filterButtonsContainer) {
        dom.filterButtonsContainer.querySelector('.active')?.classList.remove('active');
        dom.filterButtonsContainer.querySelector('[data-filter="none"]')?.classList.add('active');
    }

    goToStep(1);
}

function main() {
    assignDomElements();
    setupEventListeners();
    loadStickers();
    resetWorkflow();
    animationLoop();
}

document.addEventListener('DOMContentLoaded', main);




/* === Mouse rotation support (desktop) === */
(function() {
  const isCoarse = window.matchMedia('(pointer: coarse)').matches;
  if (isCoarse) return; // touch users handled via pinch or handles

  const rotateHandles = document.querySelectorAll('.rotate-handle');

  rotateHandles.forEach(handle => {
    let startAngle = 0;
    let center = {x: 0, y: 0};
    let sticker = null;

    const onMouseMove = (e) => {
      if (!sticker) return;
      const dx = e.clientX - center.x;
      const dy = e.clientY - center.y;
      const angle = Math.atan2(dy, dx) * 180 / Math.PI;
      const delta = angle - startAngle;
      const currentRotation = parseFloat(sticker.dataset.rotation || 0);
      const newRotation = currentRotation + delta;
      sticker.style.transform = sticker.style.transform.replace(/rotate\([^)]*\)/, '') + ` rotate(${newRotation}deg)`;
      sticker.dataset.rotation = newRotation;
      startAngle = angle; // update for smooth continuous rotation
    };

    const onMouseUp = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      sticker = null;
    };

    handle.addEventListener('mousedown', (e) => {
      sticker = handle.closest('.sticker');
      if (!sticker) return;
      const rect = sticker.getBoundingClientRect();
      center = {x: rect.left + rect.width / 2, y: rect.top + rect.height / 2};
      const dx = e.clientX - center.x;
      const dy = e.clientY - center.y;
      startAngle = Math.atan2(dy, dx) * 180 / Math.PI;
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
      e.preventDefault();
    });
  });
})();


// --- Safe toggle for prompt enhancement ---
if (dom.autoEnhancePromptToggle) {
    dom.autoEnhancePromptToggle.addEventListener('change', () => {
        if(dom.geminiApiKeyWrapper) {
            dom.geminiApiKeyWrapper?.classList.toggle('hidden', !dom.autoEnhancePromptToggle.checked);
        }
    });
}

