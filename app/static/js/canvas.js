export let stickerStack = [];
export let selectedSticker = null;
let isDragging = false, isResizing = false, isRotating = false;
let dragOffsetX, dragOffsetY;
export let activeFilter = 'none';
export function setActiveFilter(val) {
    activeFilter = val;
}
let animationId = null;
let dom;
let showErrorFn;

export function initCanvas(elRefs, showError) {
    dom = elRefs;
    showErrorFn = showError;
    setupCanvasEvents();
}

export function updateMemePreview() {
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
    stickerStack.forEach(s => drawSticker(ctx, s));
}

function drawMemeText(ctx) {
    const { canvas } = ctx;
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
    const maxWidth = canvas.width - margin * 2;
    const lineHeight = fontSize * 1.2;
    const x = canvas.width / 2;
    const lines = getWrappedLines(ctx, dom.captionTextInput.value, maxWidth);
    ctx.textBaseline = position === 'top' ? 'top' : 'bottom';
    let startY = position === 'top' ? margin : canvas.height - margin - (lines.length - 1) * lineHeight;
    lines.forEach((line, idx) => {
        const y = startY + idx * lineHeight;
        if (textBg !== 'none') {
            const metrics = ctx.measureText(line);
            const bgW = metrics.width + fontSize * 0.5;
            const bgH = lineHeight;
            const bgX = x - bgW / 2;
            const bgY = position === 'top' ? y - (bgH - fontSize) / 2 : y - fontSize - (bgH - fontSize) / 2;
            ctx.globalAlpha = 0.7;
            ctx.fillStyle = textBg;
            ctx.fillRect(bgX, bgY, bgW, bgH);
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = fontColor;
        }
        ctx.strokeText(line, x, y);
        ctx.fillText(line, x, y);
    });
}

function getWrappedLines(ctx, text, maxWidth) {
    if (!text) return [];
    const words = text.split(' ');
    const lines = [];
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
        ctx.strokeStyle = '#007bff';
        ctx.fillStyle = '#007bff';
        ctx.lineWidth = 4;
        const handleSize = 12;
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
}

export async function addStickerToCanvas(element, isVideo, isLottie, path) {
    const stickerData = { type: isVideo ? 'video' : (isLottie ? 'lottie' : 'image'), x: 20, y: 20, rotation: 0 };
    let stickerElement, naturalW, naturalH;
    if (isLottie) {
        try {
            const animationData = await (await fetch(`/lottie_json/${path.replace('static/', '')}`)).json();
            naturalW = animationData.w; naturalH = animationData.h;
            stickerElement = document.createElement('canvas');
            stickerElement.width = naturalW; stickerElement.height = naturalH;
            lottie.loadAnimation({ renderer: 'canvas', loop: true, autoplay: true, animationData,
                rendererSettings: { context: stickerElement.getContext('2d'), clearCanvas: true } });
        } catch (err) { return showErrorFn('Errore Lottie', 'Impossibile caricare l\'animazione.'); }
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

function getCoords(e) {
    const rect = dom.memeCanvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: (clientX - rect.left) * (dom.memeCanvas.width / rect.width), y: (clientY - rect.top) * (dom.memeCanvas.height / rect.height) };
}

function onStart(e) {
    const hit = getStickerAtPosition(getCoords(e).x, getCoords(e).y);
    if (hit) {
        e.preventDefault();
        selectedSticker = hit.sticker;
        if (hit.corner === 'resize') isResizing = true;
        else if (hit.corner === 'rotate') isRotating = true;
        else { const c = getCoords(e); isDragging = true; dragOffsetX = c.x - hit.sticker.x; dragOffsetY = c.y - hit.sticker.y; }
        startAnimation();
    } else {
        selectedSticker = null;
    }
    [dom.stickerDeleteBtn, dom.stickerFrontBtn, dom.stickerBackBtn].forEach(b => b.disabled = !selectedSticker);
}

function onMove(e) {
    if (!selectedSticker || !(isDragging || isResizing || isRotating)) return;
    e.preventDefault();
    const { x, y } = getCoords(e);
    const s = selectedSticker, cx = s.x + s.width / 2, cy = s.y + s.height / 2;
    if (isResizing) {
        const newW = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) * Math.sqrt(2);
        if (newW > 20) { s.x += (s.width - newW) / 2; s.y += (s.height - newW * s.aspectRatio) / 2; s.width = newW; s.height = newW * s.aspectRatio; }
    } else if (isRotating) {
        s.rotation = Math.atan2(y - cy, x - cx) + Math.PI / 2;
    } else if (isDragging) {
        s.x = x - dragOffsetX; s.y = y - dragOffsetY;
    }
}

function onEnd() {
    isDragging = isResizing = isRotating = false;
    stopAnimation();
}

function setupCanvasEvents() {
    ['mousedown', 'touchstart'].forEach(evt => dom.memeCanvas.addEventListener(evt, onStart, { passive: false }));
    ['mousemove', 'touchmove'].forEach(evt => document.addEventListener(evt, onMove, { passive: false }));
    ['mouseup', 'touchend', 'touchcancel'].forEach(evt => document.addEventListener(evt, onEnd));
}

export function startAnimation() {
    if (!animationId) animationId = requestAnimationFrame(animationLoop);
}

export function stopAnimation() {
    if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
}

function animationLoop() {
    updateMemePreview();
    animationId = requestAnimationFrame(animationLoop);
}

export function getStickerAtPosition(x, y) {
    const handleRadius = window.matchMedia?.('(pointer: coarse)').matches ? 40 : 20;
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
    return { x: cos * (x - cx) - sin * (y - cy) + cx, y: sin * (x - cx) + cos * (y - cy) + cy };
}

function distance(x1, y1, x2, y2) {
    return Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
}

export function resetCanvas() {
    stickerStack = [];
    selectedSticker = null;
    stopAnimation();
    updateMemePreview();
}

export function deleteSelected() {
    if (selectedSticker) {
        stickerStack = stickerStack.filter(s => s !== selectedSticker);
        selectedSticker = null;
        updateMemePreview();
    }
}

export function bringToFront() {
    if (!selectedSticker) return;
    const i = stickerStack.indexOf(selectedSticker);
    if (i < stickerStack.length - 1) {
        stickerStack.splice(i, 1);
        stickerStack.push(selectedSticker);
    }
}

export function sendToBack() {
    if (!selectedSticker) return;
    const i = stickerStack.indexOf(selectedSticker);
    if (i > 0) {
        stickerStack.splice(i, 1);
        stickerStack.unshift(selectedSticker);
    }
}
