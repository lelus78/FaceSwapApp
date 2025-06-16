import { state, dom } from './state.js';
import * as api from './api.js';
import { showError, startProgressBar, finishProgressBar } from './workflow.js';
import { refreshFaceBoxes } from './facebox.js';

export function updateMemePreview() {
  const imageToDrawOn = dom.resultImageDisplay;
  if (!imageToDrawOn.src || !imageToDrawOn.complete || imageToDrawOn.naturalWidth === 0) return;
  const canvas = dom.memeCanvas, ctx = canvas.getContext('2d');
  canvas.width = imageToDrawOn.naturalWidth;
  canvas.height = imageToDrawOn.naturalHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.filter = state.activeFilter;
  ctx.drawImage(imageToDrawOn, 0, 0);
  ctx.filter = 'none';
  const text = dom.captionTextInput.value;
  const shouldShowCanvas = text || state.stickerStack.length > 0;
  dom.resultImageDisplay.classList.toggle('hidden', shouldShowCanvas);
  dom.memeCanvas.classList.toggle('hidden', !shouldShowCanvas);
  if (text) drawMemeText(ctx);
  state.stickerStack.forEach(sticker => drawSticker(ctx, sticker));
  if (!dom.resultImageDisplay.classList.contains('hidden')) refreshFaceBoxes();
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
  const maxWidth = canvas.width - margin * 2;
  const lineHeight = fontSize * 1.2;
  const x = canvas.width / 2;
  const lines = getWrappedLines(ctx, dom.captionTextInput.value, maxWidth);
  ctx.textBaseline = position === 'top' ? 'top' : 'bottom';
  let startY = position === 'top' ? margin : canvas.height - margin - (lines.length - 1) * lineHeight;
  lines.forEach((line, index) => {
    const y = startY + index * lineHeight;
    if (textBg !== 'none') {
      const metrics = ctx.measureText(line);
      const bgW = metrics.width + fontSize * 0.5,
            bgH = lineHeight;
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

export function drawSticker(ctx, sticker) {
  if (!sticker.element) return;
  ctx.save();
  ctx.translate(sticker.x + sticker.width / 2, sticker.y + sticker.height / 2);
  ctx.rotate(sticker.rotation);
  ctx.drawImage(sticker.element, -sticker.width / 2, -sticker.height / 2, sticker.width, sticker.height);
  ctx.restore();
  if (sticker === state.selectedSticker) {
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

export async function handleDownloadAnimation() {
  const hasAnimatedStickers = state.stickerStack.some(s => s.type === 'video' || s.type === 'lottie');
  if (!hasAnimatedStickers) {
    showError('Nessuna Animazione', 'Non ci sono sticker animati da registrare.');
    return;
  }
  state.selectedSticker = null;
  updateMemePreview();
  await new Promise(r => setTimeout(r, 50));
  startProgressBar('Registrazione animazione...', 5);
  try {
    const stream = dom.memeCanvas.captureStream(30);
    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    const chunks = [];
    recorder.ondataavailable = e => e.data.size > 0 && chunks.push(e.data);
    recorder.onstop = async () => {
      const webmBlob = new Blob(chunks, { type: 'video/webm' });
      finishProgressBar();
      startProgressBar('Conversione server...', 20);
      try {
        const format = dom.animFmt.value;
        const result = await api.saveResultVideo(webmBlob, format);
        const link = document.createElement('a');
        link.href = result.url;
        link.download = `pro-meme-result.${format}`;
        link.click();
      } catch (serverErr) {
        showError('Errore Server', serverErr.message);
      } finally {
        finishProgressBar();
      }
    };
    recorder.start();
    setTimeout(() => recorder.stop(), 5000);
  } catch (err) {
    showError('Errore Registrazione', err.message);
    finishProgressBar();
  }
}

export function animationLoop() {
  updateMemePreview();
  const active = state.isDragging || state.isResizing || state.isRotating ||
                 state.stickerStack.some(s => s.type === 'video' || s.type === 'lottie');
  const next = () => requestAnimationFrame(animationLoop);
  if (active) next();
  else setTimeout(next, 1000);
}
