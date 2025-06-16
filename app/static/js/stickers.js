import { state, dom } from './state.js';
import * as api from './api.js';
import { showError } from './workflow.js';
import { drawSticker } from './memeEditor.js';

export async function addStickerToCanvas(element, isVideo, isLottie, path) {
  const stickerData = { type: isVideo ? 'video' : (isLottie ? 'lottie' : 'image'), x: 20, y: 20, rotation: 0 };
  let stickerElement, naturalW, naturalH;
  if (isLottie) {
    try {
      const animationData = await (await fetch(`/lottie_json/${path.replace('static/', '')}`)).json();
      naturalW = animationData.w;
      naturalH = animationData.h;
      stickerElement = document.createElement('canvas');
      stickerElement.width = naturalW;
      stickerElement.height = naturalH;
      lottie.loadAnimation({
        renderer: 'canvas',
        loop: true,
        autoplay: true,
        animationData,
        rendererSettings: { context: stickerElement.getContext('2d'), clearCanvas: true }
      });
    } catch (err) {
      return showError('Errore Lottie', 'Impossibile caricare l\'animazione.');
    }
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
    state.stickerStack.push(stickerData);
    state.selectedSticker = stickerData;
    if (isVideo || isLottie) {
      dom.downloadAnimBtn.classList.remove('hidden');
      dom.animFmt.classList.remove('hidden');
    }
  }
}

export async function loadStickers() {
  const gallery = dom.stickerGallery;
  gallery.innerHTML = '<p class="text-gray-500">Caricamento...</p>';
  try {
    const categories = await api.getStickers();
    gallery.innerHTML = '';
    if (categories.length === 0) {
      gallery.innerHTML = '<p class="text-gray-500">Nessuno sticker trovato.</p>';
      return;
    }
    categories.forEach(category => {
      const title = document.createElement('h4');
      title.className = 'font-bold text-sm text-blue-400 mt-2 mb-1 px-1 category-title w-full';
      title.textContent = category.category;
      gallery.appendChild(title);
      const container = document.createElement('div');
      container.className = 'flex flex-wrap gap-2 sticker-container';
      gallery.appendChild(container);
      category.stickers.forEach(path => {
        const isVideo = path.endsWith('.webm');
        const isLottie = path.endsWith('.tgs');
        const wrapper = document.createElement('div');
        wrapper.className = 'sticker-item-wrapper relative';
        let el;
        if (isLottie) {
          el = document.createElement('div');
          lottie.loadAnimation({ container: el, renderer: 'svg', loop: true, autoplay: true, path: `/lottie_json/${path.replace('static/', '')}` });
        } else {
          el = isVideo ? document.createElement('video') : document.createElement('img');
          if (isVideo) { el.autoplay = el.muted = el.loop = el.playsInline = true; }
          else { el.crossOrigin = 'anonymous'; }
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
    showError('Errore Sticker', err.message);
  }
}

export function getStickerAtPosition(x, y) {
  const handleRadius = window.matchMedia?.('(pointer: coarse)').matches ? 40 : 20;
  for (let i = state.stickerStack.length - 1; i >= 0; i--) {
    const s = state.stickerStack[i],
      cx = s.x + s.width / 2,
      cy = s.y + s.height / 2;
    const rotHandle = rotatePoint(cx, cy - s.height / 2 - 25, cx, cy, s.rotation);
    const resizeHandle = rotatePoint(s.x + s.width, s.y + s.height, cx, cy, s.rotation);
    if (distance(x, y, rotHandle.x, rotHandle.y) < handleRadius) return { sticker: s, corner: 'rotate' };
    if (distance(x, y, resizeHandle.x, resizeHandle.y) < handleRadius) return { sticker: s, corner: 'resize' };
    if (isPointInRotatedRectangle({ x, y }, s)) return { sticker: s, corner: 'drag' };
  }
  return null;
}

function isPointInRotatedRectangle(point, rect) {
  const cx = rect.x + rect.width / 2,
    cy = rect.y + rect.height / 2;
  const { x, y } = rotatePoint(point.x, point.y, cx, cy, -rect.rotation);
  return x > rect.x && x < rect.x + rect.width && y > rect.y && y < rect.y + rect.height;
}

function rotatePoint(x, y, cx, cy, angle) {
  const cos = Math.cos(angle), sin = Math.sin(angle);
  return { x: cos * (x - cx) - sin * (y - cy) + cx, y: sin * (x - cx) + cos * (y - cy) + cy };
}

function distance(x1, y1, x2, y2) {
  return Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2));
}
