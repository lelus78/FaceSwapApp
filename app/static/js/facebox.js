import { state, dom } from './state.js';
import * as api from './api.js';
import { showError } from './workflow.js';

export function drawFaceBoxes(container, image, faces, type) {
  if (!container || !image || !image.complete || image.naturalWidth === 0) return;
  container.innerHTML = '';
  const rect = container.getBoundingClientRect();
  if (!rect.width) return;
  const scaleX = rect.width / image.naturalWidth;
  const scaleY = rect.height / image.naturalHeight;
  faces.forEach((face, idx) => {
    const [x1, y1, x2, y2] = face.bbox;
    const box = document.createElement('div');
    box.className = 'face-box';
    box.style.left = `${x1 * scaleX}px`;
    box.style.top = `${y1 * scaleY}px`;
    box.style.width = `${(x2 - x1) * scaleX}px`;
    box.style.height = `${(y2 - y1) * scaleY}px`;
    const label = document.createElement('span');
    label.className = 'face-box-label';
    label.textContent = idx + 1;
    box.appendChild(label);
    box.onclick = e => { e.stopPropagation(); handleFaceSelection(idx, type); };
    container.appendChild(box);
  });
}

export function updateSelectionHighlights(container, selectedIndex) {
  if (!container) return;
  container.querySelectorAll('.face-box').forEach((b,i)=>b.classList.toggle('selected', i===selectedIndex));
}

export function refreshFaceBoxes() {
  drawFaceBoxes(dom.sourceFaceBoxesContainer, dom.sourceImgPreview, state.sourceFaces, 'source');
  drawFaceBoxes(dom.targetFaceBoxesContainer, dom.resultImageDisplay, state.targetFaces, 'target');
}

export async function detectAndDrawFaces(blob, image, container, faces, type) {
  const onLoad = async () => {
    try {
      const data = await api.detectFaces(blob);
      faces.splice(0, faces.length, ...data.faces);
      drawFaceBoxes(container, image, faces, type);
      updateSelectionHighlights(container, type==='source'?state.selectedSourceIndex:state.selectedTargetIndex);
    } catch (err) {
      showError('Errore Rilevamento Volti', err.message);
      faces.length = 0;
      drawFaceBoxes(container, image, [], type);
    }
  };
  if (image.complete && image.naturalWidth > 0) onLoad();
  else image.onload = onLoad;
}

function handleFaceSelection(index, type) {
  if (type === 'source') {
    state.selectedSourceIndex = index;
  } else {
    state.selectedTargetIndex = index;
  }
  updateSelectionHighlights(type==='source'?dom.sourceFaceBoxesContainer:dom.targetFaceBoxesContainer, index);
  dom.selectionStatus.classList.remove('hidden');
  dom.selectedSourceId.textContent = state.selectedSourceIndex + 1 || 'Nessuno';
  dom.selectedTargetId.textContent = state.selectedTargetIndex + 1 || 'Nessuno';
  dom.swapBtn.disabled = state.selectedSourceIndex < 0 || state.selectedTargetIndex < 0;
}

export function initFaceBoxObservers() {
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(refreshFaceBoxes);
    [dom.resultImageDisplay, dom.sourceImgPreview].forEach(el => el && ro.observe(el));
  }
  window.addEventListener('resize', refreshFaceBoxes);
}
