export const state = {
  currentStep: 1,
  subjectFile: null,
  processedSubjectBlob: null,
  sceneImageBlob: null,
  upscaledImageBlob: null,
  finalImageWithSwap: null,
  sourceFaces: [],
  targetFaces: [],
  selectedSourceIndex: -1,
  selectedTargetIndex: -1,
  sourceImageFile: null,
  activeFilter: 'none',
  stickerStack: [],
  selectedSticker: null,
  isDragging: false,
  isResizing: false,
  isRotating: false,
  dragOffsetX: 0,
  dragOffsetY: 0,
  dom: {}
};

export const dom = state.dom;

export function assignDomElements() {
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
    'theme-toggle', 'theme-icon',
    'sidebar', 'sidebar-toggle', 'gallery-toggle', 'gallery-container'
  ];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (!el) console.warn(`Elemento non trovato: #${id}`);
    state.dom[id.replace(/-(\w)/g, (_, c) => c.toUpperCase())] = el;
  });
}
