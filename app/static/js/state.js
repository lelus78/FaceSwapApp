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
};

export const dom = {};

export function assignDomElements() {
    const ids = [
        'error-modal', 'progress-modal', 'progress-bar', 'progress-text', 'progress-title',
        'result-image-display', 'result-placeholder', 'meme-canvas',
        'download-btn', 'add-gallery-btn', 'download-anim-btn', 'anim-fmt', 'share-btn', 'reset-all-btn',
        'step-1-subject', 'fileInput', 'imagePreview', 'subject-upload-prompt',
        'prepare-subject-btn', 'skip-to-swap-btn',
        'step-2-scene', 'bg-prompt-input', 'auto-enhance-prompt-toggle', 'generate-scene-btn', 'goto-step-3-btn',
        'step-3-upscale', 'enable-hires-upscale-toggle', 'tile-denoising-slider', 'tile-denoising-value',
        'start-upscale-btn', 'skip-upscale-btn',
        'step-4-finalize', 'source-img-input', 'source-img-preview', 'source-upload-prompt',
        'toggle-face-boxes', 'source-face-boxes-container', 'target-face-boxes-container',
        'selection-status', 'selected-source-id', 'selected-target-id', 'swap-btn', 'back-to-step-3-btn',
        'theme-toggle', 'theme-icon'
    ];
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            const key = id.replace(/-(\w)/g, (_, c) => c.toUpperCase());
            dom[key] = el;
        } else {
            console.warn(`Elemento DOM non trovato con ID: #${id}`);
        }
    });
}