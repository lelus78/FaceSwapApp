// Contenuto completo per app/static/js/script.js

document.addEventListener('DOMContentLoaded', () => {
    // --- 1. STATO GLOBALE E DOM ---
    const state = {
        subjectFile: null,
        sourceImageFile: null,
        upscaledImageBlob: null,
        sourceFaces: [],
        targetFaces: [],
        selectedSourceIndex: -1,
        selectedTargetIndex: -1,
    };

    const dom = {};
    const ids = [
        'imagePreview', 'subjectUploadPrompt', 'prepareSubjectBtn', 'skipToSwapBtn',
        'result-image-display', 'result-placeholder',
        'source-img-preview', 'source-upload-prompt', 'sourceFaceBoxesContainer', 'targetFaceBoxesContainer',
        'selected-source-id', 'selected-target-id', 'swapBtn',
        'fileInput', 'source-img-input'
    ];
    ids.forEach(id => {
        dom[id] = document.getElementById(id);
    });

    // --- 2. FUNZIONI CORE ---

    function displayImage(blob, imageElement, callback) {
        if (!imageElement || !blob) return;
        const newElement = imageElement.cloneNode(true);
        imageElement.parentNode.replaceChild(newElement, imageElement);
        dom[newElement.id] = newElement;

        newElement.onload = callback;
        newElement.src = URL.createObjectURL(blob);
        newElement.classList.remove('hidden');

        if (newElement.id === 'imagePreview') {
            dom.subjectUploadPrompt.style.display = 'none';
        } else if (newElement.id === 'source-img-preview') {
            dom.sourceUploadPrompt.style.opacity = '0';
        } else if (newElement.id === 'result-image-display') {
            dom.resultPlaceholder.classList.add('hidden');
        }
    }

    function drawFaceBoxes(container, image, faces, type) {
        if (!container || !image || !image.complete || image.naturalWidth === 0) {
            if (container) container.innerHTML = '';
            return;
        }
        container.innerHTML = '';
        const imageRect = image.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        if (imageRect.width === 0) return;

        const scaleX = imageRect.width / image.naturalWidth;
        const scaleY = imageRect.height / image.naturalHeight;
        const offsetX = imageRect.left - containerRect.left;
        const offsetY = imageRect.top - containerRect.top;

        faces.forEach((face, idx) => {
            const [x1, y1, x2, y2] = face.bbox;
            const box = document.createElement('div');
            box.className = 'face-box';
            box.style.left = `${offsetX + (x1 * scaleX)}px`;
            box.style.top = `${offsetY + (y1 * scaleY)}px`;
            box.style.width = `${(x2 - x1) * scaleX}px`;
            box.style.height = `${(y2 - y1) * scaleY}px`;
            
            const label = document.createElement('span');
            label.className = 'face-box-label';
            label.textContent = idx + 1;
            box.appendChild(label);

            box.onclick = () => handleFaceSelection(idx, type);
            container.appendChild(box);
        });
    }
    
    function updateSelectionHighlights(container, selectedIndex) {
        if (!container) return;
        container.querySelectorAll('.face-box').forEach((box, i) => {
            box.classList.toggle('selected', i === selectedIndex);
        });
    }
    
    function handleFaceSelection(index, type) {
        if (type === 'source') {
            state.selectedSourceIndex = (state.selectedSourceIndex === index) ? -1 : index;
            updateSelectionHighlights(dom.sourceFaceBoxesContainer, state.selectedSourceIndex);
        } else {
            state.selectedTargetIndex = (state.selectedTargetIndex === index) ? -1 : index;
            updateSelectionHighlights(dom.targetFaceBoxesContainer, state.selectedTargetIndex);
        }
        dom.selectedSourceId.textContent = state.selectedSourceIndex > -1 ? state.selectedSourceIndex + 1 : 'Nessuno';
        dom.selectedTargetId.textContent = state.selectedTargetIndex > -1 ? state.selectedTargetIndex + 1 : 'Nessuno';
        dom.swapBtn.disabled = state.selectedSourceIndex < 0 || state.selectedTargetIndex < 0;
    }

    async function detectFacesAndDraw(blob, imageElement, boxesContainer, faceArray, type) {
        try {
            const formData = new FormData();
            formData.append('image', blob);
            const response = await fetch('/detect_faces', { method: 'POST', body: formData });
            if (!response.ok) throw new Error(`Errore server: ${response.statusText}`);
            const data = await response.json();
            
            faceArray.length = 0;
            faceArray.push(...data.faces);
            drawFaceBoxes(boxesContainer, imageElement, faceArray, type);
        } catch (err) {
            console.error(`Errore rilevamento volti (${type}):`, err);
        }
    }

    // --- 3. EVENT LISTENERS ---

    dom.fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        state.subjectFile = file;
        displayImage(file, dom.imagePreview);
        dom.prepareSubjectBtn.disabled = false;
        dom.skipToSwapBtn.disabled = false;
    });

    dom.sourceImgInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        state.sourceImageFile = file;
        displayImage(file, dom.sourceImgPreview, () => {
            detectFacesAndDraw(file, dom.sourceImgPreview, dom.sourceFaceBoxesContainer, state.sourceFaces, 'source');
        });
    });

    // Aggiungi qui gli altri listener principali...
});