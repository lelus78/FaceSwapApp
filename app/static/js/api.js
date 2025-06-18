// API.JS - Modulo per la comunicazione con il server
// BASE_URL dinamica: funziona da qualsiasi host
const BASE_URL = window.location.origin;
const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');

function csrfFetch(url, options = {}) {
    options.headers = options.headers || {};
    if (csrfToken) {
        options.headers['X-CSRFToken'] = csrfToken;
    }
    return fetch(url, options);
}

async function handleResponse(response) {
    if (!response.ok) {
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") !== -1) {
            const errorData = await response.json();
            throw new Error(errorData.error || "Errore sconosciuto dal server.");
        } else {
            const errorText = await response.text();
            console.error("ERRORE HTML DAL SERVER:", errorText);
            throw new Error("Errore grave del server (non JSON). Controlla il terminale del server.");
        }
    }
    return response;
}

// Funzione helper per convertire Blob in Base64
async function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            // Rimuovi il prefisso "data:image/jpeg;base64,"
            const base64String = reader.result.split(',')[1];
            resolve(base64String);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

export async function getStickers() {
    const response = await csrfFetch(`${BASE_URL}/api/stickers`);
    await handleResponse(response);
    return response.json();
}

export async function createSceneAsync(processedSubjectBlob, finalPrompt, modelName) {
    const formData = new FormData();
    formData.append('subject_data', processedSubjectBlob);
    formData.append('prompt', finalPrompt);
    if (modelName) formData.append('model_name', modelName);

    // Chiama la nuova rotta asincrona
    const response = await csrfFetch(`${BASE_URL}/async/create_scene`, { method: 'POST', body: formData });
    await handleResponse(response);
    return response.json(); // Si aspetta una risposta JSON con il task_id
}

export async function prepareSubject(subjectFile) {
    const formData = new FormData();
    formData.append('subject_image', subjectFile);
    const response = await csrfFetch(`${BASE_URL}/prepare_subject`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.blob();
}

export async function createScene(processedSubjectBlob, finalPrompt, modelName) {
    const formData = new FormData();
    formData.append('subject_data', processedSubjectBlob);
    formData.append('prompt', finalPrompt);
    if (modelName) formData.append('model_name', modelName);
    const response = await csrfFetch(`${BASE_URL}/create_scene`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.blob();
}

export async function upscaleAndDetail(sceneImageBlob, enableHires, denoising) {
    const formData = new FormData();
    formData.append('scene_image', sceneImageBlob);
    formData.append('enable_hires', enableHires);
    formData.append('tile_denoising_strength', denoising);
    const response = await csrfFetch(`${BASE_URL}/detail_and_upscale`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.blob();
}

export async function detectFaces(imageBlob) {
    const formData = new FormData();
    formData.append('image', imageBlob);
    const response = await csrfFetch(`${BASE_URL}/detect_faces`, { method: 'POST', body: formData });
    await handleResponse(response);
    return response.json();
}

export async function performSwap(targetImageBlob, sourceImageFile, sourceIndex, targetIndex) {
    const formData = new FormData();
    formData.append('target_image_high_res', targetImageBlob);
    formData.append('source_face_image', sourceImageFile);
    formData.append('source_face_index', sourceIndex);
    formData.append('target_face_index', targetIndex);
    const response = await csrfFetch(`${BASE_URL}/final_swap`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.blob();
}

// Modificata per inviare l'immagine in base64
export async function enhancePrompt(imageBlob, userPrompt) {
    const base64ImageData = await blobToBase64(imageBlob);
    const response = await csrfFetch(`${BASE_URL}/enhance_prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: base64ImageData, prompt_text: userPrompt })
    });
    await handleResponse(response);
    return response.json();
}

// Modificata per inviare l'immagine in base64
export async function enhancePartPrompt(partName, userPrompt, imageBlob) {
    const base64ImageData = await blobToBase64(imageBlob);
    const response = await csrfFetch(`${BASE_URL}/enhance_part_prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ part_name: partName, prompt_text: userPrompt, image_data: base64ImageData })
    });
    await handleResponse(response);
    return response.json();
}

export async function generateCaption(imageBlob, tone) {
    const base64ImageData = await blobToBase64(imageBlob); // Usa la nuova helper
    const response = await csrfFetch(`${BASE_URL}/meme/generate_caption`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: base64ImageData, tone: tone })
    });
    await handleResponse(response);
    return response.json();
}

export async function saveResultVideo(videoBlob, format) {
    const response = await csrfFetch(`${BASE_URL}/save_result_video?fmt=${format}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/octet-stream' 
        },
        body: videoBlob,
        cache: 'no-cache'
    });
    await handleResponse(response);
    return response.json(); 
}

export async function generateWithMask(imageBlob, partName, prompt) { // Aggiunto partName qui
    const formData = new FormData();
    formData.append('image', imageBlob);
    formData.append('part_name', partName); // Passa part_name
    formData.append('prompt', prompt);
    // Cambiato l'endpoint da generate_with_mask a generate_all_parts
    // per coerenza con il nuovo inpainting basato sul crop
    const response = await csrfFetch(`${BASE_URL}/generate_all_parts`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.blob();
}

export async function analyzeParts(imageBlob) {
    const formData = new FormData();
    formData.append('image', imageBlob);
    const response = await csrfFetch(`${BASE_URL}/analyze_parts`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.json();
}

export async function generateAllParts(imageBlob, prompts) {
    const formData = new FormData();
    formData.append('image', imageBlob);
    formData.append('prompts', JSON.stringify(prompts));
    
    // NOTA: il problema precedente era che questo chiamava /generate_with_mask.
    // Assicurati che qui chiami generate_all_parts
    const response = await csrfFetch(`${BASE_URL}/generate_all_parts`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.blob();
}

export async function generateAllPartsAsync(imageBlob, prompts) {
    const formData = new FormData();
    formData.append('image', imageBlob);
    formData.append('prompts', JSON.stringify(prompts));
    const response = await csrfFetch(`${BASE_URL}/async/generate_all_parts`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.json();
}

export async function finalSwapAsync(targetBlob, sourceFile, sourceIndex, targetIndex) {
    const formData = new FormData();
    formData.append('target_image_high_res', targetBlob);
    formData.append('source_face_image', sourceFile);
    formData.append('source_face_index', sourceIndex);
    formData.append('target_face_index', targetIndex);
    const response = await csrfFetch(`${BASE_URL}/async/final_swap`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.json();
}

export async function getTaskStatus(taskId) {
    const response = await csrfFetch(`${BASE_URL}/task_status/${taskId}`);
    await handleResponse(response);
    return response.json();
}

/**
 * Avvia il processo di detailing e upscale in modo asincrono.
 * @param {Blob} sceneImageBlob - L'immagine della scena da elaborare.
 * @param {boolean} enableHires - Flag per abilitare o meno l'upscaling.
 * @param {number} denoisingStrength - La forza del denoising.
 * @returns {Promise<Object>} - Una promessa che si risolve con l'ID del task.
 */
export async function detailAndUpscaleAsync(sceneImageBlob, enableHires, denoisingStrength) {
    const formData = new FormData();
    formData.append('scene_image', sceneImageBlob);
    formData.append('enable_hires', enableHires);
    formData.append('tile_denoising_strength', denoisingStrength);

    const response = await csrfFetch(`${BASE_URL}/async/detail_and_upscale`, { method: 'POST', body: formData });
    await handleResponse(response);
    return response.json(); // Restituisce { task_id: "..." }
}