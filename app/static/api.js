// API.JS - Modulo per la comunicazione con il server
// BASE_URL dinamica: funziona da qualsiasi host
const BASE_URL = window.location.origin;

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

export async function getStickers() {
    const response = await fetch(`${BASE_URL}/api/stickers`);
    await handleResponse(response);
    return response.json();
}

export async function prepareSubject(subjectFile) {
    const formData = new FormData();
    formData.append('subject_image', subjectFile);
    const response = await fetch(`${BASE_URL}/prepare_subject`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.blob();
}

export async function createScene(processedSubjectBlob, finalPrompt) {
    const formData = new FormData();
    formData.append('subject_data', processedSubjectBlob);
    formData.append('prompt', finalPrompt);
    const response = await fetch(`${BASE_URL}/create_scene`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.blob();
}

export async function upscaleAndDetail(sceneImageBlob, enableHires, denoising) {
    const formData = new FormData();
    formData.append('scene_image', sceneImageBlob);
    formData.append('enable_hires', enableHires);
    formData.append('tile_denoising_strength', denoising);
    const response = await fetch(`${BASE_URL}/detail_and_upscale`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.blob();
}

export async function detectFaces(imageBlob) {
    const formData = new FormData();
    formData.append('image', imageBlob);
    const response = await fetch(`${BASE_URL}/detect_faces`, { method: 'POST', body: formData });
    await handleResponse(response);
    return response.json();
}

export async function performSwap(targetImageBlob, sourceImageFile, sourceIndex, targetIndex) {
    const formData = new FormData();
    formData.append('target_image_high_res', targetImageBlob);
    formData.append('source_face_image', sourceImageFile);
    formData.append('source_face_index', sourceIndex);
    formData.append('target_face_index', targetIndex);
    const response = await fetch(`${BASE_URL}/final_swap`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.blob();
}

export async function enhancePrompt(imageBlob, userPrompt) {
    // Note: imageBlob is not used by the server's enhance_prompt endpoint currently
    // The server-side enhance_prompt is a generic text enhancement.
    // If you want image-contextual prompt enhancement, you'd send image data to Gemini.
    const response = await fetch(`${BASE_URL}/enhance_prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt_text: userPrompt })
    });
    await handleResponse(response);
    return response.json();
}

export async function enhancePartPrompt(partName, userPrompt) {
    const response = await fetch(`${BASE_URL}/enhance_part_prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ part_name: partName, prompt_text: userPrompt })
    });
    await handleResponse(response);
    return response.json();
}

export async function generateCaption(imageBlob, tone) {
    const reader = new FileReader();
    reader.readAsDataURL(imageBlob);
    const base64ImageData = await new Promise(resolve => { reader.onloadend = () => resolve(reader.result.split(',')[1]); });
    const response = await fetch(`${BASE_URL}/meme/generate_caption`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: base64ImageData, tone: tone })
    });
    await handleResponse(response);
    return response.json();
}

export async function saveResultVideo(videoBlob, format) {
    const response = await fetch(`${BASE_URL}/save_result_video?fmt=${format}`, {
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

// NOTE: This function is now for single-part generation (if needed),
// otherwise use generateAllParts.
export async function generateWithMask(imageBlob, partName, prompt) {
    const formData = new FormData();
    formData.append('image', imageBlob);
    formData.append('part_name', partName); // Send the specific part name
    formData.append('prompt', prompt);
    const response = await fetch(`${BASE_URL}/generate_with_mask`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.blob();
}

export async function analyzeParts(imageBlob) {
    const formData = new FormData();
    formData.append('image', imageBlob);
    const response = await fetch(`${BASE_URL}/analyze_parts`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.json();
}

export async function generateAllParts(imageBlob, prompts) {
    const formData = new FormData();
    formData.append('image', imageBlob);
    formData.append('prompts', JSON.stringify(prompts)); // Send the prompts object as JSON string
    
    // Updated endpoint to call the new /generate_all_parts
    const response = await fetch(`${BASE_URL}/generate_all_parts`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.blob();
}