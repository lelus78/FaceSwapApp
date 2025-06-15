export async function generateAllParts(imageBlob, prompts) {
    const formData = new FormData();
    formData.append('image', imageBlob);
    formData.append('prompts', JSON.stringify(prompts));
    
    // NOTA: il problema precedente era che questo chiamava /generate_with_mask.
    // Assicurati che qui chiami generate_all_parts
    const response = await fetch(`${BASE_URL}/generate_all_parts`, { method: 'POST', body: formData, cache: 'no-cache' });
    await handleResponse(response);
    return response.blob();
}