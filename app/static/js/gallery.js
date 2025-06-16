const GALLERY_KEY = 'galleryData';
const USERNAME = localStorage.getItem('username') || 'user';

// Funzione da 'codex' per generare ID univoci
export function generateId() {
    return (crypto.randomUUID && crypto.randomUUID()) ||
        (Date.now().toString(36) + Math.random().toString(36).slice(2));
}

// Funzioni helper da 'ip-adapter' per la gestione dei dati
function getGalleryData() {
    return JSON.parse(localStorage.getItem(GALLERY_KEY) || '{}');
}

function saveGalleryData(data) {
    localStorage.setItem(GALLERY_KEY, JSON.stringify(data));
}

/**
 * Migra i dati della galleria per assicurare che ogni elemento abbia un ID univoco.
 * Questa funzione combina la logica di migrazione di 'codex' con la struttura dati di 'ip-adapter'.
 */
function migrateData() {
    const data = getGalleryData();
    let changed = false;
    Object.values(data).forEach(userItems => {
        Object.values(userItems).forEach(dateItems => {
            Object.values(dateItems).forEach(tagItems => {
                tagItems.forEach(item => {
                    if (!item.id) {
                        item.id = generateId();
                        changed = true;
                    }
                });
            });
        });
    });
    if (changed) {
        saveGalleryData(data);
    }
    return data;
}

/**
 * Appiattisce la struttura dati nidificata in un array semplice per la renderizzazione.
 * Modificato per includere il percorso e l'ID.
 */
function flattenUserData(data) {
    const items = [];
    Object.entries(data).forEach(([date, tags]) => {
        Object.entries(tags).forEach(([tag, arr]) => {
            arr.forEach((item, idx) => {
                items.push({
                    ...item,
                    path: { user: USERNAME, date, tag, index: idx },
                    local: true,
                });
            });
        });
    });
    return items;
}

export async function loadGallery(container) {
    const migratedData = migrateData(); // Esegue la migrazione prima di caricare
    const local = flattenUserData(migratedData[USERNAME] || {});
    let serverItems = [];
    try {
        const res = await fetch(`${window.location.origin}/api/approved_memes`);
        serverItems = await res.json();
    } catch (err) {
        console.error('Errore caricamento galleria', err);
    }
    renderGallery(container, [...local, ...serverItems]);
}

function renderGallery(container, items) {


    if (!container) return;
    container.innerHTML = '';
    items.forEach(m => {
        const card = document.createElement('div');
        card.className = 'relative group gallery-item';
        // Assicura che l'ID sia sempre presente nel dataset
        if (m.id) card.dataset.id = m.id;
        const img = document.createElement('img');
        img.src = m.url;
        img.alt = m.title || 'meme';
        img.className = 'w-full h-full object-cover rounded cursor-pointer';
        if (m.local) {
            img.dataset.local = '1';
            // Assicura che il percorso sia presente per le operazioni
            if (m.path) card.dataset.path = JSON.stringify(m.path);
        }
        const overlay = document.createElement('div');
        overlay.className = 'gallery-item-overlay';
        const title = document.createElement('p');
        title.className = 'text-xs text-white truncate';
        title.textContent = m.title || '';
        const time = document.createElement('p');
        time.className = 'text-[10px] text-gray-300';
        time.textContent = m.ts ? new Date(m.ts).toLocaleString() : '';
        const actions = document.createElement('div');
        actions.className = 'flex justify-end gap-1';
        actions.innerHTML = `
            <button class="copy-link p-1 hover:text-green-400" data-url="${m.url}" aria-label="Copia link">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M8 17v-2a4 4 0 014-4h8"/><path d="M16 7v2a4 4 0 01-4 4H4"/></svg>
            </button>
            <a class="download-item p-1 hover:text-blue-400" download="meme.png" href="${m.url}" aria-label="Scarica">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2"/><path d="M7 10l5 5 5-5M12 4v11"/></svg>
            </a>
            <button class="toggle-share p-1" aria-label="Condividi">
                <svg class="w-4 h-4 ${m.shared ? 'text-yellow-400 fill-yellow-400' : ''}" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/>
                </svg>
            </button>
            <button class="remove-item p-1 hover:text-red-500" aria-label="Rimuovi">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 18L18 6M6 6l12 12"/></svg>
            </button>`;
        overlay.appendChild(title);
        overlay.appendChild(time);
        overlay.appendChild(actions);
        card.appendChild(img);
        card.appendChild(overlay);
        container.appendChild(card);
    });
}

export function setupGalleryInteraction(container) {

    if (!container) return;
    container.addEventListener('click', e => {
        const copy = e.target.closest('.copy-link');
        const remove = e.target.closest('.remove-item');
        const toggle = e.target.closest('.toggle-share');
        const download = e.target.closest('.download-item');
        const img = e.target.closest('.gallery-item img');
        if (copy) {
            navigator.clipboard.writeText(copy.dataset.url || '').catch(() => { });
        } else if (download) {
            // anchor handles download
        } else if (toggle) {
            const card = toggle.closest('.gallery-item');
            const path = card && card.dataset.path ? JSON.parse(card.dataset.path) : null;
            if (path) {
                const data = getGalleryData();
                const item = data[path.user]?.[path.date]?.[path.tag]?.[path.index];
                if (item) {
                    item.shared = !item.shared;
                    saveGalleryData(data);
                    const svg = toggle.querySelector('svg');
                    svg.classList.toggle('text-yellow-400', item.shared);
                    svg.classList.toggle('fill-yellow-400', item.shared);
                    window.dispatchEvent(new Event('gallery-updated'));
                }
            }
        } else if (remove) {
            const card = remove.closest('.gallery-item');
            if (card?.querySelector('img')?.dataset.local === '1') {
                // Usa la logica di 'ip-adapter' che è più coerente con la struttura dati
                const path = card.dataset.path ? JSON.parse(card.dataset.path) : null;
                if (path) {
                    const data = getGalleryData();
                    const arr = data[path.user]?.[path.date]?.[path.tag];
                    if (Array.isArray(arr)) {
                        // Rimuove l'elemento usando l'indice dal percorso
                        arr.splice(path.index, 1);
                        saveGalleryData(data);
                    }
                }
            }
            card?.remove();
        } else if (img) {
            const modal = document.getElementById('gallery-modal');
            const modalImg = document.getElementById('galleryModalImg') || document.getElementById('gallery-modal-img');
            if (modal && modalImg) {
                modalImg.src = img.src;
                modal.style.display = 'flex';
            }
        }
    });
}

export function initSidebarToggle(sidebar, toggleBtn, galleryToggle, galleryContainer) {
    if (toggleBtn) {
        toggleBtn.setAttribute('aria-expanded', !sidebar.classList.contains('hidden'));
        toggleBtn.addEventListener('click', () => {
            const hidden = sidebar.classList.toggle('hidden');
            toggleBtn.setAttribute('aria-expanded', !hidden);
        });
    }
    if (galleryToggle && galleryContainer) {
        galleryToggle.addEventListener('click', () => galleryContainer.classList.toggle('hidden'));
    }
}

function showToast(msg) {
    const t = document.getElementById('toast');
    if (!t) return;
    t.textContent = msg; t.classList.remove('hidden');
    setTimeout(() => t.classList.add('hidden'), 2000);
}

async function embedCaption(imgUrl, text) {
    if (!text) return imgUrl;
    return new Promise(resolve => {
        const img = new Image();
        img.onload = () => {
            const c = document.createElement('canvas');
            c.width = img.width; c.height = img.height;
            const ctx = c.getContext('2d');
            ctx.drawImage(img, 0, 0);
            const size = Math.max(24, c.width * 0.05);
            ctx.font = `bold ${size}px Impact`;
            ctx.fillStyle = 'white';
            ctx.strokeStyle = 'black';
            ctx.lineWidth = size / 10;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.strokeText(text, c.width / 2, c.height - 10);
            ctx.fillText(text, c.width / 2, c.height - 10);
            resolve(c.toDataURL('image/png'));
        };
        img.crossOrigin = 'anonymous';
        img.src = imgUrl;
    });
}


export async function loadExplore(container) {


    // Logica di 'ip-adapter' per caricare solo gli elementi condivisi
    const data = getGalleryData();
    const local = [];
    Object.values(data).forEach(dates => {
        Object.values(dates).forEach(tags => {
            Object.values(tags).forEach(arr => {
                arr.forEach(item => { if (item.shared) local.push(item); });
            });
        });
    });



    let server = [];
    try { const r = await fetch(`${window.location.origin}/api/approved_memes`); server = await r.json(); } catch { }
    const items = [...local, ...server.filter(m => m.shared)].sort((a, b) => (b.ts || 0) - (a.ts || 0));
    let index = 0; const batch = 12; let filtered = items;
    function renderSlice(reset = false) {
        if (reset) { container.innerHTML = ''; index = 0; }
        const slice = filtered.slice(index, index + batch); index += slice.length;
        slice.forEach(m => {
            const card = document.createElement('div');
            card.className = 'relative group gallery-item';
            if (m.id) card.dataset.id = m.id;
            const img = document.createElement('img');
            img.src = m.url; img.alt = m.title || ''; img.className = 'w-full h-full object-cover rounded cursor-pointer';
            const overlay = document.createElement('div');
            overlay.className = 'gallery-item-overlay';
            const p = document.createElement('p'); p.className = 'text-xs text-white truncate'; p.textContent = m.title || '';
            const time = document.createElement('p'); time.className = 'text-[10px] text-gray-300'; time.textContent = m.ts ? new Date(m.ts).toLocaleString() : '';
            overlay.appendChild(p); overlay.appendChild(time); card.appendChild(img); card.appendChild(overlay); container.appendChild(card);
        });
    }
    function fetchMore() { if (index < filtered.length) renderSlice(); }
    function applyFilter(term, forceReload = false) {
        filtered = items.filter(m => (m.title || '').toLowerCase().includes(term));
        if (forceReload) { index = 0; }
        renderSlice(true);
        if (index < filtered.length) {/* keep going for first batch*/}
    }
    renderSlice();
    container.addEventListener('click', e => {
        const img = e.target.closest('.gallery-item img');
        if (img) {
            const modal = document.getElementById('gallery-modal');
            const modalImg = document.getElementById('gallery-modal-img');
            if (modal && modalImg) { modalImg.src = img.src; modal.style.display = 'flex'; }
        }
    });
    return { fetchMore, applyFilter };
}

export async function addToGallery(title, dataUrl, caption = '', tags = [], shared = false) {
    const withText = await embedCaption(dataUrl, caption);
    const ts = Date.now();
    const date = new Date(ts).toISOString().slice(0, 10);
    const mainTag = (tags[0] || 'misc').toLowerCase();
    const data = getGalleryData();
    data[USERNAME] = data[USERNAME] || {};
    data[USERNAME][date] = data[USERNAME][date] || {};
    data[USERNAME][date][mainTag] = data[USERNAME][date][mainTag] || [];
    
    // Aggiunge il nuovo elemento con un ID univoco
    data[USERNAME][date][mainTag].push({ id: generateId(), title, url: withText, caption, tags, ts, shared });
    
    saveGalleryData(data);
    window.dispatchEvent(new Event('gallery-updated'));
    showToast('Salvato nella galleria');
}