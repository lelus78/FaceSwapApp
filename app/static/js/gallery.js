export async function loadGallery(container) {
  const local = JSON.parse(localStorage.getItem('userGallery') || '[]');
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
    const img = document.createElement('img');
    img.src = m.url;
    img.alt = m.title || 'meme';
    img.className = 'w-full h-full object-cover rounded cursor-pointer';
    if (m.local) img.dataset.local = '1';
    const overlay = document.createElement('div');
    overlay.className = 'gallery-item-overlay';
    const title = document.createElement('p');
    title.className = 'text-xs text-white truncate';
    title.textContent = m.title || '';
    const actions = document.createElement('div');
    actions.className = 'flex justify-end gap-1';
    actions.innerHTML = `
      <button class="copy-link p-1 hover:text-green-400" data-url="${m.url}" aria-label="Copia link">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M8 17v-2a4 4 0 014-4h8"/><path d="M16 7v2a4 4 0 01-4 4H4"/></svg>
      </button>
      <a class="download-item p-1 hover:text-blue-400" download="meme.png" href="${m.url}" aria-label="Scarica">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2"/><path d="M7 10l5 5 5-5M12 4v11"/></svg>
      </a>
      <button class="remove-item p-1 hover:text-red-500" aria-label="Rimuovi">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 18L18 6M6 6l12 12"/></svg>
      </button>`;
    overlay.appendChild(title);
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
    const download = e.target.closest('.download-item');
    const img = e.target.closest('.gallery-item img');
    if (copy) {
      navigator.clipboard.writeText(copy.dataset.url || '').catch(() => {});
    } else if (download) {
      // anchor handles download
    } else if (remove) {
      const card = remove.closest('.gallery-item');
      if (card?.querySelector('img')?.dataset.local === '1') {
        const list = JSON.parse(localStorage.getItem('userGallery') || '[]');
        const idx = Array.from(container.children).indexOf(card);
        list.splice(idx, 1);
        localStorage.setItem('userGallery', JSON.stringify(list));
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
    toggleBtn.addEventListener('click', () => sidebar.classList.toggle('hidden'));
  }
  if (galleryToggle && galleryContainer) {
    galleryToggle.addEventListener('click', () => galleryContainer.classList.toggle('hidden'));
  }
}

export function addToGallery(title, dataUrl) {
  const list = JSON.parse(localStorage.getItem('userGallery') || '[]');
  list.push({ title, url: dataUrl, local: true });
  localStorage.setItem('userGallery', JSON.stringify(list));
  const container = document.getElementById('gallery-container');
  if (container) renderGallery(container, list);
}
