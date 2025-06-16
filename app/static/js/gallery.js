export async function loadGallery(container) {
  try {
    const res = await fetch(`${window.location.origin}/api/approved_memes`);
    const items = await res.json();
    renderGallery(container, items);
  } catch (err) {
    console.error('Errore caricamento galleria', err);
  }
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
    img.className = 'w-full h-full object-cover rounded';
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
    if (copy) {
      navigator.clipboard.writeText(copy.dataset.url || '').catch(() => {});
    } else if (remove) {
      remove.closest('.gallery-item')?.remove();
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
