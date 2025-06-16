export async function loadGallery(container, filters = {}) {
  const items = await fetchGalleryItems();
  renderGallery(container, applyFilters(items, filters));
  return items;
}

async function fetchGalleryItems() {
  const local = JSON.parse(localStorage.getItem('userGallery') || '[]');
  let serverItems = [];
  try {
    const res = await fetch(`${window.location.origin}/api/approved_memes`);
    serverItems = await res.json();
  } catch (err) {
    console.error('Errore caricamento galleria', err);
  }
  return [...local.map(m=>({ ...m, local: true })), ...serverItems.map(m=>({ ...m, local: false }))];
}

function extractTags(item) {
  if (Array.isArray(item.tags)) return item.tags.map(t=>t.toLowerCase());
  const txt = item.caption || '';
  return (txt.match(/#(\w+)/g) || []).map(t=>t.slice(1).toLowerCase());
}

function applyFilters(items, opts = {}) {
  const { search = '', tags = [], start = null, end = null, local = true, shared = true } = opts;
  const tagList = tags.map(t=>t.toLowerCase());
  return items.filter(it => {
    if (!local && it.local) return false;
    if (!shared && !it.local) return false;
    const title = (it.title || '').toLowerCase();
    if (search && !title.includes(search)) return false;
    const ts = it.ts || 0;
    if (start && ts < start) return false;
    if (end && ts > end) return false;
    if (tagList.length) {
      const itemTags = extractTags(it);
      if (!tagList.every(t => itemTags.includes(t))) return false;
    }
    return true;
  });
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

function showToast(msg) {
  const t = document.getElementById('toast');
  if (!t) return; 
  t.textContent = msg; t.classList.remove('hidden');
  setTimeout(()=>t.classList.add('hidden'), 2000);
}

async function embedCaption(imgUrl, text) {
  if (!text) return imgUrl;
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width = img.width; c.height = img.height;
      const ctx = c.getContext('2d');
      ctx.drawImage(img,0,0);
      const size = Math.max(24, c.width * 0.05);
      ctx.font = `bold ${size}px Impact`;
      ctx.fillStyle = 'white';
      ctx.strokeStyle = 'black';
      ctx.lineWidth = size/10;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.strokeText(text, c.width/2, c.height - 10);
      ctx.fillText(text, c.width/2, c.height -10);
      resolve(c.toDataURL('image/png'));
    };
    img.crossOrigin='anonymous';
    img.src = imgUrl;
  });
}

export async function loadExplore(container, filters = {}) {
  const items = (await fetchGalleryItems()).sort((a,b)=>(b.ts||0)-(a.ts||0));
  let index=0; const batch=12; let filtered=applyFilters(items, filters);
  function renderSlice(reset=false){
    if(reset){container.innerHTML=''; index=0;}
    const slice=filtered.slice(index,index+batch); index+=slice.length;
    slice.forEach(m=>{
      const card=document.createElement('div');
      card.className='relative group gallery-item';
      const img=document.createElement('img');
      img.src=m.url; img.alt=m.title||''; img.className='w-full h-full object-cover rounded cursor-pointer';
      const overlay=document.createElement('div');
      overlay.className='gallery-item-overlay';
      const p=document.createElement('p'); p.className='text-xs text-white truncate'; p.textContent=m.title||'';
      const time=document.createElement('p'); time.className='text-[10px] text-gray-300'; time.textContent=m.ts?new Date(m.ts).toLocaleString():'';
      overlay.appendChild(p); overlay.appendChild(time); card.appendChild(img); card.appendChild(overlay); container.appendChild(card);
    });
  }
  function fetchMore(){ if(index<filtered.length) renderSlice(); }
  function applyFilter(opts = {}, forceReload=false){
    if(typeof opts === 'string') opts = { search: opts };
    filters = { ...filters, ...opts };
    filtered = applyFilters(items, filters);
    if(forceReload){ index=0; }
    renderSlice(true);
  }
  renderSlice();
  container.addEventListener('click',e=>{
    const img=e.target.closest('.gallery-item img');
    if(img){
      const modal=document.getElementById('gallery-modal');
      const modalImg=document.getElementById('gallery-modal-img');
      if(modal&&modalImg){modalImg.src=img.src; modal.style.display='flex';}
    }
  });
  return {fetchMore, applyFilter};
}

export async function addToGallery(title, dataUrl, caption='') {
  const withText = await embedCaption(dataUrl, caption);
  const list = JSON.parse(localStorage.getItem('userGallery') || '[]');
  const tags = (caption.match(/#(\w+)/g) || []).map(t=>t.slice(1).toLowerCase());
  list.push({ title, url: withText, caption, tags, local: true, ts: Date.now() });
  localStorage.setItem('userGallery', JSON.stringify(list));
  window.dispatchEvent(new Event('gallery-updated'));
  showToast('Salvato nella galleria');
}

