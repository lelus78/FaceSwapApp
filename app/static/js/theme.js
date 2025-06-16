export function initTheme(toggleSelector, iconSelector) {
  const toggle = typeof toggleSelector === 'string' ? document.getElementById(toggleSelector) : toggleSelector;
  const icon = typeof iconSelector === 'string' ? document.getElementById(iconSelector) : iconSelector;

  function updateIcon() {
    const dark = document.documentElement.classList.contains('dark');
    icon.innerHTML = dark
      ? '<path d="M21.752 15.002A9.718 9.718 0 0112.75 22 9.75 9.75 0 013 12.25c0-3.902 2.338-7.25 5.748-8.748a.75.75 0 01.912 1.1 7.5 7.5 0 009.038 9.038.75.75 0 01.054 1.362z"/>'
      : '<path d="M12 2.25v1.5M12 20.25v1.5M4.219 4.219l1.06 1.06m13.442 13.442l1.06 1.06M2.25 12h1.5m16.5 0h1.5M6.28 17.72l1.06-1.06m9.32-9.32l1.06-1.06M8.625 12a3.375 3.375 0 106.75 0 3.375 3.375 0 00-6.75 0z"/>';
  }

  function applySaved() {
    const saved = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const useDark = saved ? saved === 'dark' : prefersDark;
    document.documentElement.classList.toggle('dark', useDark);
    updateIcon();
  }

  toggle.addEventListener('click', () => {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    updateIcon();
  });

  applySaved();
}
