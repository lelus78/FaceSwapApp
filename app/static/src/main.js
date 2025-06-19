import { createApp } from 'vue'
import { initTheme } from "../js/theme.js"
import { initSidebarToggle } from "../js/gallery.js"
import { createPinia } from 'pinia'
import App from './App.vue'

const app = createApp(App)
app.use(createPinia())
app.mount('#app')
initTheme("theme-toggle","theme-icon");
initSidebarToggle(document.getElementById("sidebar"), document.getElementById("sidebar-toggle"));
