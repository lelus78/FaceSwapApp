<template>
  <div class="wizard-step">
    <h2 class="text-2xl font-bold mb-4">Step 1: Carica Immagine</h2>
    <label class="upload-box" for="subject-input">
      <img v-if="preview" :src="preview" alt="preview" class="rounded" />
      <div v-else class="text-center">Carica immagine</div>
    </label>
    <input id="subject-input" type="file" accept="image/*" class="hidden" @change="onFile" />
    <button class="btn" :disabled="!file" @click="prepare">Rimuovi Sfondo</button>
    <button class="btn" :disabled="!file" @click="$emit('skip', file)">Usa Immagine Diretta</button>
  </div>
</template>

<script>
import { useAppStore } from '../store/appStore'
export default {
  emits: ['prepared', 'skip'],
  data() {
    return { file: null, preview: null }
  },
  methods: {
    onFile(e) {
      const f = e.target.files[0]
      if (f) {
        this.file = f
        this.preview = URL.createObjectURL(f)
      }
    },
    async prepare() {
      const store = useAppStore()
      await store.prepareSubject(this.file)
      this.$emit('prepared')
    }
  }
}
</script>
