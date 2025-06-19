<template>
  <div class="wizard-step">
    <h2 class="text-2xl font-bold mb-4">Step 2: Crea Scena</h2>
    <input v-model="prompt" class="input" placeholder="Descrivi lo sfondo" />
    <label class="inline-flex items-center mt-2">
      <input type="checkbox" v-model="auto" class="mr-2" />Auto Enh.
    </label>
    <button class="btn" @click="generate">Genera Sfondo</button>
  </div>
</template>

<script>
import { useAppStore } from '../store/appStore'
export default {
  data() {
    return { prompt: '', auto: true }
  },
  methods: {
    async generate() {
      const store = useAppStore()
      let final = this.prompt
      if (this.auto && store.subjectImage) {
        const blob = await fetch(store.subjectImage).then(r => r.blob())
        final = await store.enhancePrompt(blob, this.prompt)
      }
      // Placeholder for API call to create scene
      store.progressMessage = `Scena generata con: ${final}`
      store.currentStep = 3
    }
  }
}
</script>
