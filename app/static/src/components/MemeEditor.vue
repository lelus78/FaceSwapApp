<template>
  <div class="wizard-step">
    <h2 class="text-2xl font-bold mb-4">Meme Text</h2>
    <div class="mb-2">
      <button v-for="t in tones" :key="t" class="btn mx-1" :class="{active: tone===t}" @click="tone=t">{{t}}</button>
    </div>
    <div class="flex items-center gap-2">
      <input v-model="text" class="input flex-1" placeholder="Testo meme" />
      <button class="btn" @click="suggest">✨</button>
    </div>
  </div>
</template>

<script>
import { useAppStore } from '../store/appStore'
export default {
  data() {
    return { text: '', tone: 'scherzoso', tones: ['scherzoso','sarcastico','epico','assurdo'] }
  },
  methods: {
    async suggest() {
      const store = useAppStore()
      if (store.subjectImage) {
        const blob = await fetch(store.subjectImage).then(r => r.blob())
        await store.generateCaption(blob, this.tone)
        this.text = store.memeText
      }
    }
  }
}
</script>
