import { defineStore } from 'pinia'
import axios from 'axios'

export const useAppStore = defineStore('app', {
  state: () => ({
    subjectImage: null,
    finalImage: null,
    isLoading: false,
    progressMessage: '',
    memeText: '',
    fallbackMessage: '',
    currentStep: 1
  }),
  actions: {
    async prepareSubject(file) {
      const form = new FormData()
      form.append('subject_image', file)
      this.isLoading = true
      try {
        const res = await axios.post('/prepare_subject', form)
        this.subjectImage = URL.createObjectURL(res.data)
      } finally {
        this.isLoading = false
      }
    },
    async enhancePrompt(imageBlob, prompt) {
      const base64 = await blobToBase64(imageBlob)
      const res = await axios.post('/enhance_prompt', { image_data: base64, prompt_text: prompt })
      return res.data.enhanced_prompt
    },
    async generateCaption(imageBlob, tone) {
      const base64 = await blobToBase64(imageBlob)
      const res = await axios.post('/meme/generate_caption', { image_data: base64, tone })
      this.memeText = res.data.caption
      this.fallbackMessage = res.data.fallback_message || ''
    }
  }
})

function blobToBase64(b) {
  return new Promise(r => {
    const fr = new FileReader()
    fr.onloadend = () => r(fr.result.split(',')[1])
    fr.readAsDataURL(b)
  })
}
