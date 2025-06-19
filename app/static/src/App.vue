<template>
  <div>
    <ImageUploader v-if="step===1" @prepared="step=2" @skip="onSkip" />
    <SceneGenerator v-if="step===2" />
    <MemeEditor v-if="step===3" />
    <PreviewDisplay class="mt-4" />
  </div>
</template>

<script>
import { useAppStore } from './store/appStore'
import ImageUploader from './components/ImageUploader.vue'
import SceneGenerator from './components/SceneGenerator.vue'
import MemeEditor from './components/MemeEditor.vue'
import PreviewDisplay from './components/PreviewDisplay.vue'

export default {
  components: { ImageUploader, SceneGenerator, MemeEditor, PreviewDisplay },
  computed: {
    step: {
      get() { return useAppStore().currentStep },
      set(v){ useAppStore().currentStep = v }
    }
  },
  methods:{
    onSkip(file){
      useAppStore().subjectImage = URL.createObjectURL(file)
      this.step = 3
    }
  }
}
</script>
