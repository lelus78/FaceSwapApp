<template>
  <div>
    <div v-if="msg" class="bg-red-800 text-white p-2 mb-2">{{msg}}</div>
    <img v-if="image" :src="image" class="rounded-lg w-full" />
  </div>
</template>

<script>
import { useAppStore } from '../store/appStore'
export default {
  computed: {
    image() { return useAppStore().subjectImage },
    msg() { return useAppStore().fallbackMessage }
  }
}
</script>
