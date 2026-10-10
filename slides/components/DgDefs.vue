<script setup lang="ts">
// Arrowhead markers. `id` keeps markers unique per diagram.
defineProps<{ id: string }>()
</script>

<template>
  <defs>
    <marker :id="`${id}-ink`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--ink)" />
    </marker>
    <marker :id="`${id}-accent`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--accent)" />
    </marker>
    <marker :id="`${id}-muted`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--muted)" />
    </marker>
  </defs>
</template>
