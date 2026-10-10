import { defineConfig } from 'vite'

// Slidev 53's built-in code-block CSS trips Vite 8's lightningcss minifier; skip CSS minification.
export default defineConfig({ build: { cssMinify: false } })
