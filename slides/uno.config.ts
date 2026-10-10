import { defineConfig } from 'unocss'

// Slidev ships UnoCSS with the Tailwind-compatible Wind preset,
// so every utility below is standard Tailwind syntax.
export default defineConfig({
  theme: {
    colors: {
      paper: '#F6F4EF',
      ink: '#151515',
      muted: '#6E6A63',
      line: '#D9D5CC',
      accent: '#E2552B',
    },
  },
  shortcuts: {
    eyebrow: 'font-mono text-[13px] tracking-[0.18em] uppercase text-muted',
    headline: 'text-[56px] leading-[1.02] font-bold tracking-[-0.03em] text-ink',
    card: 'rounded-2xl border border-ink/80 bg-paper',
  },
})
