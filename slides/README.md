# One Request, Three Hops

A Slidev deck showing how an app request moves through: **App → API Gateway → Application Backend → Database**.

```bash
npm install
npm run dev      # http://localhost:3030 — ←/→ or space to step, F for fullscreen, O for overview
npm run export   # dist/api-architecture.pdf, one page per reveal step
npm run export:static   # one page per slide (final state)
npm run build    # static HTML site in dist/ — open dist/index.html via any static server
npm run export:public   # refresh public/api-architecture.pdf (the in-deck download button)
```

The diagrams are Vue SVG components in `components/`. Each one takes `step` (the slide's `$clicks`) and reveals parts as the step increases.
Styling uses Slidev's UnoCSS (Tailwind-compatible utilities); theme tokens are in `uno.config.ts` and `styles/index.css`.
