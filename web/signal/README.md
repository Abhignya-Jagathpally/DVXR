# SIGNAL — Adaptive Multimodal Health Intelligence

The React implementation of the SIGNAL research-pitch site. Companion to the
self-contained artifact at `outputs/product/signal_site.html`; both share the same
copy, design tokens, and **verified** evidence numbers (see `src/content.js`).

## Run

```bash
cd web/signal
npm install
npm run dev      # local dev server
npm run build    # production build -> dist/
npm run preview  # serve the production build
```

## Structure

- `src/content.js` — single source of truth for all copy and verified figures.
- `src/lib/signals.js` — canvas draw functions (hero silhouette, CGM forecast, mini-waves…).
- `src/lib/useReveal.js` — IntersectionObserver scroll-reveal + reduced-motion hook.
- `src/components/*` — one component per section: `HeroSection`, `ProblemSection`,
  `SignalWorld`, `IntelligencePipeline`, `InteractiveExperience`, `FeatureGrid`,
  `UseCases`, `EvidenceSection`, `ResearchRoadmap`, `ResearcherProfile`,
  `ResearchDisclaimer`, plus `Nav` and `SignalCanvas`.

## Honesty

Every quantitative claim resolves to a committed scoreboard (`docs/MODEL_CARD.md`,
`outputs/*scoreboard*`, `BENCHMARK_FINDINGS.md`). Demonstrations use synthetic/sample
data and are labeled as such. The learned-fusion honest-negative result is presented, not
hidden. Research-grade decision support — not a diagnosis.
