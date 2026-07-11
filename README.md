# synt-text-combinator

POC: **font-anchored DiffusionPen bigram units** for Hebrew HTR synthetic data.

Static fonts give perfect text fidelity but zero style diversity; raw DiffusionPen gives
style diversity but hallucinates text. This combines them: render a bigram with a real
font, then let the Hebrew-finetuned DiffusionPen (280 SCE writer styles) **restyle** that
image (SDEdit img2img: VAE-encode the font render → noise to an intermediate timestep →
denoise conditioned on the bigram text + a writer style). The font latent anchors WHAT is
written; the diffusion adds HOW it's written. `strength` trades fidelity (low) vs style (high).

Matan is never touched — it stays a pure held-out benchmark in TrOCR_Hebrew.

## Pipeline (POC: 100 bigrams × 5 style variants)

| step | script | output |
|---|---|---|
| 1 | `01_make_bigrams.py` | `out/bigrams.txt` — 100 realistic bigrams, frequency-weighted from `sentences_modern.txt` (in-word adjacent pairs → final forms appear in natural positions) |
| 2 | `02_render_fonts.py` | `out/font/NNN_<bigram>.png` — one clean font render per bigram (random font from the 57 curated) |
| 3 | `03_diffpen_restyle.py` | `out/diffused/NNN_<bigram>/sSSS.png` — 5 restyled variants (5 random writer styles) per bigram |
| 4 | `04_make_gallery.py` | `out/gallery.html` — grid: font render vs the 5 variants, for eyeballing |

## Depends on (read-only)

- `/mnt/ssd2/cyttic/projects/test-diff-pen` — DiffusionPen code + Hebrew checkpoints
  (`hebrew_model/`), style assets (`hebrew/`), SD-1.5 VAE. Loaded via its own
  `build_char_bank.load_model`.
- `/mnt/ssd2/cyttic/projects/fontsVisualizer/fonts` — 57 curated Hebrew fonts.
- `/mnt/ssd2/cyttic/projects/TrOCR_Hebrew/sentences_modern.txt` — bigram frequency source.
- Python: `/mnt/ssd2/cyttic/ml_env/bin/python`. GPU: local RTX 2080 (fp16 autocast).

## Run

```bash
PY=/mnt/ssd2/cyttic/ml_env/bin/python
$PY 01_make_bigrams.py
$PY 02_render_fonts.py
$PY 03_diffpen_restyle.py                 # ~100×5 diffusion batches on the 2080
$PY 04_make_gallery.py && xdg-open out/gallery.html
```

Key `03` flags: `--strength 0.6` (0=copy font, 1=ignore font), `--variants 5`, `--steps 30`,
`--limit N` (smoke test), `--seed`.

All images/outputs live under `out/` which is gitignored (heavyweight files never enter git).
