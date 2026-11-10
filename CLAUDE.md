# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A beginner-friendly slide deck on machine learning fundamentals. It covers ML basics, regression, loss functions, classification, evaluation metrics, neural networks, backprop and optimizers. The deck is one file: `presentation/ml_fundamentals.html`. `examples/` holds one NumPy script per part (`01_…py` to `08_…py`) that reproduces the slides' worked numbers. `main.py` is an unused placeholder from `uv init`.

## Commands

There is no build, lint or test step.

- Run an example: `uv run python examples/03_loss_functions.py`. Run all of them with `for f in examples/*.py; do uv run python "$f"; done`. They use Python 3.13 managed by uv, and NumPy is the only dependency.
- View: `open presentation/ml_fundamentals.html`. This needs internet access because reveal.js 5.1.0 and KaTeX load from the jsDelivr CDN.
- Export to PDF: open `presentation/ml_fundamentals.html?print-pdf` in Chrome, then choose Print → Save as PDF with Landscape layout and no margins.
- In the browser: `S` opens speaker notes, `Esc` opens the slide overview, `F` goes fullscreen. `hash: true` is set, so the URL tracks the current slide.

## Deck structure and conventions

- The deck is a flat list of `<section>` elements inside `.reveal > .slides`. Each of the 8 parts opens with a `<section class="section-title">` slide. That slide has its own `data-background-gradient`, a `<div class="section-num">NN</div>` and an `<!-- ===== NAME -->` comment banner. If you add, remove or reorder a part, also update the agenda grid (`.agenda`) on slide 2 and the README topic list.
- Every slide ends with `<aside class="notes">`, which holds a full speaker script (several `<p>` paragraphs, sometimes with `[bracketed]` stage directions). New slides should have notes in the same style.
- The colour code is semantic and used throughout, in CSS variables, utility classes (`.blue`, `.bg-orange`, …) and inline SVG/KaTeX colours. Blue `#2563eb` means inputs/data. Orange `#f59e0b` means weights/parameters. Green `#16a34a` means outputs/correct. Red `#dc2626` means error/loss. Purple `#7c3aed` means hidden layers. In KaTeX, write colours as `{\color{#2563eb}x}`.
- Callout boxes: `.plain` adds an "In plain English: " prefix automatically through CSS `::before`, so don't type that text yourself. `.note` is for info and `.warn` is for caveats. Layout uses `.cols`, with the variants `c3`, `c4`, `wide-left`, `wide-right` and `top`, plus `.card`.
- Math uses KaTeX: `$…$` for inline and `$$…$$` for display, usually wrapped in `<div class="eq">`. KaTeX ignores text inside `<svg>`, so SVG labels must be plain text.
- Diagrams are hand-written inline `<svg>` drawings, not images. Progressive reveals use `class="fragment"`, with `data-fragment-index` to group elements.
- Reveal config (bottom of the file): the canvas is 1280×760 and slides are centered. Keep content within that size; the `.small`/`.tiny` classes are how dense slides are made to fit.
- Examples use an Indian banking context (₹ lakh, card fraud, loan default). Worked numeric examples (e.g. backprop, parameter counting) are cross-referenced in later slides and notes, so if you change their numbers, update every slide and note that refers to them, plus the matching script in `examples/`.

## Examples conventions

- Scripts are standalone and aimed at beginners. Each defines its own small helpers (e.g. `sigmoid`) rather than sharing a module, uses NumPy only (no scikit-learn or plotting), and shows results with plain `print` output.
- Each script first reproduces the slide values exactly, then adds one small "learn it from data" step. Any randomness uses a fixed seed (`np.random.default_rng(n)`) so output is deterministic.
