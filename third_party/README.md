# third_party/

Vendored external code kept as **reference material**, not wired into the DVXR pipeline.

## autoresearch/ — karpathy/autoresearch

Source: https://github.com/karpathy/autoresearch (vendored as a shallow copy; upstream `.git` stripped).

**What it is:** an autonomous ML-research harness. An agent edits `train.py` to iterate on training a
small GPT within a fixed ~5-minute budget per experiment, judged by validation loss, driven by
high-level instructions in `program.md`.

**Why it's here (and why it is NOT a skill):** unlike `addyosmani/agent-skills` and
`imbad0202/academic-research-skills` — which are Claude skill collections and were installed under
`.claude/skills/` — autoresearch is a GPU-only training loop (requires a single NVIDIA GPU, Python 3.10+,
and the `uv` package manager). It cannot run in this repo's offline/CPU/deterministic environment. It is
kept here as the reference pattern for an **autonomous overnight-experiment loop** to inform the POW
Goal-1 fine-tuning stage (self-supervised training of the biosignal/EEG encoders). Adapt the
`program.md` + edit-train-evaluate-keep/discard loop; do not import its code.

Upstream ships no explicit LICENSE file; retained here for research reference only.
