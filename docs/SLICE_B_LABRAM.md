# Slice B — real LaBraM EEG foundation model (design + exact spec)

Goal: replace the band-power+VQ EEG baseline with a **real pretrained EEG foundation model**
(LaBraM) as a frozen feature extractor over raw EEG windows, and honestly benchmark whether it
beats the baseline on the real EEG cohorts (`eegmat_workload`, `deap_anxiety`, `deap_arousal`,
`mumtaz_depression`). This is the EEG FM the POW names — blocked before by braindecode/torchaudio,
now unblocked via a direct safetensors load + a **vendored forward** (no braindecode import).

## Why not just `import braindecode`

`braindecode` will not import under the pinned `torch==2.12` (no compatible `torchaudio`; see
[[labram-braindecode-torch-blocker]] in memory). But the **weights are plain safetensors** and the
architecture is a standard ViT-style encoder we can reimplement and load by key name.

## Weights (verified loadable here)

- Repo: **`braindecode/labram-pretrained`** (HF). Files: `model.safetensors` (23 MB), `config.json`.
- Load: `safetensors.torch.load_file(hf_hub_download("braindecode/labram-pretrained","model.safetensors"))`
  → 12-block state dict, **no braindecode needed**. The box has outbound internet from the venv
  (`urllib`/`huggingface_hub` both work). Cache once; the default path must stay offline-capable, so
  gate the FM behind a capability check like the other SOTA encoders (fall back to band-power+VQ).

## Exact architecture (reverse-engineered from the state dict + BSD-3 reference)

Reference source (BSD-3, re-downloadable):
`https://raw.githubusercontent.com/braindecode/braindecode/master/braindecode/models/labram.py`
(also has `LABRAM_CHANNEL_ORDER`, `_LABRAM_CANONICAL_INDEX`, `InterpolatedLaBraM`).

- **embed_dim = 200**, **12 blocks**, **10 heads** (head_dim 20; per-head QK LayerNorm(20)),
  **MLP hidden 800** (ratio 4), GELU, LayerScale (`gamma_1`, `gamma_2`), final `norm` (eps 1e-6).
- **Patch embedding** = `patch_embed[0]` `_SegmentPatch` then `_TemporalConv`:
  - `_SegmentPatch`: Conv1d(1→emb, kernel=patch_size=200, stride=200) segments `(B, n_chans, n_times)`
    → `(B, n_chans, n_patchs=n_times//200, patch_size=200)`.
  - `_TemporalConv`: `Rearrange("B chs npat spatch -> B () (chs npat) spatch")`, then
    conv1 `Conv2d(1→8,(1,15),stride=(1,8),pad=(0,7))` → GELU → `GroupNorm(4,8)`;
    conv2 `Conv2d(8→8,(1,3),pad=(0,1))` → GELU → GroupNorm; conv3 same;
    `Rearrange("B C NA T -> B NA (T C)")` → `(B, chs*npat, 8*25=200)`.
- **Tokens (channel-major!):** `x = cat([cls_token, patch_tokens], dim=1)`. Order is
  `[CLS, ch0·p0, ch0·p1, …, ch1·p0, …]`.
- **Position embedding** `(1,129,200)` = CLS + 128 canonical channels. Tokenizer mode:
  `pos_used = position_embedding[:, input_chans]` where `input_chans = [0] + [i+1 for i in matched]`
  (`matched` = canonical index of each input channel name, **CLS at 0, channels +1**). Then
  `_adj_position_embedding` expands each channel's pos across `n_patchs` and re-prepends CLS pos:
  ```
  pos = pos_used[:,1:,:].unsqueeze(2).expand(B,-1,n_patchs,-1).flatten(1,2)
  pos = cat([pos_used[:,0:1,:].expand(B,-1,-1), pos], dim=1);  x += pos
  ```
- **Temporal embedding** `(1,16,200)`: `time = temporal_embedding[:, 1:n_patchs+1, :]` (index 0
  reserved), tiled across channels to match the channel-major layout, added to `x[:, 1:, :]`.
- **Attention.forward:** `qkv = linear(x, qkv.weight, bias=None)` (this ckpt has **no** qkv.bias /
  q_bias / v_bias) → reshape `(B,N,3,heads,head_dim).permute(2,0,3,1,4)`; `q=q_norm(q)`, `k=k_norm(k)`
  (per-head LN); `q*=scale (head_dim**-0.5)`; `attn=softmax(q@k^T)`; `x=(attn@v)…reshape; x=proj(x)`.
- **Block:** `x = x + gamma_1 * attn(norm1(x)); x = x + gamma_2 * mlp(norm2(x))`.
- **Frozen representation:** `forward(..., return_features=True)` →
  `{"features": x[:,1:,:], "cls_token": x[:,0,:]}`. Use the normed **cls_token** (200-d) as the
  frozen embedding (default `use_mean_pooling=False`), or mean of features — pick one, document it.

## Input adaptation (per cohort)

Raw EEG windows live in `task.extra["raw"]["eeg"]` (DEAP 32-ch, eegmat 19-ch; per
`raw_seq._raw_channels`). Steps:
1. Reshape flat `(N, C*L)` → `(N, C, L)`; **resample L to 200 Hz** (LaBraM's rate). DEAP canonical
   events are decimated (~8 Hz effective — see finding in BENCHMARK_FINDINGS "Slice H"), so DEAP may
   not benefit; eegmat (64 Hz) and mumtaz (64 Hz) are the fair tests. Report per-cohort honestly.
2. Segment L into `n_patchs = L//200` patches of 200 samples (drop remainder). If `L < 200` after
   resample, this cohort can't feed LaBraM — say so, don't fake.
3. Map channel names → canonical indices via `_LABRAM_CANONICAL_INDEX` (uppercase). Drop channels
   not in `LABRAM_CHANNEL_ORDER`; pass `ch_names` so `input_chans` is correct.
4. Per-channel standardize (LaBraM was trained on ~unit-variance µV/100 scaling — check the ref's
   expected scaling; the paper uses /100). Forward frozen → cls embedding `(N, 200)`.

## Wiring + honest evaluation

- New module `src/dvxr/encoders/labram_real.py`: `LaBraMEncoder` (vendored forward + `from_pretrained`)
  + a `labram_embeddings(task)` that caches `(N,200)` per task (like `_sota_embeddings`).
- New bench config `sota:eeg_fm` (or `labram`) in `baselines.py`, guarded (absent → skip, offline
  suite still passes). Competes on the SAME folds vs `single:eeg` (band-power), `raw_cnn`, and the
  existing `sota` (MOMENT). Also add it as a candidate in the DNH library so DNH can use it.
- **Report honestly** whether real LaBraM beats the band-power+VQ baseline per cohort. A negative
  result (FM ties/loses on summary-vs-raw at these sampling rates) is a valid, reportable outcome.

## Correctness validation (do NOT skip — a wrong forward = fake "real LaBraM")

The token layout / embedding indexing is easy to get subtly wrong. Before trusting any number:
1. **Shape asserts** at every stage vs the spec above.
2. **Key-coverage assert:** every safetensors key is consumed by the module (no missing/unexpected),
   `load_state_dict(strict=True)`.
3. **Non-degeneracy:** embeddings vary across distinct inputs (std ≫ 0), finite.
4. **Reference cross-check (strongest):** the box has internet — in a scratch venv, `pip install
   braindecode` into an ISOLATED env with a compatible torch, run the real `Labram.forward` on a
   fixed random input, and assert our vendored forward matches within 1e-4. If that env can't be
   built, at minimum do 1–3 and label the FM "vendored-forward (unverified against reference)".

## Status

Weights confirmed loadable; full architecture + reference in hand. Implementation is the next
iteration's work (tasks #5, #6). This note is the exact spec to build against.
