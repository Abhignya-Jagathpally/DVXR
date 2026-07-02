"""dvxr.bench — rigorous real-label evaluation harness.

The plumbing package that makes the CACMF claim testable:
  * tasks.py           real-label task adapters (no circular proxies)
  * representations.py  {raw, pca, neural, vq, fused} -> one shared head
  * baselines.py        persistence/majority, classical, best-single, SOTA encoder
  * ablation.py         true modality ablation (retrain, not zero-fill)
  * protocol.py         repeated grouped CV, bootstrap CIs, paired significance
  * scoreboard.py       the relativity table (RER%, CI, p, meets >=50%?)

Design rules (enforced, not aspirational):
  - Headline metrics use REAL external labels only; circular proxies are gated
    behind an explicit demo flag and excluded here.
  - No label fabrication in the benchmark path (no class-flipping, no subject
    synthesis) — assert_no_fabrication() guards it.
  - Non-overlapping windows for any reported metric.
  - The test split is evaluated once; all search happens on validation.
"""
