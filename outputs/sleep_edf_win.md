# Sleep-EDF raw-signal win benchmark

Proposed = raw-signal multimodal 1D-CNN (reads raw EEG/EOG/EMG/resp windows). Floor = tuned GBM on the SAME windows' summary-stat features. Held-out-subject CV. 1-AUROC (lower better). A **win** = RER>0 with bootstrap-CI lower bound >0.

Recordings used: 5 (5 subjects).

| target | N | pos% | rawCNN | floor | RER% | 95% CI | win |
|---|---|---|---|---|---|---|---|
| wake_sleep | 2000 | 31.2 | 0.0170 | 0.0018 | -824.2 | -1449.6..77.4 | — |
| rem | 2000 | 6.2 | 0.1549 | 0.0639 | -142.4 | -158.6..-25.1 | — |
| deep | 2000 | 6.1 | 0.0367 | 0.0196 | -87.7 | -99.9..0.3 | — |

**No CI-backed win yet** at this recording count — report honestly; add recordings / try harder targets.
