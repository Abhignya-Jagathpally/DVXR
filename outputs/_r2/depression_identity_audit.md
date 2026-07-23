# Depression identity-leakage audit (Mumtaz, band-power features)

- subjects: **58** | windows: 812 | features: 228
- diagnosis is subject-level (each subject one label): **True**

| test | metric | value | reference |
|---|---|---:|---|
| A) Depression (subject-held-out) | AUROC | 0.884 | vs LaBraM headline 0.961 (band-power is weaker) |
| B) Subject-identity decodability | accuracy | 0.888 | chance = 1/58 = 0.017 (52x chance) |

**Verdict:** CONFOUNDED: subject identity is highly decodable AND each subject carries one diagnosis — subject-held-out CV cannot separate a depression biomarker from subject identity. The 0.961 headline is an upper bound; a within-subject-label-variation cohort (or an identity-adversarial control) is required to validate it.

This audits the *protocol confound*, not the model: any resting-state depression result on a between-subject cohort inherits it. The recommended fix is a cohort with within-subject label variation (e.g. pre/post-treatment) or an identity-adversarial training control.
