# Clinical Notes Scoreboard (real unstructured EHR text)

- Corpus: MTSamples (mtsamples.com), public domain, via HF mirror rungalileo/medical_transcription_40; real de-identified transcribed medical reports.
- Notes: **4499**   |   grouped CV folds: 5   |   each note = its own CV group
- Frozen clinical transformer: `clinicalnotes:emilyalsentzer/Bio_ClinicalBERT`
- Honest relativity: the transformer is NOT assumed to beat the classical floors; numbers are measured. Label-free FM embedding computed once (no leak); heads + TF-IDF vocabulary fit on TRAIN folds only.

## Task 1 — Surgery vs rest (binary, AUROC ↑)

| config | auroc |
|---|---|
| clinicalbert+lr | 0.9103 |
| tfidf+lr | 0.8228 |
| hashing+gbm | 0.7892 |
| majority | 0.5000 |

## Task 2 — 40-way specialty (multi-class, macro-F1 ↑)

| config | macro_f1 | accuracy | macro_auroc |
|---|---|---|---|
| tfidf+lr | 0.4371 | 0.4070 | 0.9606 |
| clinicalbert+lr | 0.4048 | 0.4723 | 0.9308 |
| hashing+gbm | 0.0353 | 0.1138 | 0.5003 |
| majority | 0.0093 | 0.2218 | 0.5000 |
