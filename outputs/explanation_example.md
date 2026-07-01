# CACMF Prediction Explanation

One prediction explained across four auditable views. Attention/codes are associations, not causal claims.


## 1. Physiological biomarkers

```
subject_id session_id  hrv_sdnn  hrv_rmssd  eda_tonic_mean  eda_scr_rate  resp_rate_bpm  glucose_cv  glucose_tir_70_180  eeg_beta_alpha_ratio
      demo         s1       NaN        NaN        2.205198          27.0            NaN      0.1691                 0.7                   NaN
```


## 2. Neural saliency (top features)

```
feature  saliency          method
     f4  0.058155 neural_saliency
     f0  0.056889 neural_saliency
     f5  0.056626 neural_saliency
     f3  0.056603 neural_saliency
     f8  0.056353 neural_saliency
     f9  0.056240 neural_saliency
     f7  0.055460 neural_saliency
     f2  0.055087 neural_saliency
     f1  0.052800 neural_saliency
     f6  0.051518 neural_saliency
```


## 3. Modality attention / fusion weights

```
 sample modality  attention  weight
      0      eeg   0.258649     NaN
      1      eeg   0.269853     NaN
      2      eeg   0.269786     NaN
      3      eeg   0.265584     NaN
      4      eeg   0.259918     NaN
      5      eeg   0.262844     NaN
      6      eeg   0.256544     NaN
      7      eeg   0.283794     NaN
      8      eeg   0.270094     NaN
      9      eeg   0.240595     NaN
     10      eeg   0.268685     NaN
     11      eeg   0.274549     NaN
```


## 4. Active codebook entries

```
modality  code_index  count
     eeg           0      9
     eeg           1      9
     eeg           2      1
     eeg           3      1
     eeg           4      7
     eeg           5      1
     eeg           6      3
     eeg           7      2
     eeg           8     14
     eeg           9      4
     eeg          10      1
     eeg          11     11
```


Codebook perplexity: eeg=23.18, wearable_phys=21.63, cgm=24.76
