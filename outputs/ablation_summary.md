# CACMF Ablation Summary

Fused vs single-modality vs aggregation on subject-held-out splits. No configuration is assumed to win — these are measured results.


## glucose

```
config_type   config_name       mae  coverage  interval_radius
     single           eeg 25.094353     0.900        53.573929
     single wearable_phys 25.607981     0.975        50.712639
     single           cgm  3.304727     0.825         6.040771
     fusion         early 22.993738     0.875        46.024658
     fusion  intermediate 22.335249     0.900        48.159920
     fusion late_weighted 21.959959     0.925        52.701584
     fusion     attention 26.232861     0.825        49.131210
     fusion   cross_modal 23.830573     0.900        49.428360
aggregation  ensemble_avg 17.380217     0.925        34.298492
```

_(primary metric: mae, lower is better)_


## stress_detection

```
config_type         config_name    auroc       f1  accuracy      ece
     single                 eeg 0.605333 0.540541     0.575 0.156316
     single       wearable_phys 0.912000 0.764706     0.800 0.162686
     single                 cgm 0.458667 0.432432     0.475 0.235077
     fusion               early 0.536000 0.512821     0.525 0.135734
     fusion        intermediate 0.536000 0.545455     0.500 0.137095
     fusion       late_weighted 0.736000 0.648649     0.675 0.134833
     fusion           attention 0.562667 0.512821     0.525 0.122920
     fusion         cross_modal 0.610667 0.545455     0.375 0.135161
aggregation       weighted_late 0.877333 0.833333     0.850 0.267223
aggregation        ensemble_avg 0.877333 0.833333     0.850 0.267223
aggregation confidence_weighted 0.869333 0.742857     0.775 0.158304
```

_(primary metric: auroc, higher is better)_
