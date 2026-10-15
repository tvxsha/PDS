# Latency benchmark

200 validation documents of seed 42 (sampled with random.Random(0)), 3 timed runs per document and detector, after a warm-up on other documents. Each document's figure is the median of its runs; the columns below are the median, 95th percentile and mean over documents. Model loading and fitting are not timed.

Hardware and software: platform: Windows-11-10.0.26200-SP0; cpu: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel; logical_cores: 18; threads_requested: 4; power: plugged in; python: 3.14.3; date: 2026-10-15; torch: 2.14.0; transformers: 5.16.1; sentence-transformers: 6.0.1; scikit-learn: 1.9.0; numpy: 2.5.2; torch_threads: 4.

| detector | median ms | p95 ms | mean ms |
|---|---|---|---|
| pattern | 0.12 | 0.86 | 0.25 |
| embedding | 31.14 | 102.00 | 46.23 |
| perplexity | 200.06 | 2716.01 | 796.12 |
| sentence classifier | 1.60 | 3.44 | 1.86 |
| published classifier | 344.45 | 1339.29 | 578.00 |
| ours3 | 227.59 | 2545.59 | 818.54 |
| ours3 + sentence | 231.20 | 2614.59 | 817.58 |

## By document length (median ms / p95 ms, number of documents)

| detector | under 500 characters | 500 to 2,000 characters | over 2,000 characters |
|---|---|---|---|
| pattern | 0.06 / 0.16 (n=122) | 0.35 / 1.03 (n=63) | 0.84 / 1.80 (n=15) |
| embedding | 24.08 / 37.49 (n=122) | 72.72 / 115.59 (n=63) | 86.96 / 111.21 (n=15) |
| perplexity | 157.14 / 266.99 (n=122) | 1532.73 / 2949.49 (n=63) | 2561.03 / 3274.98 (n=15) |
| sentence classifier | 1.41 / 2.65 (n=122) | 1.86 / 3.80 (n=63) | 2.48 / 4.65 (n=15) |
| published classifier | 287.66 / 402.84 (n=122) | 716.75 / 1722.80 (n=63) | 1223.72 / 4321.27 (n=15) |
| ours3 | 179.79 / 317.81 (n=122) | 1580.10 / 2872.05 (n=63) | 2429.87 / 3277.56 (n=15) |
| ours3 + sentence | 173.28 / 332.79 (n=122) | 1577.95 / 2939.69 (n=63) | 2421.84 / 2945.77 (n=15) |

Use the median and the p95 in the paper, not the mean: the mean is pulled up by long documents. These numbers are valid for this machine and these settings only.
