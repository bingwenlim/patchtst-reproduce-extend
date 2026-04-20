# Benchmark Results

Generated: 2026-04-21 02:43:28

## traffic

| Model | Test MSE | Test MAE | Best Epoch | Time |
| --- | --- | --- | --- | --- |
| PatchTST | 0.5306 | 0.4608 | 58 | 2962.6s |
| PatchTST (ACCA linear) | 0.5301 | 0.4602 | 58 | 3051.2s |
| DLinear | 0.5806 | 0.5168 | 15 | 20.5s |
| Autoformer | 0.6851 | 0.5390 | 10 | 526.3s |

## air

| Model | Test MSE | Test MAE | Best Epoch | Time |
| --- | --- | --- | --- | --- |
| PatchTST | 0.2216 | 0.2017 | 50 | 1668.3s |
| PatchTST (ACCA linear) | 0.2214 | 0.2014 | 50 | 1731.5s |
| DLinear | 0.2781 | 0.2887 | 11 | 11.7s |
| Autoformer | 0.4036 | 0.3374 | 9 | 429.9s |

## fx

| Model | Test MSE | Test MAE | Best Epoch | Time |
| --- | --- | --- | --- | --- |
| PatchTST | 0.0889 | 0.1848 | 88 | 934.2s |
| PatchTST (ACCA linear) | 0.0891 | 0.1850 | 88 | 981.6s |
| DLinear | 0.1555 | 0.2596 | 47 | 8.4s |
| Autoformer | 0.1657 | 0.2867 | 65 | 238.8s |

