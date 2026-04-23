# Benchmark Results

Generated: 2026-04-23 01:21:20

## traffic

| Model | Test MSE | Test MAE | Best Epoch | Train Time | Inference Time |
| --- | --- | --- | --- | --- | --- |
| PatchTST | 0.5306 | 0.4608 | 58 | 2846.3s | 2.307s |
| PatchTST (ACCA) | 0.5307 | 0.4609 | 52 | 2842.0s | 2.752s |
| DLinear | 0.5806 | 0.5168 | 15 | 24.6s | 0.131s |
| Autoformer | 0.6851 | 0.5390 | 10 | 522.5s | 0.809s |

## air

| Model | Test MSE | Test MAE | Best Epoch | Train Time | Inference Time |
| --- | --- | --- | --- | --- | --- |
| PatchTST | 0.2216 | 0.2017 | 50 | 1828.0s | 1.602s |
| PatchTST (ACCA) | 0.2212 | 0.2010 | 47 | 1794.4s | 1.907s |
| DLinear | 0.2781 | 0.2887 | 11 | 13.7s | 0.115s |
| Autoformer | 0.4036 | 0.3374 | 9 | 428.8s | 0.955s |

## fx

| Model | Test MSE | Test MAE | Best Epoch | Train Time | Inference Time |
| --- | --- | --- | --- | --- | --- |
| PatchTST | 0.0889 | 0.1848 | 88 | 1047.0s | 0.549s |
| PatchTST (ACCA) | 0.0890 | 0.1849 | 89 | 1044.8s | 0.542s |
| DLinear | 0.1555 | 0.2596 | 47 | 8.1s | 0.025s |
| Autoformer | 0.1657 | 0.2867 | 65 | 238.0s | 0.104s |

