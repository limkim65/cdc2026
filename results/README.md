# Results Directory Layout

This folder is organized by experiment/output type.

- `realtime_verify/`
  - Outputs from `experiments/realtime_verify_run.py`
  - Files: `realtime_verify_*.npz`, `realtime_verify_*.png`

- `realtime_verify_onlinebuffer/`
  - Outputs from `experiments/realtime_verify_onlinebuffer_run.py`
  - Files: `realtime_verify_onlinebuffer_*.npz`, `realtime_verify_onlinebuffer_*.png`
  - `debug/`: interactive debug snapshots/final debug plots

- `verify_measurement_noise/`
  - Outputs from `experiments/verify_measurement_noise.py`
  - Files: `verify_measurement_noise*.png`, `diag_k40_vs_k50*.txt`

- `k_compare_debug/`
  - Outputs from `experiments/k_compare_debug.py` and related K-comparison diagnostics
  - Files: `k_compare_debug*`, `k40_vs_k50*`, `noise_sweep_K.png`

- `cache/`
  - Reusable cached artifacts
  - Files: `offline_dataset_cache.pkl`

- `misc/`
  - Reserved for uncategorized artifacts.

## Naming convention

Most run artifacts include parameters and timestamp in filename:
`..._k{K}_sigma{sigma}_seed{seed}_{YYYYMMDD_HHMMSS}`

## Recommended usage

Use per-experiment output roots to avoid mixing:

```bash
python3 cdc2026/experiments/realtime_verify_onlinebuffer_run.py \
  --results-dir cdc2026/results/realtime_verify_onlinebuffer
```
