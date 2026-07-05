# workload-characterization

Profiling utilities for characterizing model inference workloads on-device
(currently ResNet18 on Apple Silicon MPS).

## Layout

- `scripts/profile_resnet18.py` — runs a batch-size latency/throughput sweep
  and/or a per-op CPU profile of ResNet18, writing results as JSON.
- `results/` — JSON outputs and Chrome traces from the last run.

## Usage

```bash
python3 scripts/profile_resnet18.py --mode both
```

- `--mode sweep` — only run the batch-size sweep, writes `results/batch_sweep_resnet18.json`
- `--mode ops` — only run the per-op profile, writes `results/op_profile_resnet18.json`
- `--mode both` (default) — run both
- `--batch-sizes` — override the sweep's batch sizes (default `1 2 4 8 16 32 64 128`)
- `--output-dir` — override where JSON/trace files are written (default `results/`)

The op profile also exports a Chrome trace to `results/batch_sweep_resnet18_trace.json`,
viewable at `chrome://tracing` or https://ui.perfetto.dev.

## Output schema

**`batch_sweep_resnet18.json`**

```json
{
  "device": "mps",
  "n_warmup": 5,
  "n_runs": 50,
  "results": [
    {"batch_size": 1, "total_ms": ..., "per_image_ms": ..., "throughput_img_per_s": ...},
    ...
  ]
}
```

**`op_profile_resnet18.json`**

```json
{
  "device": "mps",
  "batch_size": 1,
  "ops": [
    {
      "name": "aten::native_batch_norm",
      "count": 5,
      "cpu_time_total_us": ...,
      "self_cpu_time_total_us": ...,
      "cpu_time_avg_us": ...,
      "input_shapes": [...],
      "cpu_memory_usage_bytes": ...,
      "self_cpu_memory_usage_bytes": ...
    },
    ...
  ]
}
```

`ops` is sorted descending by `cpu_time_total_us`, grouped by (op name, input shape).
Since profiling only captures `ProfilerActivity.CPU`, these times reflect host-side
dispatch/launch cost for MPS ops, not raw GPU kernel execution time.

## Findings (ResNet18, batch=1..128, Apple Silicon MPS)

The two outputs tell complementary halves of the same story:

- **`op_profile_resnet18.json` (batch=1):** the forward pass's total self-CPU
  time (~1.37ms, dominated by `aten::native_batch_norm` and
  `aten::_mps_convolution` dispatch) is on the same order as the *entire*
  measured batch=1 wall-clock latency (~1.41ms). At batch=1 there isn't enough
  work per op to amortize the cost of dispatching each op to the GPU, so the
  op-level profile shows the model is **CPU-overhead gated**, not GPU-compute
  gated.

- **`batch_sweep_resnet18.json`:** throughput rises steeply and per-image
  latency drops as batch size grows from 1 → 16 (708 → 1430 img/s), because
  the fixed per-op dispatch overhead is amortized over more images per
  launch. Past batch≈16, throughput flattens (~1430-1436 img/s from batch 16
  through 128) and per-image latency stops improving — the bottleneck has
  shifted from CPU dispatch overhead to the GPU's actual compute/memory
  throughput, so adding more images per batch no longer buys anything.

Put together: the op profile explains *why* small batches are slow (dispatch
overhead dominates), and the sweep shows *where* that overhead stops
mattering and the GPU's raw compute/memory bandwidth takes over as the
limiting factor.
