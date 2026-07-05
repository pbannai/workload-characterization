# workload-characterization

Profiling utilities for characterizing model inference workloads on-device
(currently ResNet18 on Apple Silicon MPS, with an fp32/int8 precision comparison).

## Layout

- `scripts/profile_resnet18.py` — runs a batch-size latency/throughput sweep
  and/or a per-op CPU profile of ResNet18, in fp32 or statically-quantized
  INT8, writing results as JSON.
- `results/` — JSON outputs and Chrome traces from the last run of each variant.

## Usage

```bash
python3 scripts/profile_resnet18.py --mode both
```

- `--mode sweep` — only run the batch-size sweep, writes `results/batch_sweep_resnet18*.json`
- `--mode ops` — only run the per-op profile, writes `results/op_profile_resnet18*.json`
- `--mode both` (default) — run both
- `--precision fp32|int8` (default `fp32`) — `int8` statically quantizes the
  model (qnnpack) and always runs on CPU, since MPS has no quantized-op
  support. Adds an `_int8` suffix to output filenames.
- `--device auto|cpu|mps` (default `auto`) — device for **fp32** runs only
  (`auto` prefers MPS). Forcing `--device cpu` adds a `_cpu` suffix, useful
  for an apples-to-apples fp32-CPU vs. int8-CPU comparison.
- `--batch-sizes` — override the sweep's batch sizes (default `1 2 4 8 16 32 64 128`)
- `--output-dir` — override where JSON/trace files are written (default `results/`)

The op profile also exports a Chrome trace to `results/batch_sweep_resnet18*_trace.json`,
viewable at `chrome://tracing` or https://ui.perfetto.dev.

## Output schema

**`batch_sweep_resnet18*.json`**

```json
{
  "device": "mps",
  "precision": "fp32",
  "n_warmup": 5,
  "n_runs": 50,
  "results": [
    {"batch_size": 1, "total_ms": ..., "per_image_ms": ..., "throughput_img_per_s": ...},
    ...
  ]
}
```

**`op_profile_resnet18*.json`**

```json
{
  "device": "mps",
  "precision": "fp32",
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
Since profiling only captures `ProfilerActivity.CPU`, times for MPS/CPU fp32 runs
reflect host-side dispatch/launch cost rather than raw GPU kernel execution time;
for the CPU int8 run they reflect actual quantized-kernel execution time, since
there's no separate device to dispatch to.

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

## INT8 quantization: fp32-CPU vs. int8-CPU (batch=1..128)

Because MPS has no quantized-op support, INT8 must run on CPU. Comparing
INT8 against the MPS fp32 numbers above would confound "quantization" with
"CPU vs. GPU," so the fair comparison is fp32-CPU (`--device cpu`) vs.
int8-CPU — both single-device, CPU-only runs:

| Batch | fp32-CPU (img/s) | int8-CPU (img/s) |
|------:|-----------------:|------------------:|
|     1 |             165  |               142 |
|     8 |             212  |               163 |
|    16 |              82  |               174 |
|    32 |              98  |               182 |
|   128 |             105  |               172 |

Two things stand out:

- **At small batches, fp32-CPU is actually faster than int8-CPU.**
  Quantization adds `quantize`/`dequantize` overhead around each op, and at
  batch=1 there isn't enough compute per op to pay that cost back.

- **fp32-CPU falls off a cliff past batch≈8, while int8-CPU stays flat.**
  The per-op profile explains why: fp32-CPU's convolutions run through
  `aten::_slow_conv2d_forward` (~70% of self-CPU time) — this Mac's PyTorch
  build has no MKLDNN/oneDNN backend for CPU fp32 conv, so it falls back to
  a naive, non-vectorized reference kernel that scales poorly with batch
  size. The quantized model instead uses `quantized::conv2d` /
  `quantized::conv2d_relu`, qnnpack's actual vectorized (ARM NEON) INT8
  kernels, whose cost scales far more gracefully with batch size.

So the INT8 win here isn't purely "smaller numbers compute faster" — a good
chunk of it is that quantization routes execution through a properly
optimized kernel path (qnnpack) that fp32 CPU inference doesn't get on this
platform. On hardware where fp32 CPU conv *does* hit an optimized backend
(e.g. x86 with MKLDNN, or just using MPS as above), the quantization delta
would look different. (Single-run measurements — illustrative, not a
rigorous benchmark.)
