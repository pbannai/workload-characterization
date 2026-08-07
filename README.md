# workload-characterization

Profiling utilities for characterizing model inference workloads on-device
(currently ResNet18 on an Apple M5 Pro's GPU via MPS, with an fp32/int8 precision comparison).

## Layout

- `scripts/profile_resnet18.py` — runs a batch-size latency/throughput sweep
  and/or a per-op CPU profile of ResNet18, in fp32 or statically-quantized
  INT8, writing results as JSON.
- `results/` — JSON outputs, Chrome traces, and the roofline plot PNG from the last run of each variant.

## Usage

```bash
python3 scripts/profile_resnet18.py --mode both
```

- `--mode sweep` — only run the batch-size sweep, writes `results/batch_sweep_resnet18*.json`
- `--mode ops` — only run the per-op profile, writes `results/op_profile_resnet18*.json`
- `--mode both` (default) — run both
- `--mode ai` — fvcore-based arithmetic-intensity analysis, swept over
  `--batch-sizes`, writes `results/arithmetic_intensity_resnet18_batch.json`.
  Not device-suffixed — FLOPs/bytes are a static property of the model graph
  and batch size, confirmed identical whether computed on CPU or MPS.
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

## Arithmetic intensity & roofline (ResNet18, FP32, batch=1..128)

fvcore's `FlopCountAnalysis` (`--mode ai`) gives a per-layer, device-independent
count of FLOPs and bytes moved, from which we can compute **arithmetic
intensity** (FLOPs/byte) and plot it against **attained performance**
(FLOPs/s, measured on the Apple M5 Pro's GPU via MPS) — a roofline view of
where this workload sits relative to the chip's advertised limits: 307 GB/s
memory bandwidth and 8.3 TFLOP/s (FP32) peak compute, both from Apple's
published M5 Pro specs. Ridge point (where the two roofs meet) sits at
AI ≈ 27.04 ops/byte, i.e. 8,300 ÷ 307.

| Batch | AI (ops/byte) | Attained (GFLOP/s) | Region |
|------:|---------------:|--------------------:|--------|
|     1 |          16.86 |               1,276 | memory-bound |
|     2 |          21.52 |               1,807 | memory-bound |
|     4 |          24.98 |               2,217 | memory-bound |
|     8 |          27.16 |               2,451 | compute-bound |
|    16 |          28.40 |               2,591 | compute-bound |
|    32 |          29.07 |               2,605 | compute-bound |
|    64 |          29.41 |               2,602 | compute-bound |
|   128 |          29.58 |               2,595 | compute-bound |

- **AI rises with batch size because weight reads get amortized.** Bytes moved
  = weights (fixed — read once per layer regardless of batch) + activations
  (scale linearly with batch). At batch=1, weight reads are a sizeable
  fraction of total bytes; by batch≥32 they're negligible next to activation
  traffic, so AI climbs toward an asymptote (~29.6 ops/byte) rather than
  growing indefinitely.

- **Batches 1/2/4 are memory-bound; batch 8 onward is compute-bound**, by the
  roofline's own arithmetic-intensity criterion — they fall left vs. right of
  the ridge point.

- **Even past the ridge, attained throughput never exceeds ~31% of peak
  compute** (2.6 of 8.3 TFLOP/s). So crossing into the "compute-bound" region
  arithmetically doesn't mean the GPU is anywhere near its ceiling — the
  remaining gap is overhead the roofline model doesn't capture by design
  (kernel-launch cost, no operator fusion across ResNet18's many small
  conv/BN/ReLU ops, etc.), not insufficient arithmetic intensity.

See `results/resnet18_roofline_fp32.png` for the plot and
`results/arithmetic_intensity_resnet18_batch.json` for the full per-layer,
per-batch-size breakdown.
