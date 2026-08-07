import argparse
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import torch
import torch.ao.quantization as tq
import torchvision.models as models
from fvcore.nn import FlopCountAnalysis
from torch.profiler import profile, ProfilerActivity
from torchvision.models.quantization import resnet18 as quantizable_resnet18

# torch.ao.quantization eager-mode API is deprecated in favor of torchao but is
# still the simplest path for static INT8 quantization as of torch 2.8.
warnings.filterwarnings("ignore", message=".*torch.ao.quantization is deprecated.*")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_fp32_model(device):
    model = models.resnet18(weights=None)
    model.eval()
    return model.to(device)


def build_int8_model(n_calib_batches=5, calib_batch_size=1):
    """Static INT8 quantization via qnnpack. Quantized ops only run on CPU."""
    torch.backends.quantized.engine = "qnnpack"

    model = quantizable_resnet18(weights=None, quantize=False)
    model.eval()
    model.fuse_model()
    model.qconfig = tq.get_default_qconfig("qnnpack")
    tq.prepare(model, inplace=True)

    with torch.no_grad():
        for _ in range(n_calib_batches):
            model(torch.randn(calib_batch_size, 3, 224, 224))

    tq.convert(model, inplace=True)
    return model


def build_model(device, precision):
    if precision == "int8":
        return build_int8_model(), torch.device("cpu")
    return build_fp32_model(device), device


def measure_latency(model, device, batch_size, n_warmup=5, n_runs=50):
    inp = torch.randn(batch_size, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(inp)
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(inp)
        if device.type == "mps":
            torch.mps.synchronize()
        end = time.perf_counter()
    total_ms = (end - start) / n_runs * 1000
    per_image_ms = total_ms / batch_size
    throughput = batch_size / (total_ms / 1000)
    return total_ms, per_image_ms, throughput


def run_batch_sweep(model, device, precision, batch_sizes, output_path, n_warmup=5, n_runs=50):
    results = [(bs, *measure_latency(model, device, bs, n_warmup, n_runs)) for bs in batch_sizes]

    print("\n── Batch size sweep (CPU dispatch overhead vs. throughput) ──")
    print(f"{'Batch':>6} | {'Total (ms)':>11} | {'Per-image (ms)':>15} | {'Throughput (img/s)':>19}")
    print("-" * 62)
    for bs, total_ms, per_image_ms, throughput in results:
        print(f"{bs:>6} | {total_ms:>11.3f} | {per_image_ms:>15.4f} | {throughput:>19.1f}")

    data = {
        "device": str(device),
        "precision": precision,
        "n_warmup": n_warmup,
        "n_runs": n_runs,
        "results": [
            {
                "batch_size": bs,
                "total_ms": total_ms,
                "per_image_ms": per_image_ms,
                "throughput_img_per_s": throughput,
            }
            for bs, total_ms, per_image_ms, throughput in results
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nBatch sweep results saved to: {output_path}")


def run_op_profile(model, device, precision, output_path, trace_path, batch_size=1, n_warmup=3):
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    with profile(
        activities=[ProfilerActivity.CPU],  # MPS ops show up under CPU activity
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            _ = model(dummy_input)

    print("\n── Top 20 ops by CPU time ──────────────────────────────")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    ops = [
        {
            "name": e.key,
            "count": e.count,
            "cpu_time_total_us": e.cpu_time_total,
            "self_cpu_time_total_us": e.self_cpu_time_total,
            "cpu_time_avg_us": e.cpu_time_total / e.count if e.count else 0.0,
            "input_shapes": e.input_shapes,
            "cpu_memory_usage_bytes": e.cpu_memory_usage,
            "self_cpu_memory_usage_bytes": e.self_cpu_memory_usage,
        }
        for e in prof.key_averages(group_by_input_shape=True)
    ]
    ops.sort(key=lambda o: o["cpu_time_total_us"], reverse=True)

    data = {
        "device": str(device),
        "precision": precision,
        "batch_size": batch_size,
        "ops": ops,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nOp profile results saved to: {output_path}")

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace saved to: {trace_path}")
    print("Open it at chrome://tracing or https://ui.perfetto.dev")


def _tensor_bytes(t):
    return t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0


def _flatten_tensors(x):
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for item in x:
            yield from _flatten_tensors(item)


def _weight_bytes(module):
    total = sum(_tensor_bytes(p) for p in module.parameters(recurse=False))
    total += sum(_tensor_bytes(b) for b in module.buffers(recurse=False))
    return total


def _arithmetic_intensity_for_batch(model, device, precision, batch_size):
    """Returns None if fvcore's tracer can't handle this model (e.g. eager-mode quantized)."""
    inp = torch.randn(batch_size, 3, 224, 224).to(device)

    flop_analysis = FlopCountAnalysis(model, inp)
    flop_analysis.unsupported_ops_warnings(False)
    flop_analysis.uncalled_modules_warnings(False)
    try:
        flops_by_module = flop_analysis.by_module()
        whole_model_flops = flop_analysis.total()
    except RuntimeError as e:
        print(
            f"\nSkipping arithmetic-intensity analysis for precision={precision!r}: "
            f"fvcore's FlopCountAnalysis relies on torch.jit.trace, which doesn't "
            f"support this eager-mode quantized model's forward pass ({e})."
        )
        return None

    leaf_names = {name for name, m in model.named_modules() if next(m.children(), None) is None}

    io_bytes = defaultdict(lambda: [0, 0])

    def make_hook(name):
        def hook(module, inputs, output):
            io_bytes[name][0] += sum(_tensor_bytes(t) for t in _flatten_tensors(inputs))
            io_bytes[name][1] += sum(_tensor_bytes(t) for t in _flatten_tensors(output))
        return hook

    handles = [
        m.register_forward_hook(make_hook(name))
        for name, m in model.named_modules()
        if name in leaf_names
    ]
    with torch.no_grad():
        model(inp)
    for h in handles:
        h.remove()

    layers = []
    total_flops = 0
    total_bytes = 0
    for name, m in model.named_modules():
        if name not in leaf_names:
            continue
        layer_flops = flops_by_module.get(name, 0)
        in_bytes, out_bytes = io_bytes.get(name, (0, 0))
        w_bytes = _weight_bytes(m)
        layer_bytes = in_bytes + out_bytes + w_bytes
        if layer_flops == 0 and layer_bytes == 0:
            continue
        layers.append({
            "name": name,
            "type": type(m).__name__,
            "flops": layer_flops,
            "input_bytes": in_bytes,
            "output_bytes": out_bytes,
            "weight_bytes": w_bytes,
            "total_bytes": layer_bytes,
            "arithmetic_intensity": layer_flops / layer_bytes if layer_bytes else 0.0,
        })
        total_flops += layer_flops
        total_bytes += layer_bytes

    layers.sort(key=lambda l: l["flops"], reverse=True)
    return {
        "batch_size": batch_size,
        "whole_model_flops": whole_model_flops,
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "arithmetic_intensity": total_flops / total_bytes if total_bytes else 0.0,
        "layers": layers,
    }


def run_arithmetic_intensity(model, device, precision, output_path, batch_sizes=(1,)):
    """Per-layer and overall arithmetic intensity (ops / byte) via fvcore.nn, swept over batch size.

    Ops come from fvcore's FlopCountAnalysis, attributed per leaf module
    (conv/bn/relu/pool/linear). Bytes moved per leaf module are estimated as
    weights + input activations + output activations (captured via forward
    hooks), assuming no operator fusion. Functional ops that run directly in
    a block's forward -- the residual add, the final flatten -- aren't
    attributed to any leaf module, so they're excluded from both sides of
    the ratio; fvcore's whole-model FLOP total is reported separately for
    reference and will be marginally higher than the summed leaf FLOPs.

    FLOPs scale linearly with batch size, but weight bytes are read once per
    layer regardless of batch size (only activation bytes scale with batch),
    so AI is expected to rise with batch size and asymptote once weight
    reads become negligible relative to activation traffic.
    """
    results = []
    for i, batch_size in enumerate(batch_sizes):
        r = _arithmetic_intensity_for_batch(model, device, precision, batch_size)
        if r is None:
            return
        if i == 0:
            print(f"\n── Arithmetic intensity by layer (batch={batch_size}, top 20 by FLOPs) ──")
            print(f"{'Layer':<28} {'Type':<16} {'FLOPs':>14} {'Bytes':>12} {'AI (ops/B)':>11}")
            print("-" * 86)
            for l in r["layers"][:20]:
                print(
                    f"{l['name']:<28} {l['type']:<16} {l['flops']:>14,} "
                    f"{l['total_bytes']:>12,} {l['arithmetic_intensity']:>11.2f}"
                )
        results.append(r)

    print("\n── Arithmetic intensity vs. batch size ──")
    print(f"{'Batch':>6} | {'Total FLOPs':>14} | {'Total bytes':>14} | {'AI (ops/B)':>11}")
    print("-" * 56)
    for r in results:
        print(
            f"{r['batch_size']:>6} | {r['total_flops']:>14,} | "
            f"{r['total_bytes']:>14,} | {r['arithmetic_intensity']:>11.3f}"
        )

    data = {
        "device": str(device),
        "precision": precision,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nArithmetic intensity results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile ResNet18: batch-size sweep and/or per-op analysis."
    )
    parser.add_argument(
        "--mode",
        choices=["sweep", "ops", "ai", "both"],
        default="both",
        help=(
            'Which analysis to run: "sweep" (latency vs. batch size), "ops" '
            '(per-op CPU profile), "ai" (fvcore-based arithmetic intensity), '
            'or "both" (sweep+ops, default).'
        ),
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "int8"],
        default="fp32",
        help="Model precision. int8 uses static quantization (qnnpack) and always runs on CPU (default: fp32).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps"],
        default="auto",
        help="Device for fp32 runs (default: auto, prefers mps). Ignored for --precision int8, which always runs on CPU.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to write JSON results to (default: ./results).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="Batch sizes to use for the sweep.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"Running on: {device}")
    model, device = build_model(device, args.precision)
    if args.precision == "int8":
        print(f"INT8 quantization requires the qnnpack backend (CPU-only); running on: {device}")

    suffix = "_int8" if args.precision == "int8" else ("_cpu" if device.type == "cpu" else "")

    if args.mode in ("sweep", "both"):
        run_batch_sweep(
            model, device, args.precision, args.batch_sizes,
            args.output_dir / f"batch_sweep_resnet18{suffix}.json",
        )

    if args.mode in ("ops", "both"):
        run_op_profile(
            model, device, args.precision,
            args.output_dir / f"op_profile_resnet18{suffix}.json",
            args.output_dir / f"batch_sweep_resnet18{suffix}_trace.json",
        )

    if args.mode == "ai":
        # FLOPs/bytes are a static property of the model graph and batch size,
        # independent of device (confirmed identical on cpu vs. mps), so no
        # device suffix is needed here -- unlike the sweep/ops filenames above.
        run_arithmetic_intensity(
            model, device, args.precision,
            args.output_dir / "arithmetic_intensity_resnet18_batch.json",
            batch_sizes=args.batch_sizes,
        )


if __name__ == "__main__":
    main()
