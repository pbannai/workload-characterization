import argparse
import json
import time
import warnings
from pathlib import Path

import torch
import torch.ao.quantization as tq
import torchvision.models as models
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


def main():
    parser = argparse.ArgumentParser(
        description="Profile ResNet18: batch-size sweep and/or per-op analysis."
    )
    parser.add_argument(
        "--mode",
        choices=["sweep", "ops", "both"],
        default="both",
        help="Which analysis to run (default: both).",
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


if __name__ == "__main__":
    main()
