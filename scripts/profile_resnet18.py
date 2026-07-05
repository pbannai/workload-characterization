import argparse
import json
import time
from pathlib import Path

import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_model(device):
    model = models.resnet18(weights=None)
    model.eval()
    return model.to(device)


def measure_latency(model, device, batch_size, n_warmup=5, n_runs=50):
    inp = torch.randn(batch_size, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(inp)
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(inp)
        torch.mps.synchronize()
        end = time.perf_counter()
    total_ms = (end - start) / n_runs * 1000
    per_image_ms = total_ms / batch_size
    throughput = batch_size / (total_ms / 1000)
    return total_ms, per_image_ms, throughput


def run_batch_sweep(model, device, batch_sizes, output_path, n_warmup=5, n_runs=50):
    results = [(bs, *measure_latency(model, device, bs, n_warmup, n_runs)) for bs in batch_sizes]

    print("\n── Batch size sweep (CPU dispatch overhead vs. throughput) ──")
    print(f"{'Batch':>6} | {'Total (ms)':>11} | {'Per-image (ms)':>15} | {'Throughput (img/s)':>19}")
    print("-" * 62)
    for bs, total_ms, per_image_ms, throughput in results:
        print(f"{bs:>6} | {total_ms:>11.3f} | {per_image_ms:>15.4f} | {throughput:>19.1f}")

    data = {
        "device": str(device),
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


def run_op_profile(model, device, output_path, batch_size=1, n_warmup=3):
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
        "batch_size": batch_size,
        "ops": ops,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nOp profile results saved to: {output_path}")

    trace_path = output_path.parent / "batch_sweep_resnet18_trace.json"
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

    device = get_device()
    print(f"Running on: {device}")
    model = build_model(device)

    if args.mode in ("sweep", "both"):
        run_batch_sweep(model, device, args.batch_sizes, args.output_dir / "batch_sweep_resnet18.json")

    if args.mode in ("ops", "both"):
        run_op_profile(model, device, args.output_dir / "op_profile_resnet18.json")


if __name__ == "__main__":
    main()
