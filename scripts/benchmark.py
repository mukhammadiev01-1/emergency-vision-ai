"""Model and Pipeline Latency & FPS Benchmarking Script."""
import argparse
import logging
import os
import time
from typing import List
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


def benchmark_model(
    model_path: str,
    device: str = "cpu",
    img_size: int = 640,
    iterations: int = 100,
    warmup: int = 10,
) -> None:
    """Benchmark inference latency and FPS for a model."""
    from ultralytics import YOLO

    logger.info("Loading model %s on device=%s...", model_path, device)
    model = YOLO(model_path)

    # Generate synthetic RGB frame
    synthetic_frame = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    # Warmup
    logger.info("Running %d warmup iterations...", warmup)
    for _ in range(warmup):
        model(synthetic_frame, device=device, verbose=False)

    # Benchmarking
    logger.info("Running %d benchmark iterations...", iterations)
    latencies: List[float] = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        model(synthetic_frame, device=device, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies_np = np.array(latencies)
    mean_ms = np.mean(latencies_np)
    median_ms = np.median(latencies_np)
    p95_ms = np.percentile(latencies_np, 95)
    p99_ms = np.percentile(latencies_np, 99)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0

    print("\n" + "=" * 50)
    print(f" Benchmark Summary: {model_path} ({device})")
    print("=" * 50)
    print(f" Iterations : {iterations}")
    print(f" Mean       : {mean_ms:.2f} ms")
    print(f" Median     : {median_ms:.2f} ms")
    print(f" P95        : {p95_ms:.2f} ms")
    print(f" P99        : {p99_ms:.2f} ms")
    print(f" Throughput : {fps:.2f} FPS")
    print("=" * 50 + "\n")


def main():
    default_model = "models/detection/yolo11n.pt" if os.path.exists("models/detection/yolo11n.pt") else "yolo11n.pt"
    parser = argparse.ArgumentParser(description="Benchmark vision model inference")
    parser.add_argument("--model", type=str, default=default_model, help="Model path (.pt or .onnx)")
    parser.add_argument("--device", type=str, default="cpu", help="Target device (cpu/cuda/mps)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    args = parser.parse_args()

    benchmark_model(
        model_path=args.model,
        device=args.device,
        img_size=args.imgsz,
        iterations=args.iterations,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
