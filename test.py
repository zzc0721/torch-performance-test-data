from __future__ import annotations

import argparse
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass

import torch

BENCHMARK_VERSION = "2.0"
DEFAULT_MATRIX_SIZES = (1024, 2048, 4096, 8192, 10240)


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    name: str
    total_memory_gb: float | None
    backend: str
    capability: tuple[int, int] | None = None


@dataclass(frozen=True)
class PrecisionSpec:
    name: str
    dtype: torch.dtype
    use_tf32: bool = False
    experimental: bool = False


@dataclass(frozen=True)
class Measurement:
    size: int
    median_tflops: float
    min_tflops: float
    max_tflops: float
    cv_percent: float
    iterations: int
    repeats: int


@dataclass(frozen=True)
class ProbeResult:
    supported: bool
    reason: str = ""
    relative_rmse: float | None = None


@dataclass(frozen=True)
class BenchmarkSettings:
    sizes: tuple[int, ...]
    warmup: int
    repeats: int
    target_seconds: float
    max_iterations: int
    memory_fraction: float
    include_fp8: bool


def get_apple_device_name() -> str:
    if sys.platform != "darwin":
        return "Apple MPS"
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        chip_name = result.stdout.strip()
        if chip_name:
            return chip_name
    except (OSError, subprocess.SubprocessError):
        pass
    return "Apple MPS"


def get_system_memory_gb() -> float | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return page_size * page_count / 1024**3


def get_accelerator_device(device_arg: str | None = None) -> DeviceInfo | None:
    if device_arg is not None:
        requested = torch.device(device_arg)
        if requested.type not in ("cuda", "mps"):
            raise ValueError("--device 目前仅支持 cuda、cuda:N 或 mps")
    elif torch.cuda.is_available():
        requested = torch.device("cuda:0")
    else:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            return None
        requested = torch.device("mps")

    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求了 CUDA 设备，但当前 PyTorch 无法使用 CUDA/ROCm")
        index = requested.index if requested.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"设备 cuda:{index} 不存在，当前仅检测到 {torch.cuda.device_count()} 个设备"
            )
        requested = torch.device(f"cuda:{index}")
        properties = torch.cuda.get_device_properties(index)
        hip_version = getattr(torch.version, "hip", None)
        backend = "ROCm" if hip_version else "CUDA"
        capability = None
        if not hip_version:
            try:
                capability = tuple(torch.cuda.get_device_capability(index))
            except (AttributeError, RuntimeError):
                pass
        return DeviceInfo(
            device=requested,
            name=torch.cuda.get_device_name(index),
            total_memory_gb=properties.total_memory / 1024**3,
            backend=backend,
            capability=capability,
        )

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None or not mps_backend.is_available():
        raise RuntimeError("请求了 MPS 设备，但当前 PyTorch 无法使用 MPS")
    return DeviceInfo(
        requested,
        get_apple_device_name(),
        get_system_memory_gb(),
        "MPS",
    )


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def supports_tf32(device_info: DeviceInfo) -> bool:
    if device_info.backend != "CUDA" or device_info.capability is None:
        return False
    if device_info.capability < (8, 0):
        return False
    cuda_backend = getattr(torch.backends, "cuda", None)
    return cuda_backend is not None and hasattr(cuda_backend, "matmul")


def build_precision_specs(
    device_info: DeviceInfo, include_fp8: bool = False
) -> list[PrecisionSpec]:
    specs = [PrecisionSpec("FP32", torch.float32)]
    if supports_tf32(device_info):
        specs.append(PrecisionSpec("TF32", torch.float32, use_tf32=True))
    specs.extend(
        [
            PrecisionSpec("FP16", torch.float16),
            PrecisionSpec("BF16", torch.bfloat16),
        ]
    )
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if include_fp8 and fp8_dtype is not None:
        specs.append(PrecisionSpec("FP8 E4M3FN", fp8_dtype, experimental=True))
    return specs


@contextmanager
def tf32_mode(device_info: DeviceInfo, enabled: bool):
    if device_info.backend != "CUDA" or not hasattr(torch.backends.cuda, "matmul"):
        yield
        return

    matmul_backend = torch.backends.cuda.matmul
    previous = matmul_backend.allow_tf32
    matmul_backend.allow_tf32 = enabled
    try:
        yield
    finally:
        matmul_backend.allow_tf32 = previous


def is_oom_error(error: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(error, oom_type):
        return True
    message = str(error).lower()
    return "out of memory" in message or "not enough memory" in message


def is_unsupported_error(error: BaseException) -> bool:
    message = str(error).lower()
    markers = (
        "not implemented",
        "not support",
        "unsupported",
        "not available for",
        "does not support",
        "invalid device function",
    )
    return any(marker in message for marker in markers)


def make_matrices(
    dtype: torch.dtype, size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is not None and dtype == fp8_dtype:
        # Most backends cannot generate random FP8 tensors directly.
        a = torch.randn((size, size), dtype=torch.float16, device=device).to(dtype)
        b = torch.randn((size, size), dtype=torch.float16, device=device).to(dtype)
    else:
        a = torch.randn((size, size), dtype=dtype, device=device)
        b = torch.randn((size, size), dtype=dtype, device=device)
    output = torch.empty((size, size), dtype=dtype, device=device)
    return a, b, output


def relative_rmse(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual_float = actual.detach().float().cpu()
    reference_float = reference.detach().float().cpu()
    error_rms = (actual_float - reference_float).square().mean().sqrt().item()
    reference_rms = reference_float.square().mean().sqrt().item()
    return error_rms / max(reference_rms, 1e-12)


def correctness_threshold(spec: PrecisionSpec) -> float:
    if spec.experimental:
        return 0.35
    if spec.name == "BF16":
        return 0.05
    if spec.name in ("FP16", "TF32"):
        return 0.02
    return 0.001


def probe_precision(spec: PrecisionSpec, device_info: DeviceInfo) -> ProbeResult:
    device = device_info.device
    a = b = output = None
    try:
        torch.manual_seed(20260804)
        a, b, output = make_matrices(spec.dtype, 64, device)
        torch.mm(a, b, out=output)
        synchronize_device(device)

        if not bool(torch.isfinite(output.float()).all().item()):
            return ProbeResult(False, "正确性检查产生 NaN 或 Inf")

        reference = torch.mm(a.float().cpu(), b.float().cpu())
        error = relative_rmse(output, reference)
        threshold = correctness_threshold(spec)
        if error > threshold:
            return ProbeResult(
                False,
                f"正确性检查失败：相对 RMSE {error:.3g} > {threshold:.3g}",
                error,
            )
        return ProbeResult(True, relative_rmse=error)
    except (RuntimeError, TypeError) as error:
        if is_unsupported_error(error):
            return ProbeResult(False, f"后端不支持该矩阵运算：{error}")
        if is_oom_error(error):
            return ProbeResult(False, "正确性检查时显存不足")
        return ProbeResult(False, f"精度探测失败：{error}")
    finally:
        del a, b, output
        empty_device_cache(device)


def available_device_memory(device: torch.device) -> int | None:
    if device.type == "cuda" and hasattr(torch.cuda, "mem_get_info"):
        try:
            free_memory, _ = torch.cuda.mem_get_info(device)
        except TypeError:
            with torch.cuda.device(device):
                free_memory, _ = torch.cuda.mem_get_info()
        except RuntimeError:
            return None
        return int(free_memory)
    if device.type == "mps":
        recommended = getattr(torch.mps, "recommended_max_memory", None)
        allocated = getattr(torch.mps, "current_allocated_memory", None)
        if recommended is not None and allocated is not None:
            try:
                return max(0, int(recommended()) - int(allocated()))
            except RuntimeError:
                return None
    return None


def estimated_tensor_memory(dtype: torch.dtype, size: int) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is not None and dtype == fp8_dtype:
        # FP8 inputs are generated through a temporary FP16 tensor.
        return 5 * size * size
    return 3 * size * size * element_size


def measure_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    device: torch.device,
    iterations: int,
) -> float:
    synchronize_device(device)
    if device.type == "cuda":
        # Events must be created and recorded on the selected device. Without
        # this context, --device cuda:N can accidentally time cuda:0's stream.
        with torch.cuda.device(device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                torch.mm(a, b, out=output)
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / 1000.0

    started_at = time.perf_counter()
    for _ in range(iterations):
        torch.mm(a, b, out=output)
    synchronize_device(device)
    return time.perf_counter() - started_at


def choose_iterations(
    calibration_seconds: float,
    calibration_iterations: int,
    target_seconds: float,
    max_iterations: int,
) -> int:
    if calibration_seconds <= 0:
        return max_iterations
    estimate = target_seconds * calibration_iterations / calibration_seconds
    estimated = math.ceil(estimate - 1e-12)
    return max(1, min(max_iterations, estimated))


def summarize_rates(rates: list[float], size: int, iterations: int) -> Measurement:
    median = statistics.median(rates)
    cv_percent = 0.0
    if len(rates) > 1 and statistics.mean(rates) > 0:
        cv_percent = statistics.stdev(rates) / statistics.mean(rates) * 100.0
    return Measurement(
        size=size,
        median_tflops=median,
        min_tflops=min(rates),
        max_tflops=max(rates),
        cv_percent=cv_percent,
        iterations=iterations,
        repeats=len(rates),
    )


def benchmark_precision(
    spec: PrecisionSpec,
    matrix_size: int,
    device_info: DeviceInfo,
    warmup: int = 5,
    repeats: int = 5,
    target_seconds: float = 0.2,
    max_iterations: int = 10000,
    memory_fraction: float = 0.8,
) -> tuple[Measurement | None, str]:
    device = device_info.device
    free_memory = available_device_memory(device)
    required_memory = estimated_tensor_memory(spec.dtype, matrix_size)
    if free_memory is not None and required_memory > free_memory * memory_fraction:
        return None, (
            f"预计至少需要 {required_memory / 1024**3:.2f} GiB，"
            f"当前可用 {free_memory / 1024**3:.2f} GiB"
        )

    a = b = output = None
    try:
        torch.manual_seed(20260804 + matrix_size)
        a, b, output = make_matrices(spec.dtype, matrix_size, device)
        for _ in range(warmup):
            torch.mm(a, b, out=output)
        synchronize_device(device)

        calibration_iterations = 3
        calibration_seconds = measure_mm(a, b, output, device, calibration_iterations)
        iterations = choose_iterations(
            calibration_seconds,
            calibration_iterations,
            target_seconds,
            max_iterations,
        )

        total_flops = 2 * matrix_size**3 * iterations
        rates = []
        for _ in range(repeats):
            elapsed = measure_mm(a, b, output, device, iterations)
            if elapsed <= 0:
                return None, "计时器返回了非正耗时"
            rates.append(total_flops / elapsed / 1e12)
        return summarize_rates(rates, matrix_size, iterations), ""
    except (RuntimeError, TypeError) as error:
        if is_oom_error(error):
            return None, "显存不足"
        if is_unsupported_error(error):
            return None, f"后端不支持：{error}"
        return None, f"测试失败：{error}"
    finally:
        del a, b, output
        empty_device_cache(device)


def format_result(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def generate_github_issue_link(
    device_info: DeviceInfo,
    best_results: dict[str, float],
    detailed_results: dict[str, list[Measurement]],
    skipped: dict[str, str],
    settings: BenchmarkSettings,
) -> None:
    python_version = platform.python_version()
    torch_version = torch.__version__
    torch_version_info = getattr(torch, "version", None)
    backend_version = getattr(torch_version_info, "hip", None)
    if backend_version is None:
        backend_version = getattr(torch_version_info, "cuda", None)
    backend_version = backend_version or "N/A"
    title = f"新增性能数据：{device_info.name}"
    summary_names = ("FP32", "TF32", "FP16", "BF16")
    summary = {name: format_result(best_results.get(name)) for name in summary_names}

    detail_lines = ["## 详细性能数据", "```"]
    for precision_name, measurements in detailed_results.items():
        if not measurements:
            continue
        detail_lines.append(f"\n{precision_name}:")
        for measurement in measurements:
            detail_lines.append(
                f"  {measurement.size}x{measurement.size}: "
                f"{measurement.median_tflops:.2f} TFLOPS "
                f"(CV {measurement.cv_percent:.1f}%, "
                f"n={measurement.repeats}x{measurement.iterations})"
            )
    if skipped:
        detail_lines.append("\n跳过项目:")
        for name, reason in skipped.items():
            detail_lines.append(f"  {name}: {reason}")
    detail_lines.append("```")
    perf_details = "\n".join(detail_lines)

    body = f"""## 设备信息
- 设备名称：{device_info.name}
- 后端：{device_info.backend}
- 后端运行时版本：{backend_version}
- 操作系统：{platform.platform()}
- Python版本：{python_version}
- PyTorch版本：{torch_version}
- Benchmark版本：v{BENCHMARK_VERSION}

## Benchmark配置
- 矩阵尺寸：{", ".join(map(str, settings.sizes))}
- 预热次数：{settings.warmup}
- 采样组数：{settings.repeats}
- 每组目标时长：{settings.target_seconds:g}秒
- 每组最大迭代数：{settings.max_iterations}
- 可用显存比例上限：{settings.memory_fraction:g}

## 性能数据
```
| {device_info.name} | {summary["FP32"]} | {summary["TF32"]} | {summary["FP16"]} | {summary["BF16"]} | benchmark v{BENCHMARK_VERSION}; **请填写环境** | **请填写contributor** |
```

{perf_details}

## 填写说明
1. **note列**：请填写测试环境（实体机、笔记本、Docker或云平台）
2. **contributor列**：格式为 `[用户名](https://github.com/用户名)`

感谢您的贡献！"""

    issue_url = (
        "https://github.com/zzc0721/torch-performance-test-data/issues/new?"
        f"title={urllib.parse.quote(title)}&body={urllib.parse.quote(body)}"
    )

    print(f"\n{'=' * 60}")
    print("测试完成！")
    print("\n性能数据摘要：")
    print(f"设备：{device_info.name}")
    print(
        " | ".join(
            f"{name}: {summary[name]}" + (" TFLOPS" if summary[name] != "N/A" else "")
            for name in summary_names
        )
    )
    fp8_result = best_results.get("FP8 E4M3FN")
    if fp8_result is not None:
        print(f"FP8 E4M3FN（实验性）: {fp8_result:.2f} TFLOPS")
    print("\n提交数据请打开以下链接：")
    print(issue_url)
    print(f"{'=' * 60}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def fraction(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("必须大于 0 且不大于 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU GEMM 性能基准测试 v2")
    parser.add_argument("--device", help="指定设备，例如 cuda:0 或 mps")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=positive_int,
        default=list(DEFAULT_MATRIX_SIZES),
        metavar="N",
        help="方阵尺寸，默认：%(default)s",
    )
    parser.add_argument("--warmup", type=positive_int, default=5, help="预热次数")
    parser.add_argument("--repeats", type=positive_int, default=5, help="独立采样组数")
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=0.2,
        help="每组目标计时时长，默认 0.2 秒",
    )
    parser.add_argument(
        "--max-iterations",
        type=positive_int,
        default=10000,
        help="每组最大矩阵乘次数",
    )
    parser.add_argument(
        "--memory-fraction",
        type=fraction,
        default=0.8,
        help="最多使用当前可用显存的比例，默认 0.8",
    )
    parser.add_argument(
        "--include-fp8",
        action="store_true",
        help="启用实验性 FP8 测试（不属于榜单标准四列）",
    )
    args = parser.parse_args(argv)
    if args.target_seconds <= 0:
        parser.error("--target-seconds 必须大于 0")
    return args


def settings_from_args(args: argparse.Namespace) -> BenchmarkSettings:
    return BenchmarkSettings(
        sizes=tuple(args.sizes),
        warmup=args.warmup,
        repeats=args.repeats,
        target_seconds=args.target_seconds,
        max_iterations=args.max_iterations,
        memory_fraction=args.memory_fraction,
        include_fp8=args.include_fp8,
    )


def print_environment(device_info: DeviceInfo) -> None:
    print(f"Benchmark版本: v{BENCHMARK_VERSION}")
    print(f"测试设备: {device_info.name}")
    print(f"计算后端: {device_info.backend}")
    if device_info.capability is not None:
        print(f"CUDA计算能力: {device_info.capability[0]}.{device_info.capability[1]}")
    if device_info.backend == "MPS" and device_info.total_memory_gb is not None:
        print(f"统一内存大小: {device_info.total_memory_gb:.1f} GiB")
    elif device_info.total_memory_gb is not None:
        print(f"显存大小: {device_info.total_memory_gb:.1f} GiB")
    elif device_info.backend == "MPS":
        print("显存: Apple统一内存（PyTorch不提供可靠的总容量）")
    print(f"Python版本: {platform.python_version()}")
    print(f"PyTorch版本: {torch.__version__}")


def run_benchmark(args: argparse.Namespace) -> int:
    try:
        device_info = get_accelerator_device(args.device)
    except (ValueError, RuntimeError) as error:
        print(f"设备初始化失败：{error}", file=sys.stderr)
        return 2
    if device_info is None:
        print("未检测到可用的 CUDA、ROCm 或 MPS 设备", file=sys.stderr)
        return 1

    settings = settings_from_args(args)
    print_environment(device_info)
    specs = build_precision_specs(device_info, settings.include_fp8)
    if not supports_tf32(device_info):
        print("TF32: N/A（仅测试支持 TF32 的 NVIDIA Ampere 或更新架构）")

    results: dict[str, list[Measurement]] = {}
    skipped: dict[str, str] = {}
    for spec in specs:
        label = f"{spec.name}（实验性）" if spec.experimental else spec.name
        print(f"\n测试 {label}:")
        results[spec.name] = []
        with tf32_mode(device_info, spec.use_tf32):
            probe = probe_precision(spec, device_info)
            if not probe.supported:
                skipped[spec.name] = probe.reason
                print(f"  跳过：{probe.reason}")
                continue
            print(f"  正确性检查通过（相对 RMSE {probe.relative_rmse:.3g}）")

            for size in settings.sizes:
                measurement, reason = benchmark_precision(
                    spec,
                    size,
                    device_info,
                    warmup=settings.warmup,
                    repeats=settings.repeats,
                    target_seconds=settings.target_seconds,
                    max_iterations=settings.max_iterations,
                    memory_fraction=settings.memory_fraction,
                )
                if measurement is None:
                    print(f"  {size}x{size}: 跳过（{reason}）")
                    continue
                results[spec.name].append(measurement)
                print(
                    f"  {size}x{size}: {measurement.median_tflops:.2f} TFLOPS "
                    f"[中位数, CV {measurement.cv_percent:.1f}%, "
                    f"{measurement.repeats}组 x {measurement.iterations}次]"
                )

    print(f"\n{'=' * 60}")
    print("性能总结（每个尺寸先取多组中位数，再选择最高尺寸）：")
    best_results: dict[str, float] = {}
    for precision_name, measurements in results.items():
        if not measurements:
            continue
        best = max(measurements, key=lambda item: item.median_tflops)
        best_results[precision_name] = best.median_tflops
        suffix = "（实验性）" if precision_name.startswith("FP8") else ""
        print(
            f"{precision_name:12} {best.median_tflops:8.2f} TFLOPS "
            f"@ {best.size}x{best.size}，CV {best.cv_percent:.1f}%{suffix}"
        )
    print("=" * 60)

    generate_github_issue_link(device_info, best_results, results, skipped, settings)
    return 0 if best_results else 1


if __name__ == "__main__":
    raise SystemExit(run_benchmark(parse_args()))
