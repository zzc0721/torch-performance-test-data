from __future__ import annotations

import importlib.util
import io
import sys
import types
import unittest
import urllib.parse
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


def load_benchmark_module():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        fake_torch = types.ModuleType("torch")
        fake_torch.float32 = object()
        fake_torch.float16 = object()
        fake_torch.bfloat16 = object()
        fake_torch.__version__ = "test"
        fake_torch.backends = types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False))
        )
        fake_torch.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
        sys.modules["torch"] = fake_torch

    module_path = Path(__file__).parents[1] / "test.py"
    spec = importlib.util.spec_from_file_location("gpu_benchmark", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = load_benchmark_module()


class BenchmarkLogicTests(unittest.TestCase):
    def device_info(self, backend="CUDA", capability=(8, 0)):
        device = types.SimpleNamespace(type="cuda")
        return benchmark.DeviceInfo(device, "Test GPU", 24.0, backend, capability)

    def test_tf32_requires_nvidia_ampere_or_newer(self):
        self.assertTrue(benchmark.supports_tf32(self.device_info()))
        self.assertFalse(benchmark.supports_tf32(self.device_info(capability=(7, 5))))
        self.assertFalse(benchmark.supports_tf32(self.device_info(backend="ROCm")))
        self.assertFalse(
            benchmark.supports_tf32(self.device_info(backend="MPS", capability=None))
        )

    def test_precision_list_omits_unsupported_tf32(self):
        specs = benchmark.build_precision_specs(
            self.device_info(backend="MPS", capability=None)
        )
        self.assertEqual([spec.name for spec in specs], ["FP32", "FP16", "BF16"])

    def test_apple_device_name_uses_sysctl_chip_name(self):
        completed = types.SimpleNamespace(stdout="Apple M1 Max\n")
        with (
            mock.patch.object(benchmark.sys, "platform", "darwin"),
            mock.patch.object(benchmark.subprocess, "run", return_value=completed),
        ):
            self.assertEqual(benchmark.get_apple_device_name(), "Apple M1 Max")

    def test_tf32_mode_restores_previous_value(self):
        matmul = benchmark.torch.backends.cuda.matmul
        matmul.allow_tf32 = False
        with benchmark.tf32_mode(self.device_info(), True):
            self.assertTrue(matmul.allow_tf32)
        self.assertFalse(matmul.allow_tf32)

    def test_choose_iterations_targets_duration_and_honors_cap(self):
        self.assertEqual(benchmark.choose_iterations(0.03, 3, 0.2, 10000), 20)
        self.assertEqual(benchmark.choose_iterations(0.001, 3, 0.2, 100), 100)
        self.assertEqual(benchmark.choose_iterations(10.0, 3, 0.2, 100), 1)

    def test_summary_uses_median_and_reports_variation(self):
        result = benchmark.summarize_rates([10.0, 100.0, 11.0], 1024, 20)
        self.assertEqual(result.median_tflops, 11.0)
        self.assertEqual(result.min_tflops, 10.0)
        self.assertEqual(result.max_tflops, 100.0)
        self.assertGreater(result.cv_percent, 0)

    def test_memory_preflight_skips_before_allocating(self):
        spec = benchmark.PrecisionSpec("FP32", benchmark.torch.float32)
        with (
            mock.patch.object(benchmark, "available_device_memory", return_value=100),
            mock.patch.object(benchmark, "estimated_tensor_memory", return_value=90),
            mock.patch.object(benchmark, "make_matrices") as make_matrices,
        ):
            measurement, reason = benchmark.benchmark_precision(
                spec, 1024, self.device_info(), memory_fraction=0.8
            )
        self.assertIsNone(measurement)
        self.assertIn("当前可用", reason)
        make_matrices.assert_not_called()

    def test_error_classification(self):
        self.assertTrue(benchmark.is_oom_error(RuntimeError("CUDA out of memory")))
        self.assertTrue(
            benchmark.is_unsupported_error(RuntimeError("operation not implemented"))
        )
        self.assertFalse(benchmark.is_unsupported_error(RuntimeError("driver reset")))

    def test_issue_contains_v2_metadata_and_na(self):
        measurement = benchmark.Measurement(1024, 12.5, 12.0, 13.0, 2.1, 8, 5)
        output = io.StringIO()
        settings = benchmark.BenchmarkSettings(
            sizes=(1024,),
            warmup=5,
            repeats=5,
            target_seconds=0.2,
            max_iterations=10000,
            memory_fraction=0.8,
            include_fp8=False,
        )
        with redirect_stdout(output):
            benchmark.generate_github_issue_link(
                self.device_info(),
                {"FP32": 12.5},
                {"FP32": [measurement]},
                {"BF16": "unsupported"},
                settings,
            )
        url = next(
            line
            for line in output.getvalue().splitlines()
            if line.startswith("https://")
        )
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        body = query["body"][0]
        self.assertIn("Benchmark版本：v2.0", body)
        self.assertIn("| Test GPU | 12.50 | N/A | N/A | N/A |", body)
        self.assertIn("benchmark v2.0", body)
        self.assertIn("矩阵尺寸：1024", body)

    def test_summary_does_not_append_units_to_na(self):
        output = io.StringIO()
        settings = benchmark.BenchmarkSettings(
            sizes=(1024,),
            warmup=5,
            repeats=5,
            target_seconds=0.2,
            max_iterations=10000,
            memory_fraction=0.8,
            include_fp8=False,
        )
        with redirect_stdout(output):
            benchmark.generate_github_issue_link(
                self.device_info(), {}, {}, {}, settings
            )
        self.assertIn("TF32: N/A", output.getvalue())
        self.assertNotIn("N/A TFLOPS", output.getvalue())


if __name__ == "__main__":
    unittest.main()
