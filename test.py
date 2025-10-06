import torch
import time
import sys
import urllib.parse
import platform
from traceback import print_exc


def get_accelerator_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        total_memory_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )
        return device, device_name, total_memory_gb

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        device = torch.device("mps")
        device_name = "Apple MPS"
        return device, device_name, None

    return None, None, None


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_device_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def generate_github_issue_link(device_name, results, detailed_results):
    """生成GitHub issue链接，包含预填充的测试数据"""

    # 获取系统信息
    python_version = (
        f"py{sys.version_info.major}{sys.version_info.minor}{sys.version_info.micro}"
    )
    torch_version = f"torch{torch.__version__.replace('.', '').replace('+', '')}"

    # 构建issue标题
    title = f"新增性能数据：{device_name}"

    # 添加性能数据
    fp32_result = results.get("FP32", "N/A")
    tf32_result = results.get("TF32", "N/A")
    fp16_result = results.get("FP16", "N/A")
    bf16_result = results.get("BF16", "N/A")

    if fp32_result != "N/A":
        fp32_result = f"{fp32_result:.2f}"
    if tf32_result != "N/A":
        tf32_result = f"{tf32_result:.2f}"
    if fp16_result != "N/A":
        fp16_result = f"{fp16_result:.2f}"
    if bf16_result != "N/A":
        bf16_result = f"{bf16_result:.2f}"

    # 构建详细性能数据表格
    perf_details = "\n## 详细性能数据\n```\n"
    for precision_name, measurements in detailed_results.items():
        if measurements:
            perf_details += f"\n{precision_name}:\n"
            for size, tflops in measurements:
                perf_details += f"  {size}x{size}: {tflops:.2f} TFLOPS\n"
    perf_details += "```\n"

    # 构建简化的issue内容
    body = f"""## 设备信息
- 设备名称：{device_name}
- Python版本：{python_version}
- PyTorch版本：{torch_version}

## 性能数据
```
| {device_name} | {fp32_result} | {tf32_result} | {fp16_result} | {bf16_result} | **请填写note** | **请填写contributor** |
```

{perf_details}

## 填写说明
1. **note列**：请填写测试环境，包含以下关键字会自动归类：
   - `GCP` (GCP云实例)
   - `实体机` (物理机器)
   - `笔记本` (笔记本电脑)
   - `docker` (Docker容器)
   - `优云智算` (优云智算平台)
   - `智算云扉` (智算云扉平台)

2. **contributor列**：格式为 `[用户名](https://github.com/用户名)`，不填默认你自己

感谢您的贡献！"""

    # URL编码
    encoded_title = urllib.parse.quote(title)
    encoded_body = urllib.parse.quote(body)

    # 生成GitHub issue链接
    issue_url = f"https://github.com/zzc0721/torch-performance-test-data/issues/new?title={encoded_title}&body={encoded_body}"

    print(f"\n{'=' * 60}")
    print("🎉 测试完成！")
    print("\n📊 性能数据摘要：")
    print(f"设备：{device_name}")
    print(
        f"FP32: {fp32_result} TFLOPS | TF32: {tf32_result} TFLOPS | FP16: {fp16_result} TFLOPS | BF16: {bf16_result} TFLOPS"
    )

    # 如果有 FP8 数据，单独显示
    fp8_result = results.get("FP8 E4M3FN", "N/A")
    if fp8_result != "N/A":
        print(f"FP8: {fp8_result:.2f} TFLOPS")
    print("\n🔗 提交数据请点击以下链接：")
    print(f"{issue_url}")
    print(f"\n{'=' * 60}")
    print("💡 提示：")
    print("1. 点击链接会自动填充设备信息和性能数据")
    print("2. 请在issue中填写note（测试环境）和contributor信息")
    print("3. 包含特定关键字的note将被自动归类")


def benchmark_precision(precision, matrix_size, device, warmup=10, test_iters=30):
    if device.type not in ("cuda", "mps"):
        raise RuntimeError("当前仅支持CUDA或MPS设备进行测试")

    # 初始化矩阵
    try:
        if precision == torch.int8:
            # INT8 特殊处理
            a = torch.randint(
                -128, 127, (matrix_size, matrix_size), dtype=precision, device=device
            )
            b = torch.randint(
                -128, 127, (matrix_size, matrix_size), dtype=precision, device=device
            )
        elif precision == torch.float8_e4m3fn:
            # FP8 特殊处理
            a = torch.randn(matrix_size, matrix_size, device=device)
            b = torch.randn(matrix_size, matrix_size, device=device)
            # 防止全为0，重新赋值非零随机数
            a = a + torch.randn_like(a) * 1e-3
            b = b + torch.randn_like(b) * 1e-3
            a = a.to(dtype=precision)
            b = b.to(dtype=precision)
        else:
            a = torch.randn(matrix_size, matrix_size, dtype=precision, device=device)
            b = torch.randn(matrix_size, matrix_size, dtype=precision, device=device)
    except RuntimeError as e:
        message = str(e).lower()
        if "not implemented" in message or "not support" in message:
            return None
        print_exc()
        raise

    # 预热
    for _ in range(warmup):
        torch.mm(a, b)
    synchronize_device(device)

    # 正式测试 - 使用 CUDA Event 进行精确计时
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(test_iters):
            torch.mm(a, b)
        end_event.record()
        torch.cuda.synchronize()

        elapsed = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
    else:
        # MPS 使用原有的 time.time() 方式
        start_time = time.time()
        for _ in range(test_iters):
            torch.mm(a, b)
        synchronize_device(device)
        elapsed = time.time() - start_time

    # 计算FLOPS
    flops_per_iter = 2 * matrix_size**3
    total_flops = flops_per_iter * test_iters
    tflops = (total_flops / elapsed) / 1e12

    # 清理显存
    del a, b
    empty_device_cache(device)

    return tflops


if __name__ == "__main__":
    device, device_name, total_memory_gb = get_accelerator_device()
    if device is None:
        print("未检测到CUDA或MPS设备")
        exit(1)

    print(f"测试设备: {device_name}")
    if total_memory_gb is not None:
        print(f"显存大小: {total_memory_gb:.1f} GB\n")
    elif device.type == "mps":
        print("使用Apple MPS图形加速器\n")

    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")

    # 显式设置 TF32 标志（避免 Ampere+ 显卡的伪 FP32 问题）
    if device.type == "cuda" and hasattr(torch.backends.cuda, "matmul"):
        print(f"\n当前 TF32 设置:")
        print(f"  matmul.allow_tf32 = {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  cudnn.allow_tf32 = {torch.backends.cudnn.allow_tf32}")

    # 测试不同精度和矩阵大小
    matrix_sizes = [1024, 2048, 4096, 8192, 10240]
    precisions = [
        ("FP32", torch.float32, False),  # 禁用 TF32
        ("TF32", torch.float32, True),   # 启用 TF32（仅 Ampere+ 显卡）
        ("FP16", torch.float16, None),   # FP16 不受 TF32 影响
        ("BF16", torch.bfloat16, None),  # BF16 不受 TF32 影响
        # ("INT8", torch.int8, None),    # 可选：如果需要测试INT8
    ]

    try:
        # 检查FP8类型是否存在并且实际可用
        test_a = torch.randn(2, 2, device=device).to(torch.float8_e4m3fn)
        test_b = torch.randn(2, 2, device=device).to(torch.float8_e4m3fn)
        _ = torch.mm(test_a, test_b)  # 测试实际运算
        fp8_precision = torch.float8_e4m3fn
        del test_a, test_b
        precisions.append(("FP8 E4M3FN", fp8_precision, None))
    except (AttributeError, RuntimeError) as e:
        print("\nPyTorch 当前不支持 FP8 E4M3FN，跳过该项测试")
        if isinstance(e, RuntimeError):
            print(f"  原因: {str(e)}")

    results = {}
    for precision_name, precision, tf32_setting in precisions:
        # 设置 TF32（仅对 FP32 精度）
        if device.type == "cuda" and tf32_setting is not None and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = tf32_setting
            torch.backends.cudnn.allow_tf32 = tf32_setting
            print(f"\n[{precision_name}] 已设置 TF32 = {tf32_setting}")

        print(f"\n测试 {precision_name}:")
        results[precision_name] = []

        for size in matrix_sizes:
            # 避免在循环内打印过多，减少隐式同步
            tflops = benchmark_precision(precision, size, device)
            if tflops is not None:
                results[precision_name].append((size, tflops))
            else:
                print(f"  不支持此精度")
                break

        # 在该精度所有测试完成后统一打印
        if results[precision_name]:
            print(f"  完成 {len(results[precision_name])} 个尺寸测试")

    # 打印总结
    print("\n" + "=" * 60)
    print("性能总结:")
    best_results = {}
    for precision_name, measurements in results.items():
        if measurements:
            best_size, max_tflops = max(measurements, key=lambda x: x[1])
            best_results[precision_name] = max_tflops
            print(f"{precision_name:12} 最大算力: {max_tflops:7.2f} TFLOPS @ {best_size}x{best_size}")
    print("=" * 60)

    # 生成GitHub issue链接
    generate_github_issue_link(device_name, best_results, results)
