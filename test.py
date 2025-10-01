import torch
import time
import sys
import urllib.parse
import platform
from traceback import print_exc


def generate_github_issue_link(device_name, results):
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
    fp16_result = results.get("FP16", "N/A")
    bf16_result = results.get("BF16", "N/A")
    fp8_result = results.get("FP8 E4M3FN", "N/A")

    if fp32_result != "N/A":
        fp32_result = f"{fp32_result:.2f}"
    if fp16_result != "N/A":
        fp16_result = f"{fp16_result:.2f}"
    if bf16_result != "N/A":
        bf16_result = f"{bf16_result:.2f}"
    if fp8_result != "N/A":
        fp8_result = f"{fp8_result:.2f}"

    # 构建简化的issue内容
    body = f"""## 设备信息
- 设备名称：{device_name}
- Python版本：{python_version}
- PyTorch版本：{torch_version}

## 性能数据
```
| {device_name} | {fp32_result} | {fp16_result} | {bf16_result} | {fp8_result} | **请填写note** | **请填写contributor** |
```

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
        f"FP32: {fp32_result} TFLOPS | FP16: {fp16_result} TFLOPS | BF16: {bf16_result} TFLOPS | FP8: {fp8_result} TFLOPS"
    )
    print("\n🔗 提交数据请点击以下链接：")
    print(f"{issue_url}")
    print(f"\n{'=' * 60}")
    print("💡 提示：")
    print("1. 点击链接会自动填充设备信息和性能数据")
    print("2. 请在issue中填写note（测试环境）和contributor信息")
    print("3. 包含特定关键字的note将被自动归类")


def benchmark_precision(precision, matrix_size, warmup=6, test_iters=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("需要CUDA显卡进行测试")

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
        print_exc()
        if "not implemented" in str(e):
            return None
        raise

    # 预热
    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()

    # 正式测试
    start_time = time.time()
    for _ in range(test_iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    # 计算FLOPS
    flops_per_iter = 2 * matrix_size**3
    total_flops = flops_per_iter * test_iters
    tflops = (total_flops / elapsed) / 1e12

    # 清理显存
    del a, b
    torch.cuda.empty_cache()

    return tflops


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("未检测到CUDA设备")
        exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"测试设备: {device_name}")
    print(
        f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n"
    )
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")

    # 测试不同精度和矩阵大小
    matrix_sizes = [1024, 2048, 4096, 8192, 10240]
    precisions = [
        ("FP32", torch.float32),
        ("FP16", torch.float16),
        ("BF16", torch.bfloat16),
        ("FP8 E4M3FN", torch.float8_e4m3fn),
        # ("INT8", torch.int8),  # 可选：如果需要测试INT8
    ]

    results = {}
    for precision_name, precision in precisions:
        print(f"\n测试 {precision_name}:")
        results[precision_name] = []

        for size in matrix_sizes:
            print(f"测试矩阵大小: {size}x{size}")
            tflops = benchmark_precision(precision, size)
            if tflops is not None:
                results[precision_name].append((size, tflops))
                print(f"  性能: {tflops:.2f} TFLOPS")
            else:
                print("  不支持此精度")
                break

    # 打印总结
    print("\n性能总结:")
    best_results = {}
    for precision_name, measurements in results.items():
        if measurements:
            max_tflops = max(tflops for _, tflops in measurements)
            best_results[precision_name] = max_tflops
            print(f"{precision_name} 最大算力: {max_tflops:.2f} TFLOPS")

    # 生成GitHub issue链接
    generate_github_issue_link(device_name, best_results)
