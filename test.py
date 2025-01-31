import torch
import time

def benchmark_precision(precision, matrix_size=10240, warmup=5, test_iters=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        raise RuntimeError("需要CUDA显卡进行测试")

    # 初始化矩阵
    try:
        a = torch.randn(matrix_size, matrix_size, dtype=precision, device=device)
        b = torch.randn(matrix_size, matrix_size, dtype=precision, device=device)
    except RuntimeError as e:
        if "not implemented" in str(e) and precision == torch.float16:
            print("FP16在该显卡上不可用")
            return 0.0
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

    # 计算FLOPS (矩阵乘法运算量: 2*N^3)
    flops_per_iter = 2 * matrix_size ** 3
    total_flops = flops_per_iter * test_iters
    tflops = (total_flops / elapsed) / 1e12

    # 清理显存
    del a, b
    torch.cuda.empty_cache()
    
    return tflops

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("未检测到CUDA设备")
    else:
        device_name = torch.cuda.get_device_name(0)
        print(f"测试设备: {device_name}\n")

        # FP32测试
        fp32_tflops = benchmark_precision(torch.float32)
        print(f"FP32算力: {fp32_tflops:.2f} TFLOPS\n")

        # FP16测试
        fp16_tflops = benchmark_precision(torch.float16)
        if fp16_tflops > 0:
            print(f"FP16算力: {fp16_tflops:.2f} TFLOPS")

        # BF16测试
        bf16_tflops = benchmark_precision(torch.bfloat16)
        if bf16_tflops > 0:
            print(f"BF16算力: {bf16_tflops:.2f} TFLOPS")