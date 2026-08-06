# GPU 矩阵运算性能测试工具

这是一个用于测试 GPU 矩阵乘（GEMM）吞吐量的基准测试程序，主要测试不同精度下的计算能力。

天梯榜/统计数据：https://perf.svcfusion.com/

## Benchmark v2

- 支持 FP32、TF32、FP16、BF16，以及可选的实验性 FP8
- 仅在 NVIDIA Ampere（计算能力 8.0）或更新架构上测试 TF32
- 在计时前执行精度支持和小矩阵正确性检查
- 预分配输出张量，避免把输出内存分配计入 GEMM 时间
- 自动校准每组迭代次数，每个尺寸默认采样 5 组并报告中位数和 CV
- 在 CUDA/ROCm 上使用设备 Event，在 MPS 上使用单调时钟
- 根据当前可用显存预检矩阵尺寸，并安全跳过不支持、OOM 或失败的项目
- 输出包含 Benchmark 版本、后端和详细采样参数，便于复现

TFLOPS 按方阵乘法 `C = A @ B` 的 `2 × N³` 次浮点运算计算。它反映当前 PyTorch、驱动和硬件组合下的矩阵乘吞吐量，不等于所有实际模型负载的性能。

## 默认参数

- 矩阵大小：1024、2048、4096、8192、10240
- 每个尺寸预热：5 次
- 每个尺寸采样：5 组
- 每组目标计时：0.2 秒，迭代次数自动校准
- 显存上限：当前可用显存的 80%

最终成绩先取每个尺寸的多组中位数，再选择其中最高的尺寸。历史 v1 数据使用“单组测试后直接取最大值”的协议，与 v2 不应视为完全同口径数据。

为保持排行榜口径一致，提交标准成绩时请使用上述默认参数和默认矩阵尺寸。命令行自定义参数适合诊断、兼容性检查和研究，不应与默认 v2 成绩直接混排。

## 输出结果

- GPU 设备信息
- PyTorch、Python、计算后端和 Benchmark 版本
- 每个尺寸的 TFLOPS 中位数、变异系数（CV）、采样组数和迭代数
- 各精度下的最佳稳定 TFLOPS

## 已经测试过的数据

- [database.md](database.md)

您也可以通过提交 PR 的方式，添加您测试过的数据。

## 如何运行

### 1. 克隆仓库
```bash
git clone https://github.com/zzc0721/torch-performance-test-data.git
cd torch-performance-test-data
```

### 2. 安装依赖

使用 uv 管理虚拟环境：
```bash
uv sync
```
Linux/Windows 默认通过项目配置安装 CUDA 12.8 版 PyTorch。也可以根据本机驱动和平台，从 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择匹配的版本。例如：
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

### 3. 运行

使用 uv 运行：
```bash
uv run test.py
```
直接使用 Python 运行：
```bash
python test.py
```
常用选项：

```bash
# 指定第二张卡和矩阵尺寸
uv run test.py --device cuda:1 --sizes 2048 4096 8192

# 增加采样组数和单组目标时长，提高稳定性
uv run test.py --repeats 7 --target-seconds 0.3

# 启用实验性 FP8（不属于榜单标准四列）
uv run test.py --include-fp8
```

查看全部参数：

```bash
uv run test.py --help
```

运行完成后，可以打开程序生成的预填充 Issue 链接，或按照 [database.md](database.md) 的格式提交 PR。提交 v2 结果时请保留 Benchmark 版本和详细采样数据。

## 结果解释

- `TF32 = N/A`：设备不是支持 TF32 的 NVIDIA Ampere 或更新架构；MPS、ROCm 和旧 NVIDIA 不会把第二次 FP32 误标为 TF32。
- `跳过：后端不支持`：矩阵创建或实际 `torch.mm` 不支持该精度。
- `跳过：显存不足`：该尺寸不会中断后续精度测试。
- `CV`：多组 TFLOPS 的样本标准差除以均值。CV 高于 5% 时不建议提交榜单，应先检查温度、功耗、后台负载或虚拟化环境并重新测试。
- FP8 是实验项。不同硬件和 PyTorch 版本可能使用不同接口或缩放策略，因此不进入标准四列排名。

## 开发验证

```bash
python -m unittest discover -s tests -v
python -m py_compile test.py
```

## 贡献者
- [zzc0208](https://github.com/zzc0208)
- [KAl(SO₄)₂·12H₂O](https://github.com/CN17161)
- [turning point](https://github.com/colstone)  (算法是他做的)
