# GPU矩阵运算性能测试工具

这是一个用于测试GPU矩阵运算性能的基准测试程序，主要测试不同精度下的计算能力。

天梯榜/统计数据：https://perf.svcfusion.com/

## 功能特点

- 支持多种数值精度测试（FP32/FP16/BF16）
- 可配置矩阵大小和测试参数
- 自动进行GPU预热，确保测试准确性
- 提供TFLOPS性能指标
- 包含内存自动管理

## 默认参数

- 矩阵大小：1024x1024~10240x10240
- 预热次数：6次
- 测试次数：30次

## 输出结果

- GPU设备信息
- 各精度下的TFLOPS性能指标

## 已经测试过的数据

- [database.md](database.md)

您也可以通过提交PR的方式，添加您测试过的数据。

## 如何运行

### 1.克隆仓库
```bash
git clone https://github.com/zzc0721/torch-performence-test-data.git
cd torch-performence-test-data
```

### 2.安装依赖
使用uv作为虚拟环境管理，使用以下命令安装：
```bash
uv sync
```
使用最新的torch版本以及cudatoolkit，也可以参考以下方法安装：
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

### 3.运行
使用uv运行
```bash
uv run python test.py
```
直接使用python运行
```bash
python test.py
```
运行完之后如需提交数据可以直接将运行结果发在issue，或是按照 [database.md](database.md) 的格式向本仓库提交PR

## 贡献者
- [zzc0208](https://github.com/zzc0208)
- [KAl(SO₄)₂·12H₂O](https://github.com/CN17161)
- [turning point](https://github.com/colstone)  (算法是他做的)
