# GPU矩阵运算性能测试工具

这是一个用于测试GPU矩阵运算性能的基准测试程序，主要测试不同精度下的计算能力。

## 功能特点

- 支持多种数值精度测试（FP32/FP16/BF16）
- 可配置矩阵大小和测试参数
- 自动进行GPU预热，确保测试准确性
- 提供TFLOPS性能指标
- 包含内存自动管理

## 默认参数

- 矩阵大小：10240x10240
- 预热次数：5次
- 测试次数：20次

## 输出结果

- GPU设备信息
- 各精度下的TFLOPS性能指标

## 贡献者
- [zzc0208](https://github.com/zzc0208)
- [KAl(SO₄)₂·12H₂O](https://github.com/CN17161)
- [turning point](https://github.com/colstone)  (算法是他做的)

## 已经测试过的数据

- [database.md](database.md)

您也可以通过提交issue或PR的方式，添加您测试过的数据。

