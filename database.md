| device                                       | fp32   | tf32   | fp16    | bf16    | note                                                                                     | contributor                                         |
| -------------------------------------------- | ------ | ------ | ------- | ------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------- |
| NVIDIA B200 180GB                            | 66.24  |        | 1627.64 | 1696.03 | GCP a4-highgpu-8g 实例；Python 3.12 + PyTorch 2.8.0                                      | [zzc0208](https://github.com/zzc0208)               |
| NVIDIA Jeston Thor T5000 128GB               | 6.17   | 63.87  | 110.57  | 108.51  | 实体机；PyTorch 2.10.0dev20251013 + CUDA 13.0                                            | [zzc0208](https://github.com/zzc0208)               |
| NVIDIA Tesla V100S 32GB                      | 13.55  |        | 88.92   | 9.85    | Docker 容器云（参考）；Python 3.10 + PyTorch 2.2.0                                          | [zzc0208](https://github.com/zzc0208)               |
| NVIDIA Tesla T4 16GB                         | 4.17   |        | 41.91   | 2.46    | Google Colab; Python 3.12 + PyTorch 2.8.0                                               | [zzc0208](https://github.com/zzc0208)               |
| NVIDIA A100 40GB                             | 18.11  |        | 247.63  | 246.33  | Docker 容器云（参考）；Python 3.10 + PyTorch 2.2.0                                         | [zzc0208](https://github.com/zzc0208)               |
| NVIDIA RTX 5090 Laptop 24GB                  | 28.58  |        | 95.68   | 97.13   | 实体机；Python 3.10 + PyTorch 2.8.0                                                      | [Charming](https://github.com/aiguoliuguo)          |
| NVIDIA RTX 4090 48GB                         | 54.69  |        | 158.93  | 160.53  | Docker 容器云（参考）；Python 3.10 + PyTorch 2.8.0 + CUDA 12.8                           | [Charming](https://github.com/aiguoliuguo)          |
| NVIDIA RTX 2080 8GB                          | 9.23   |        | 39.03   | 5.14    | Docker 容器云（参考）；Python 3.10 + PyTorch 2.8.0 + CUDA 12.8                           | [Charming](https://github.com/aiguoliuguo)          |
| NVIDIA RTX 3080 Ti 12GB                      | 24.37  |        | 75.56   | 76.47   | Docker 容器云（参考）；Python 3.10 + PyTorch 2.8.0 + CUDA 12.8                           | [Charming](https://github.com/aiguoliuguo)          |
| NVIDIA Tesla P40 24GB                        | 10.07  |        | 10.02   | 5.26    | Docker 容器云（参考）；Python 3.10 + PyTorch 2.8.0 + CUDA 12.8                           | [Charming](https://github.com/aiguoliuguo)          |
| NVIDIA A100 SXM4 80GB                        | 19.18  |        | 258.74  | 264.53  | Docker 容器云（参考）；Python 3.10 + PyTorch 2.8.0 + CUDA 12.8                           | [Charming](https://github.com/aiguoliuguo)          |
| NVIDIA A800 SXM4 80GB                        | 19.08  |        | 266.04  | 266.48  | Docker 容器云（参考）；Python 3.10 + PyTorch 2.8.0 + CUDA 12.8                           | [Charming](https://github.com/aiguoliuguo)          |
| NVIDIA H20 96GB                              | 31.95  |        | 141.73  | 141.86  | Docker 容器云（参考）；Python 3.10 + PyTorch 2.8.0 + CUDA 12.8                           | [Charming](https://github.com/aiguoliuguo)          |
| NVIDIA RTX 3090 24GB                         | 24.46  |        | 75.64   | 76.34   | 实体机；Python 3.13 + PyTorch 2.6.0                                                      | [zzc0208](https://github.com/zzc0208)               |
| NVIDIA RTX 4070 SUPER 12GB                   | 24.52  | 37.19  | 75.88   | 76.10   | 实体机；Python 3.10 + PyTorch 2.5.1                                                      | [zzc0208](https://github.com/zzc0208)               |
| NVIDIA RTX 4060 Laptop 8GB                   | 9.08   |        | 30.82   | 30.87   | 笔记本；Python 3.12 + PyTorch 2.6.0                                                      | [KAl(SO₄)₂·12H₂O](https://github.com/CN17161)       |
| NVIDIA GeForce RTX 5050 Laptop GPU           | 9.56   |        | 26.76   | 27.03   | 笔记本；i7-13650HX；16G D5内存；Python 3.11.13 + PyTorch 2.8.0+cu128                     | [VanillaNahida](https://github.com/VanillaNahida)   |
| NVIDIA RTX 3060 Laptop 6GB                   | 7.83   |        | 25.70   | 25.85   | 笔记本；Python 3.10 + PyTorch 2.5.1                                                      | [turning point](https://github.com/colstone)        |
| NVIDIA RTX 4090 24GB                         | 55.01  |        | 165.08  | 170.06  | 实体机；Python 3.10 + PyTorch 2.4.0                                                      | [Charming](https://github.com/aiguoliuguo)          |
| NVIDIA Tesla M40 12GB                        | 3.61   |        | 2.88    | 1.94    | 实体机；Python 3.12 + PyTorch 2.6.0                                                      | [barryblueice](https://github.com/barryblueice)     |
| NVIDIA RTX 3050 Ti Laptop 4GB                | 5.99   |        | 18.34   | 18.67   | 实体机；Python 3.10 + PyTorch 2.6.0                                                      | [barryblueice](https://github.com/barryblueice)     |
| NVIDIA RTX 4090 24GB                         | 54.77  |        | 171.06  | 173.31  | 实体机；Python 3.10 + PyTorch 2.4.1；Arch Linux                                          | [sd0ric4](https://github.com/sd0ric4)               |
| NVIDIA Tesla P4 8GB                          | 4.96   |        | 4.80    | 2.82    | 实体机；Python 3.12 + PyTorch 2.2.2                                                      | [kaiserKOA](https://github.com/kaiserKOA)           |
| NVIDIA RTX 4090D 48GB                        | 51.41  |        | 155.5   | 151.09  | Docker 容器云（参考）；Python 3.10 + PyTorch 2.6.0                                       | [turning point](https://github.com/colstone)        |
| NVIDIA RTX 4090 24GB                         | 46.48  |        | 162.35  | 162.39  | Docker 容器云（优云智算，参考）；Python 3.10.14 + PyTorch 2.4.0 + CUDA 12.1；显存 23.6GB | [HuanLin](https://github.com/HuanLinOTO)            |
| NVIDIA RTX 5090 32GB                         | 69.16  |        | 224.41  | 236.03  | 智算云扉 5090 实例（参考）；Python 3.10 + PyTorch 2.8.0                                  | [HuanLin](https://github.com/HuanLinOTO)            |
| NVIDIA GeForce RTX 3090 24GB                 | 24.84  |        | 75.88   | 76.68   | 智算云扉 3090 实例（参考）；Python 3.10 + PyTorch 2.8.0                                  | [a-cold-bird](https://github.com/a-cold-bird)       |
| NVIDIA GeForce RTX 4090D 48GB                | 49.05  |        | 149.15  | 145.77  | 智算云扉 4090D 实例（参考）；Python 3.10 + PyTorch 2.8.0；DDR5 显存                      | [a-cold-bird](https://github.com/a-cold-bird)       |
| NVIDIA GeForce RTX 4090 48GB                 | 53.79  |        | 167.27  | 162.43  | 智算云扉 4090 实例（参考）；Python 3.10 + PyTorch 2.8.0；DDR5 显存                       | [a-cold-bird](https://github.com/a-cold-bird)       |
| NVIDIA GeForce RTX 4090D 24GB                | 50.07  |        | 152.64  | 148.91  | 智算云扉 4090 实例（参考）；Python 3.10 + PyTorch 2.8.0                                  | [a-cold-bird](https://github.com/a-cold-bird)       |
| NVIDIA GeForce RTX 4090 24GB                 | 54.67  |        | 167.80  | 163.09  | 智算云扉 4090 实例（参考）；Python 3.10 + PyTorch 2.8.0；DDR5 显存                       | [a-cold-bird](https://github.com/a-cold-bird)       |
| NVIDIA H100 80GB HBM3                        | 51.88  | 402.15 | 760.04  | 797.51  |  Docker 容器云；Python 3.12.3 + PyTorch 2.7.0 + CUDA 12.6                              | [HaxxorCialtion](https://github.com/HaxxorCialtion) |
| NVIDIA RTX PRO 6000 Workstation 96G          | 77.24  |        | 315.51  | 417.98  | 实体机；Python 3.13 + PyTorch 2.8.0                                                      | [AlfreSama](https://github.com/AlfreScarlet)        |
| Ascend 910ProA (Ascend PyTorch)              | 39.83  |        | 110.01  | 1.11    | openi； 仅供参考，910 有硬件向量缓存，基准测试不准                                          | [HuanLinOTO](https://github.com/HuanLinOTO)         |
| Hygon DCU K100_AI                            | 21.75  |        | 88.03   | 90.01   | openi；py31016，torch241                                                                 | [HuanLinOTO](https://github.com/HuanLinOTO)         |
| AMD Radeon RX 7900 XT                        | 23.03  |        | 81.83   | 83.18   | docker; py312+torch280 rocm700(git64359f59)                                              | [cp-yu](https://github.com/cp-yu)                   |
| NVIDIA Tesla T4 16GB                         | 4.43   |        | 42.71   | 2.27    | Kaggle; Python 3.11.13 + PyTorch 2.6.0 + CUDA 12.4                                       | [sxjeru](https://github.com/sxjeru)                 |
| NVIDIA RTX 4070 Laptop 8GB                   | 15.37  |        | 46.30   | 64.41   | 笔记本; Python 3.12.6 + PyTorch 2.8.0 + CUDA 12.6                                        | [sxjeru](https://github.com/sxjeru)                 |
| NVIDIA GeForce RTX 4060 Ti 16GB              | 11.50  |        | 59.38   | 42.65   | 实体机；测自 SVCFusion 整合包                                                            | [HuanLinOTO](https://github.com/HuanLinOTO)         |
| Apple M4 10CPU+10GPU 24GB                    | 2.98   |        | 3.12    | 2.92    | 笔记本；MacBook Air 15.3' 2025 24+512 ; Python 3.12.11 + PyTorch 2.8.0 + MPS             | [zzc0208](https://github.com/zzc0208)               |
| Apple M4 10CPU+10GPU 24GB                    | 2.88   |        | 1.57    | 1.87    | 笔记本；MacBook Air 13.2' 2025 24+512 ; Python 3.13.5 + PyTorch 2.8.0 + MPS              | [sakmist](https://github.com/sakmist)               |
| NVIDIA Tesla T10 16GB                        | 9.46   |        | 59.27   | 5.33    | 实体机；12400 d4内存条; Python 3.12.10 + PyTorch 2.8.0+cu126                             | [sakmist](https://github.com/sakmist)               |
| NVIDIA RTX 5090 32GB                         | 67.57  |        | 237.63  | 242.18  | 实体机；14700k d5内存条; Python 3.11.11 + PyTorch 2.7.1+cu128                            | [sakmist](https://github.com/sakmist)               |
| Apple M3 Max 16CPU+40GPU 60GB                | 11.50  |        | 12.84   | 11.25   | 笔记本                                                                                   | [gouzil](https://github.com/gouzil)                 |
| NVIDIA GeForce RTX 5090 D v2                 | 66.97  |        | 228.57  | 232.22  | 实体机                                                                                   | [richi-shek](https://github.com/richi-shek)         |
| AMD Radeon 780M Graphics                     | 2.80   |        | 4.53    | 4.55    | 笔记本                                                                                   | [TheSmallHanCat](https://github.com/TheSmallHanCat) |
| NVIDIA GeForce RTX 5060 Laptop 8GB           | 12.82  | 18.96  | 38.06   | 38.26   | 笔记本；Python 3.14.0 + PyTorch 2.10.0.dev20251018+cu130                              | [Citrinae-Lime](https://github.com/Citrinae-Lime)   |
| NVIDIA RTX 4070 SUPER 12GB                   | 25.07  |        | 74.73   | 73.63   | 实体机；Python 3.10 + PyTorch 2.8.0 + CUDA 12.8                                          | [RepentStar](https://github.com/RepentStar)         |
| Intel Ultra 7 155H                           | 3.52   |        | 7.70    | 3.76    | 笔记本                                                                                   | [caih-pi-pi](https://github.com/caih-pi-pi)         |
| NVIDIA H100 80GB HBM3                        | 52.11  |        | 758.14  | 783.66  | 实体机                                                                                   | [clow1710](https://github.com/clow1710)             |
| NVIDIA GeForce RTX 4090 48GB                 | 56.01  |        | 172.39  | 167.71  | 实体机                                                                                   | [Mightlaus](https://github.com/Mightlaus)           |
| NVIDIA GeForce RTX 3090                      | 24.79  |        | 75.48   | 76.58   | 实体机                                                                                   | [Mightlaus](https://github.com/Mightlaus)           |
| NVIDIA L40S                                  | 46.88  |        | 264.02  | 269.15  | docker                                                                                   | [Mightlaus](https://github.com/Mightlaus)           |
| NVIDIA RTX PRO 6000 Blackwell Server Edition | 77.01  |        | 312.78  | 409.42  | docker                                                                                   | [Mightlaus](https://github.com/Mightlaus)           |
| NVIDIA H200                                  | 51.34  |        | 759.40  | 790.29  | docker                                                                                   | [Mightlaus](https://github.com/Mightlaus)           |
| NVIDIA RTX A6000                             | 23.92  |        | 120.49  | 123.34  | docker                                                                                   | [Mightlaus](https://github.com/Mightlaus)           |
| NVIDIA GeForce RTX 4070 Ti SUPER 16GB        | 30.25  |        | 89.93   | 89.36   | 实体机                                                                                   | [RaruseReiji](https://github.com/RaruseReiji)       |
| NVIDIA GeForce RTX 3070 8GB                  | 13.69  |        | 43.02   | 41.44   | 实体机                                                                                   | [Nian-Ci](https://github.com/Nian-Ci)               |
| AMD Radeon RX 7900 XTX                       | 24.78  |        | 90.73   | 84.90   | 实体机; Python 3.13 + PyTorch 2.8.0 + ROCm 6.4                                           | [Orion-zhen](https://github.com/Orion-zhen)         |
| NVIDIA GeForce RTX 5060 Laptop GPU           | 13.25  |        | 36.93   | 36.94   | 笔记本；Python 3.13 + PyTorch 2.7.1 + CUDA 12.8                                          | [sjzsd147](https://github.com/sjzsd147)             |
| NVIDIA GeForce GTX 1650 Laptop               | 2.26   |        | 0.34    | 1.63    | 笔记本; Python 3.13 + PyTorch 2.7.1 + CUDA 12.8                                          | [A1phaCaO](https://github.com/A1phaCaO)             |
| NVIDIA GeForce RTX 4090                      | 51.57  |        | 165.94  | 165.71  | 实体机；Python 3.13 + PyTorch 2.7.1 + CUDA 12.8                                          | [CharonlYMY](https://github.com/CharonlYMY)         |
| NVIDIA Tesla V100-SXM2-16GB                  | 14.06  |        | 94.84   | 10.34   | 实体机；Python 3.13 + PyTorch 2.7.1 + CUDA 12.8                                          | [hhhhhge](https://github.com/hhhhhge)               |
| NVIDIA GeForce RTX 3060 Ti                   | 13.83  | 18.23  | 36.47   | 36.50   | 实体机；Python 3.10 + PyTorch 2.8.0 + CUDA 12.9                                          | [sjzsd147](https://github.com/sjzsd147)             |
| NVIDIA Tesla P100 PCIE 16GB                  | 8.63   | 8.63   | 8.39    | 6.11    | Kaggle P100 实例；Python 3.10 + PyTorch 2.6.0 + CUDA 12.4                                | [zzc0208](https://github.com/zzc0208)               |
| NVIDIA Tesla T4 16GB                         | 4.35   | 3.97   | 41.93   | 2.01    | Kaggle T4 x2 实例；Python 3.11 + PyTorch 2.6.0 + CUDA 12.4                               | [zzc0208](https://github.com/zzc0208)               |
| NVIDIA GeForce GTX 1080 Ti                   | 10.09  | 9.96   | 9.62    | 5.38    | 实体机                                                                                   | [ZZDirty](https://github.com/zzccppp)               |
| NVIDIA P106-100                              | 4.46   | 4.36   | 2.30    | N/A     | 实体机 (cpu R7-5700G ES-50)；Python 3.13 + PyTorch 2.7.0 + CUDA 12.8                     | [denouement-alpha](https://github.com/denouement-alpha) |
| AMD Radeon RX 7900 XTX                       | 26.84  | 26.43  | 105.43  | 99.37   | 实体机；PyTorch 2.7.1 + ROCM 6.3 + CPU 5950x                                             | [TheRainstorm](https://github.com/TheRainstorm)     |
| NVIDIA GeForce RTX 3050 Laptop GPU           | 5.16   | 7.71   | 15.17   | 15.24   | 笔记本                                                                                   | [icelly_QAQ](http://www.icelly.xyz)                 |
| NVIDIA GeForce RTX 3060 Ti                   | 13.83  | 18.23  | 36.47   | 36.50   | 实体机；Python 3.10 + PyTorch 2.8.0 + CUDA 12.9                                          | [sjzsd147](https://github.com/sjzsd147)              |
| NVIDIA GeForce RTX 3080 Laptop GPU           | 11.43  | 19.18  | 37.02   | 35.17   | 笔记本; Python 3.13 + PyTorch 2.6.0 + CUDA 12.4                                          | [SPT32422](https://github.com/SPT32422)             |
| MTT S70                                      | 4.01   | N/A    | 0.39    | N/A     | 实体机; Python 3.10 + PyTorch 2.5.0 + MUSA-SDA 4.2.0 + CPU i3-12100                      | [kakaru](https://github.com/KakaruHayate)           |
| NVIDIA GeForce RTX 4080 Laptop GPU           | 19.25  | 32.84  | 54.48   | 49.11   | 笔记本；Python 3.11 + PyTorch 2.8.0 + CUDA 12.9                                          | [ZackZhao233](https://github.com/ZackZhao233)     |
| Apple M4 Max 16CPU+40GPU 64GB                | 13.78  | 13.77  | 14.82   | 13.18   | 笔记本；PyTorch 2.8.0 + MPS                                                              | [Alexw1111](https://github.com/Alexw1111)         |
| NVIDIA GeForce RTX 4060                      | 8.34   | 15.60  | 32.64   | 31.21   | 实体机; PyTorch 2.7.0 + CUDA 12.8                                                        | [ytinyu](https://github.com/ytinyui)                |
| NVIDIA GeForce RTX 5080 Laptop GPU           | 23.26  | 37.80  | 76.15   | 77.11   | 笔记本; PyTorch 2.7.0 + CUDA 12.8                                                        | [Zichen0424](https://github.com/Zichen0424)         |
| AMD Radeon RX 7900 XTX                       | 25.67  | 25.57  | 97.52   | 93.65   | 实体机; PyTorch 2.10.0dev20251007 + ROCM 7.0                                             | [Boneflame](https://github.com/Boneflame)           |
| NVIDIA GeForce RTX 5070 Ti Laptop GPU        | 17.47  | 29.11  | 54.94   | 59.21   | 笔记本; PyTorch 2.8.0 + CUDA 12.9                                                        | [Dehunc](https://github.com/Dehunc)                 |
| NVIDIA GeForce RTX 4070 Ti SUPER             | 31.25  | 46.21  | 92.73   | 92.74   | 实体机; PyTorch 2.2.1                                                                    | [Kalt003](https://github.com/WinHGGG)              |
| NVIDIA GeForce RTX 4070 Ti SUPER *OC         | 33.91  | 50.39  | 99.26   | 99.18   | 实体机; PyTorch 2.8.0 + CUDA 12.8 + OC VRAM@11043MHz Core@3045MHz                        | [Kalt003](https://github.com/WinHGGG)              |
| NVIDIA GeForce RTX 3080                      | 15.56  | 12.63  | 53.41   | 24.94   | 实体机                                                                                   | [xiaoxi68](https://github.com/xiaoxi68)            |
| NVIDIA H20                                   | 31.99  | 71.80  | 142.09  | 142.25  | 派欧云                                                                                   | [xiaoxi68](https://github.com/xiaoxi68)            |
| NVIDIA RTX A2000 12GB                        | 5.51   | 14.42  | 35.99   | 37.26   | 实体机；PyTorch 2.8.0 + CUDA 12.8                                                        | [UsamiOmega](https://github.com/UsamiOmega)        |
| NVIDIA GeForce GTX 1650 Ti                   | 2.52   | 2.58   | 0.39    | 1.67    | 笔记本; PyTorch 2.8.0 + CUDA 12.9                                                        | [Pan4v](https://github.com/Pan4v)                  |
| NVIDIA GeForce RTX 5090                      | 63.99  |        | 220.90  | 221.65  | 实体机；9950X + DDR5 32GB 6400 C32x1 ; 测自 SVCFusion 整合包                               | [TendoArisu](https://github.com/XUANHLGG)          |
| NVIDIA GeForce RTX 4090                      | 53.84  | 85.65  | 168.38  | 164.81  | 派欧云                                                                                   | [xiaoxi68](https://github.com/xiaoxi68)            |
| NVIDIA GeForce RTX 3080 20G                  | 21.45  | 31.93  | 63.53   | 64.71   | 实体机                                                                                   | [2048Nemo](https://github.com/2048Nemo)            |
| MTT S4000                                    | 15.61  | 40.64  | 76.75   | 78.18   | AutoDL                                                                                  | [kakaru](https://github.com/KakaruHayate)          |
| NVIDIA GeForce RTX 5070 Ti                   | 33.08  | 50.45  | 98.70   | 99.06   | 实体机; WSL2 + PyTorch 2.9.0+cu128                                                       | [dayi](https://github.com/rabbit-dayi)             |
| NVIDIA GeForce RTX 5060 Ti                   | 17.43  | 24.06  | 47.87   | 48.24   | 实体机; PyTorch 2.9.0+cu128                                                              | [Alkaid-C](https://github.com/Alkaid-C)            |
| NVIDIA GeForce RTX 5060 Laptop GPU           | 10.62  | 15.53  | 33.16   | 33.33   | 笔记本; PyTorch 2.7.0+cu128                                                              | [Qingchen Jia](https://github.com/QingchenJia)     |
| NVIDIA GeForce GTX 1660 Ti                   | 5.24   | 5.13   | 0.61    | 2.84    | 实体机                                                                                   | [AnteriorTAg127](https://github.com/AnteriorTAg127) |
| NVIDIA TU-AUTO-PROD                          | 7.87   | 56.01  | 4.26    | N/A     | 实体机 (cpu R7-5700G ES-50)                                                              | [denouement-alpha](https://github.com/denouement-alpha) |
| NVIDIA GeForce RTX 5060                      | 13.33  | 19.24  | 39.63   | 40.12   | 实体机; i7-14700KF; Python 3.13.7 + PyTorch 2.7.0 + CUDA 12.8                            | [Do1e](https://github.com/Do1e)                    |
| AMD Radeon RX 7700 XT                        | 8.90   | 8.88   | 55.87   | 51.80   | PyTorch 2.6.0 + ROCM 6.4.2-git76481f7c                                                  | [China-Pony](https://github.com/China-Pony)        |
| NVIDIA GeForce RTX 5090 D                    | 66.20  | 121.23 | 214.77  | 234.39  | 实体机; PyTorch 2.9.0 + CUDA 12.8                                                        | [xxy9983](https://github.com/xxy983)               |
| NVIDIA GeForce RTX 5070 Laptop GPU           | 12.87  | 22.53  | 43.83   | 44.25   | 笔记本; python 3.12.10 + torch 2.7.0 + CUDA 13.1                                         | [pork-beIIy](https://github.com/pork-beIIy)        |
| NVIDIA H100 PCIe                             | 36.95  | 240.43 | 458.03  | 495.93  | 实体机; PyTorch 2.7.0 + CUDA 13.1                                                        | [htdxd](https://github.com/htdxd)                  |
| NVIDIA GeForce RTX 2070                      | 7.90   | 7.85   | 31.48   | 4.42    | 实体机；PyTorch 2.2.0 + CUDA 12.1                                                        | [kakaru](https://github.com/KakaruHayate)          |
| NVIDIA GeForce RTX 5090                      | 70.47  | 122.19 | 234.63  | 240.05  | 优云智算; PyTorch 2.7.0 + CUDA 12.8                                                       | [mfzzf](https://github.com/mfzzf)                 |
| NVIDIA RTX A4000                             | 11.90  | 34.32  | 65.44   | 62.78   | 虚拟机; PyTorch 2.8.0 + CUDA 12.9                                                         | [kakaru](https://github.com/KakaruHayate)         |
| MTT M1000                                    | 1.73   | 3.45   | 7.15    | 7.10    | 笔记本，MDC2025摩尔学院展台，摩尔线程AIBook，长江SoC; PyTorch 2.5.0                            | [kakaru](https://github.com/KakaruHayate)         |
| NVIDIA GeForce RTX 5070                      | 24.11  | 33.78  | 67.44   | 67.19   | PyTorch 2.7.1 + CUDA 12.8                                                                | [lukezzt8](https://github.com/lukezzt8)           |
| AMD Radeon RX 7900 XTX                       | 23.28  | 24.14  | 95.35   | 88.09   | 实体机；PyTorch 2.10.0.a0 + ROCM 7.10.0.a20251120                                         | [vatervato](https://github.com/VaterVato)         |
| NVIDIA GeForce RTX 5080                      | 33.64  | 50.69  | 102.42  | 102.89  | 实体机 | [zhoujiahao111](https://github.com/zhoujiahao111)         |
| NVIDIA GeForce RTX 3070 Laptop GPU | 11.60 | 17.49 | 34.35 | 33.14 | 笔记本 |[yjzxkxdn](https://github.com/yjzxkxdn)  |
| CPU (Intel64 Family 6 Model 154 Stepping 3, GenuineIntel, 12th Gen Intel(R) Core(TM) i7-12700H) | 0.56 | N/A | N/A | N/A | 笔记本| [yjzxkxdn](https://github.com/yjzxkxdn)         |
| NVIDIA L20 | 37.27 | 58.16 | 113.26 | 114.14 | AutoDL |[yjzxkxdn](https://github.com/yjzxkxdn)|
| NVIDIA A100-PCIE-40GB | 17.56 | 108.32 | 241.49 | 234.14 | AutoDL |[yjzxkxdn](https://github.com/yjzxkxdn)|
| Ascend 910B2 | 84.53 | 84.56 | 314.52 | 315.36 | AutoDL; 仅供参考，910 有硬件向量缓存，基准测试不准 | [yjzxkxdn](https://github.com/yjzxkxdn)|
| NVIDIA Jetson Orin Nano | 1.31 | 4.92 | 10.18 | 10.91 | 实体机 |[yjzxkxdn](https://github.com/yjzxkxdn)|
| NVIDIA GeForce MX450 | 2.10 | 2.38 | 0.36 | 1.56 | 笔记本 |[yjzxkxdn](https://github.com/yjzxkxdn)|
| NVIDIA H800 PCIe | 37.77 | 242.83 | 478.23 | 503.83 | AutoDL | [yjzxkxdn](https://github.com/yjzxkxdn)|
| CPU (32 vCPU Intel(R) Xeon(R) Gold 6459C) | 4.87 | 4.89 | 4.65 | 28.43 | AutoDL |[yjzxkxdn](https://github.com/yjzxkxdn)|
| NVIDIA GeForce RTX 3080 10GB | 21.07 | 32.03 | 64.10 | 64.68 | AutoDL | [yjzxkxdn](https://github.com/yjzxkxdn)|
| NVIDIA GeForce RTX 2080 Ti 22GB | 13.04 | 12.90 | 58.58 | 7.28 | AutoDL | [yjzxkxdn](https://github.com/yjzxkxdn)|
| NVIDIA GeForce RTX 5090 Laptop GPU | 28.07 | 44.98 | 89.51 | 83.78 | 笔记本 | [Dalka0064](https://github.com/Dalka0064)|
| NVIDIA GeForce RTX 3080 Ti | 24.90 | 38.15 | 77.83 | 78.34 | 未备注 | [datdanboi25](https://github.com/datdanboi25)|
| NVIDIA GeForce RTX 3070 Ti Laptop GPU | 8.64 | 15.60 | 34.35 | 34.11 |  笔记本 | [datdanboi25](https://github.com/datdanboi25)|
| NVIDIA GeForce RTX 5060 Laptop GPU | 12.15 | 16.64 | 33.89 | 35.27 |  笔记本 |  [LingHe-9639](https://github.com/LingHe-9639)|
| NVIDIA L4 | 15.38 | 31.88 | 68.83 | 70.89 | 实体机 |[Arcannite](https://github.com/Arcannite)|
| AMD Radeon RX 5700 XT [ZLUDA] | 7.15 | 7.06 | 5.87 | 3.52 | 实体机 | [isah1221](https://github.com/isah1221) |
| NVIDIA GeForce RTX 5090 | 69.83 | 122.13 | 223.98 | 236.99 | 智算云扉 | [yjzxkxdn](https://github.com/yjzxkxdn) |
| NVIDIA P104-100 | 5.01 | 5.08 | 4.99 | 2.96 | 实体机 | [llll415](https://github.com/llll415) |
| NVIDIA A10G | 23.37 | 34.71 | 69.47 | 69.47 | 智算云扉 物理机单卡A100 | [a-cold-bird](https://github.com/a-cold-bird) |
| NVIDIA TITAN RTX | 11.91 | 8.79 | 66.32 | 5.01 | 未备注 | [DarkNightGhost0](https://github.com/DarkNightGhost0) |
| NVIDIA GeForce RTX 5060 | 14.12 | 20.49 | 41.92 | 41.87 | 实体机 | [monologue82](https://github.com/monologue82) |
| NVIDIA GeForce RTX 5060 Ti | 15.85 | 22.85 | 45.89 | 46.05 | 未备注 | [S1ntinel](https://github.com/S1ntinel) |
| NVIDIA GeForce RTX 4060 Ti | 11.92 | 21.61 | 43.36 | 44.74 | 实体机 | [Starlight-Elysia](https://github.com/Starlight-Elysia) |
| NVIDIA GeForce RTX 3060 | 8.16 | 12.44 | 24.80 | 25.47 | 实体机 | [JFeiF](https://github.com/JFeiF) |
| NVIDIA GeForce RTX 3080 Ti | 24.38 | 37.75 | 75.75 | 76.05 | 未备注 | [zhenhuangsdu](https://github.com/zhenhuangsdu) |
| NVIDIA GeForce RTX 3060 Ti | 11.57 | 18.06 | 36.54 | 33.00 | 未备注 | [NichijouNano](https://github.com/NichijouNano) |
| NVIDIA GeForce RTX 5060 Ti | 16.37 | 23.59 | 47.51 | 47.41 | 未备注 | [1753617247-maker](https://github.com/1753617247-maker) |
| NVIDIA GeForce RTX 5060 Ti | 16.37 | 23.59 | 47.51 | 47.41 | 实体机 | [1753617247-maker](https://github.com/1753617247-maker) |
| NVIDIA GeForce RTX 5060 Ti | 17.26 | 24.51 | 48.76 | 49.20 | 实体机 | [soul1688](https://github.com/soul1688) |
| NVIDIA GeForce RTX 5080 Laptop GPU | 20.60 | 33.49 | 67.95 | 68.19 | 笔记本 | [Sakuraleiying](https://github.com/Sakuraleiying) |
| AMD Radeon RX 6650 XT | 9.36 | 9.18 | 17.25 | 4.90 | 实体机；Windows 11 | [Pitiedwzr](https://github.com/Pitiedwzr) |
| NVIDIA GeForce RTX 4050 Laptop GPU | 7.99 | 12.62 | 25.49 | 26.36 | 笔记本 | [yamanoko-do](https://github.com/yamanoko-do) |
| Tesla V100-SXM2-16GB | 14.51 | 14.47 | 97.36 | 10.65 | 实体机 | [YiLg8765](https://github.com/YiLg8765) |
| NVIDIA GeForce RTX 5080 Laptop GPU | 23.27 | 36.72 | 74.46 | 78.07 | 未备注 | [NasNeo777](https://github.com/NasNeo777) |
| NVIDIA GeForce RTX 2050 | 4.57 | 6.89 | 12.59 | 13.80 | 实体机 | [TenSin](https://github.com/SkyDream01) |
| NVIDIA GeForce RTX 4060 Laptop GPU | 8.39 | 14.36 | 28.36 | 28.40 | 笔记本 | [NekoLaska](https://github.com/znzsofficial) |
| NVIDIA GeForce RTX 3080 Ti | 25.56 | 38.44 | 77.02 | 77.72 | 未备注 | [datdanboi25](https://github.com/datdanboi25) |
| NVIDIA GeForce RTX 4060 Ti | 15.62 | 23.84 | 47.80 | 47.78 | 实体机 | [yamanoko-do](https://github.com/yamanoko-do) |
| NVIDIA GeForce RTX 5060 Laptop GPU | 12.63 | 18.67 | 38.68 | 38.70 | 笔记本 | [lhzlhz419](https://github.com/lhzlhz419) |
| NVIDIA GeForce RTX 5060 | 13.32 | 20.50 | 41.32 | 41.83 | 实体机 | [NanamiChiaki-7](https://github.com/NanamiChiaki-7) |
| Intel(R) Iris(R) Xe Graphics | 2.02 | 2.07 | 4.45 | 2.08 | 笔记本 i9-13900H Iris Xe Graphics 96EU 2023 | [CarlGao4](https://github.com/CarlGao4) |
| NVIDIA GeForce RTX 5070 | 22.79 | 32.19 | 65.07 | 65.36 | 实体机 | [llightos](https://github.com/llightos) |
| AMD Radeon RX 6800 | 14.08 | 14.05 | 25.39 | 7.59 | 未备注 | [vaselisk96](https://github.com/vaselisk96) |
| NVIDIA Thor | 6.37 | 59.50 | 121.46 | 135.37 | 未备注 | [gpzlx1](https://github.com/gpzlx1) |
| AMD Radeon RX 7900 XT [ZLUDA] | 20.02 | 20.55 | 73.49 | 76.86 | 实体机 | [Genwohuijiangnan](https://github.com/Genwohuijiangnan) |
| K500SM_AI | 17.29 | 24.55 | 67.11 | 66.60 | SCNet | [hydrogen114](https://github.com/hydrogen114) |
| MetaX C500 | 29.80 | 107.57 | 226.28 | 226.69 | GiteeAI | [hydrogen114](https://github.com/hydrogen114) |
| Iluvatar MR-V100 | 23.76 | 22.94 | 75.92 | 82.11 | GiteeAI | [hydrogen114](https://github.com/hydrogen114) |
| AMD Radeon RX 7700 XT | 9.44 | 9.38 | 51.93 | 54.22 | 实体机 | [CarlGao4](https://github.com/CarlGao4) |
| NVIDIA RTX 6000D | 66.17 | 82.19 | 244.99 | 240.84 | 未备注 | [turning point](https://github.com/colstone) |
| NVIDIA GeForce RTX 5070 | 22.92 | 33.22 | 66.28 | 67.45 | 实体机 | [ATTLES123](https://github.com/ATTLES123) |
| NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 74.03 | 183.51 | 390.92 | 368.43 | 未备注 | [yoda1125](https://github.com/yoda1125) |
| NVIDIA RTX PRO 6000 Blackwell Server Edition | 78.99 | 205.23 | 406.97 | 417.34 | 未备注 | [HarryZhang0806](https://github.com/HarryZhang0806) |
| NVIDIA GeForce RTX 5060 Ti | 16.85 | 25.06 | 50.24 | 50.21 | 未备注 | [nnnbdxc](https://github.com/nnnbdxc) |
| Tesla P100-PCIE-16GB | 8.12 | 8.15 | 7.89 | 6.14 | 实体机 | [xxx11-OPS](https://github.com/xxx11-OPS) |
| AMD Radeon RX 9060 XT | 9.39 | 9.34 | 65.59 | 66.14 | 实体机; R5-5600G | [98Fengyu](https://github.com/98Fengyu) |
| NVIDIA GeForce RTX 5090 D | 62.90 | 112.26 | 213.89 | 218.90 | 实体机；GAINWARD RTX 5090D + AMD RYZEN 7950X3D + 96GB DDR5 RAM | [tlljyang](https://github.com/tlljyang) |
| NVIDIA GeForce RTX 3060 Laptop GPU | 7.45 | 11.04 | 22.03 | 21.73 | 笔记本 | [Yalightyear](https://github.com/Yalightyear) |
| NVIDIA GeForce RTX 5050 | 8.74 | 12.22 | 24.53 | 28.37 | 实体机 | [3205323920](https://github.com/3205323920) |
| NVIDIA GeForce RTX 3070 16GB | 13.29 | 18.46 | 37.23 | 38.36 | 实体机；主频2070MHz 显存8000Mhz | [2661540950](https://github.com/2661540950) |
| NVIDIA GeForce RTX 5070 Ti | 28.29 | 43.38 | 83.95 | 85.32 | 未备注 | [chenling-06](https://github.com/chenling-06) |
| AMD Radeon Graphics 9070XT | 15.15 | 14.91 | 129.35 | 133.36 | 实体机 | [bbslxj](https://github.com/bbslxj) |
| Tesla P100-PCIE-16GB | 8.63 | 8.63 | 8.39 | 6.16 | 实体机 | [DQZ618](https://github.com/DQZ618) |
| NVIDIA GeForce RTX 5070 | 23.68 | 33.37 | 67.27 | 66.74 | 未备注 | [hedouchencn-blip](https://github.com/hedouchencn-blip) |
| NVIDIA GeForce RTX 5070 | 22.52 | 30.82 | 62.01 | 62.04 | 未备注 | [knosk123](https://github.com/knosk123) |
| Tesla P100-SXM2-16GB | 9.50 | 9.51 | 8.82 | 5.99 | 未备注 | [rider760](https://github.com/rider760) |
| NVIDIA GeForce RTX 2080 Ti | 11.52 | 11.90 | 56.78 | 6.95 | 实体机 | [lsfdc233](https://github.com/lsfdc233) |
| Intel(R) Arc(TM) A770 Graphics | 15.12 | 14.98 | 80.36 | 85.14 | 实体机；win11 32gb i5-14600kf | [Jesselrj](https://github.com/Jesselrj) |
| NVIDIA GeForce RTX 5060 Laptop GPU | 11.55 | 16.86 | 33.58 | 37.86 | 未备注 | [fishalanp](https://github.com/fishalanp) |
| NVIDIA GeForce RTX 3070 Ti | 17.00 | 22.87 | 45.80 | 45.79 | 未备注 | [YBA0312](https://github.com/YBA0312) |
| NVIDIA A40 | 23.73 | 59.52 | 119.18 | 122.29 | 实体机 | [jimtjames](https://github.com/jimtjames) |
| AMD Radeon RX 6600 XT | 8.35 | 8.38 | 15.61 | 3.07 | 实体机；i5-12400F + 32GB DDR4 RAM (2400MHz) | [musesun727](https://github.com/musesun727) |
| NVIDIA Graphics Device | 12.33 | 79.49 | 163.12 | 175.72 | 实体机 | [zhiyu52](https://github.com/zhiyu52) |
| NVIDIA Graphics Device(CMP170HX) | 12.15 | 77.09 | 166.48 | 177.91 | 未备注 | [bendy2](https://github.com/bendy2) |
| NVIDIA Graphics Device | 12.28 | 80.45 | 162.75 | 174.94 | 未备注 | [litingqu18](https://github.com/litingqu18) |
| NVIDIA Graphics Device | 25.10 | 38.43 | 76.31 | 77.34 | 实体机 | [ilovesouthpark](https://github.com/ilovesouthpark) |
| Apple MPS | 4.28 | 4.36 | 4.75 | 2.30 | 笔记本；Python 3.12.9 + PyTorch 2.7.0 + MPS | [salvere002](https://github.com/salvere002) |
