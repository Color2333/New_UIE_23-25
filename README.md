# Underwater Image Enhancement (UIE) Algorithms Comparison 🌊

收集对比了 2023-2025 年比较新的水下图像增强算法的实验项目。

## 📋 项目概述

本项目整合了 5 个主流的水下图像增强算法，并提供了统一的评估框架：

- **DM_underwater**: 基于 Transformer 的扩散模型方法
- **HAAM-GAN**: 混合注意力增强对抗生成网络
- **Shallow-UWnet**: 轻量化的水下图像增强网络
- **SS-UIE**: 基于空间-频谱双域自适应学习的方法
- **U-shape_Transformer**: U 型 Transformer 网络架构

## 🚀 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.11.0+
- 其他依赖见各算法目录下的 `requirements.txt`

### 数据准备

1. 将测试图像放入 `input/` 文件夹
2. 将对应的真值图像放入 `gt/` 文件夹

### 运行算法

#### 1. DM_underwater

```bash
cd DM_underwater
pip install -r requirement.txt
# 下载预训练模型到 experiments_supervised/ 文件夹
python infer.py
```

#### 2. HAAM-GAN

```bash
cd HAAM-GAN
# 将测试图像放入 data/input/ 文件夹
python test.py
# 结果保存在 data/output/ 文件夹
```

#### 3. Shallow-UWnet

```bash
cd Shallow-UWnet
python test.py
```

#### 4. SS-UIE

```bash
cd SS-UIE
python test_ss_uie.py
```

#### 5. U-shape_Transformer

```bash
cd U-shape_Transformer_for_Underwater_Image_Enhancement
python test.py
```

## 📊 性能评估

项目提供了完整的评估框架，包含以下指标：

- **PSNR**: 峰值信噪比
- **SSIM**: 结构相似性指数
- **UCIQE**: 水下色彩图像质量评估
- **UIQM**: 水下图像质量测量
- **信息熵**: 图像信息量度量

运行评估：

```bash
cd compare
python compare.py
```

评估结果将保存在 `compare/evaluation_results/` 文件夹中。

## 📈 实验结果

基于 20 张测试图像的评估结果：

| 算法          | PSNR      | SSIM      | UCIQE     | UIQM       | 信息熵   |
| ------------- | --------- | --------- | --------- | ---------- | -------- |
| DM            | **29.98** | **0.931** | **6.52M** | 329.84     | 7.47     |
| HAAM-GAN      | 21.76     | 0.897     | 32.1K     | 385.67     | **7.56** |
| UWnet         | 19.50     | 0.830     | 869K      | **228.42** | 7.12     |
| U-Transformer | 24.91     | 0.886     | 10.7K     | 307.34     | 7.38     |
| SS-UIE        | 27.54     | 0.923     | 19.85     | 307.02     | 7.35     |

详细结果请参考 `compare/evaluation_results/evaluation_report.md`

## 📚 原文引用

### DM_underwater

```bibtex
@article{dm_underwater,
  title={Underwater Image Enhancement by Transformer-based Diffusion Model with Non-uniform Sampling for Skip Strategy},
  author={作者信息待补充},
  journal={期刊信息待补充},
  year={2023}
}
```

### HAAM-GAN

```bibtex
@article{HAAMGAN,
  title={HAAM-GAN相关论文信息},
  author={作者信息待补充},
  journal={期刊信息待补充},
  year={2023}
}
```

### Shallow-UWnet

```bibtex
@article{shallow_uwnet,
  title={Shallow-UWnet : Compressed Model for Underwater Image Enhancement},
  author={作者信息},
  journal={arXiv preprint arXiv:2101.02073},
  year={2021}
}
```

### SS-UIE

```bibtex
@article{ss_uie,
  title={Adaptive Dual-domain Learning for Underwater Image Enhancement},
  author={作者信息},
  journal={AAAI},
  year={2025}
}
```

### U-shape Transformer

```bibtex
@article{u_transformer,
  title={U-shape Transformer for Underwater Image Enhancement},
  author={作者信息},
  journal={arXiv preprint arXiv:2111.11843},
  year={2021}
}
```

## 📁 项目结构

```
UIE/
├── input/                    # 输入测试图像
├── gt/                      # 真值图像
├── output_*/                # 各算法输出结果
├── compare/                 # 评估比较脚本和结果
├── DM_underwater/           # 扩散模型算法
├── HAAM-GAN/               # GAN-based算法
├── Shallow-UWnet/          # 轻量化网络
├── SS-UIE/                 # 空间-频谱双域方法
└── U-shape_Transformer_for_Underwater_Image_Enhancement/
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目！

## 📄 许可证

本项目遵循各原始算法的开源许可证。具体许可信息请参考各子项目的 LICENSE 文件。

---

如有问题，请提交 Issue 或联系项目维护者。
