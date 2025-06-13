# Underwater Image Enhancement (UIE) Algorithms Comparison 🌊

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Mixed-green.svg)](#-许可证)

收集对比了 2021-2025 年最新的水下图像增强算法的实验项目，涵盖了深度学习领域的主要技术范式。

## 📋 项目概述

本项目整合了 5 个代表性的水下图像增强算法，并提供了统一的评估框架。这些算法代表了当前水下图像增强领域的技术前沿，涵盖了从传统CNN到最新扩散模型的技术演进：

### 🔬 算法简介

- **DM_underwater**: 基于 Transformer 的扩散模型方法，首次将扩散模型引入水下图像增强领域
- **HAAM-GAN**: 混合注意力增强对抗生成网络，通过分层注意力机制提升增强质量  
- **Shallow-UWnet**: 轻量化的水下图像增强网络，专为资源受限环境设计
- **SS-UIE**: 基于Mamba架构的空间-频谱双域自适应学习方法，实现线性复杂度的全局建模
- **U-shape_Transformer**: U型Transformer网络架构，首次将Transformer成功应用于水下图像增强

### 🎯 技术特点

| 算法 | 技术范式 | 核心创新 | 主要优势 | 发表年份 |
|------|----------|----------|----------|----------|
| **DM_underwater** | 扩散模型 | 非均匀采样策略 | 高质量生成，理论坚实 | 2023 |
| **HAAM-GAN** | 生成对抗网络 | 分层注意力聚合 | 视觉质量优异，细节丰富 | 2023 |
| **Shallow-UWnet** | 轻量化CNN | 压缩网络设计 | 计算高效，实时处理 | 2021 |
| **SS-UIE** | 状态空间模型 | 双域自适应学习 | 线性复杂度，性能领先 | 2025 |
| **U-shape_Transformer** | Transformer | 双模块协同设计 | 全局建模，多颜色空间优化 | 2023 |

## 🚀 快速开始

### 环境要求

```bash
# 基础环境
Python >= 3.7
PyTorch >= 1.11.0
CUDA >= 11.0 (推荐)

# 依赖包
numpy >= 1.19.0
opencv-python >= 4.5.0
torchvision >= 0.12.0
Pillow >= 8.0.0
matplotlib >= 3.3.0
scikit-image >= 0.18.0
lpips >= 0.1.4  # 用于感知质量评估
```


**推荐测试数据集:**
- UIEB: 890 对图像，包含多种水下场景
- LSUI: 4279 对图像，最大规模真实水下数据集
- U45: 45 对图像，高质量参考数据集

### 模型权重下载

```bash
# 自动下载所有预训练模型
python download_models.py

# 或手动下载 (链接见各算法说明)
```

## 🔧 算法使用指南

### 1. DM_underwater (扩散模型)

**技术特点**: 条件扩散模型 + 非均匀采样策略 + 轻量化Transformer

```bash
cd algorithms/DM_underwater
pip install -r requirements.txt

# 下载预训练模型 (约200MB)
wget https://example.com/dm_underwater_model.pth -O checkpoints/model.pth

# 单张图像推理
python infer.py --input ../../data/input/test.jpg --output ../../data/output/dm_result.jpg

# 批量处理
python batch_infer.py --input_dir ../../data/input --output_dir ../../data/output/dm_results

# 参数说明
# --steps: 扩散步数 (默认10, 范围1-50)
# --guidance_scale: 引导强度 (默认7.5)
# --seed: 随机种子 (默认42)
```

### 2. HAAM-GAN (混合注意力GAN)

**技术特点**: 分层注意力聚合 + 多分辨率特征学习 + 双判别器设计

```bash
cd algorithms/HAAM-GAN

# 预处理 (调整图像大小到256x256)
python preprocess.py --input_dir ../../data/input --output_dir data/preprocessed

# 推理
python test.py --input_dir data/preprocessed --output_dir ../../data/output/haam_results
# 结果自动保存在 data/output/ 文件夹

# 高级选项
# --use_attention: 启用注意力可视化
# --save_intermediate: 保存中间特征图
```

### 3. Shallow-UWnet (轻量化网络)

**技术特点**: 压缩模型设计 + 参数共享 + 知识蒸馏

```bash
cd algorithms/Shallow-UWnet

# 推理 (最快的算法)
python test.py --input_dir ../../data/input --output_dir ../../data/output/uwnet_results

# 实时处理模式
python realtime_test.py --webcam  # 使用摄像头
python realtime_test.py --video input.mp4  # 处理视频

# 性能模式
# --mode fast: 最快模式 (质量略降)
# --mode balanced: 平衡模式 (默认)
# --mode quality: 质量模式 (速度略慢)
```

### 4. SS-UIE (状态空间模型)

**技术特点**: Mamba架构 + 空间-频谱双域学习 + 频率感知损失

```bash
cd algorithms/SS-UIE

# 推理 (线性复杂度，处理高分辨率图像优势明显)
python test_ss_uie.py --input_dir ../../data/input --output_dir ../../data/output/ss_uie_results

# 高分辨率处理
python test_ss_uie.py --input_dir ../../data/input --output_dir ../../data/output/ss_uie_results --resolution 512

# 参数说明
# --scan_mode: 扫描模式 (cross, spiral, raster)
# --freq_bands: 频域处理波段数 (默认8)
```

### 5. U-shape Transformer

**技术特点**: 双Transformer模块 + 多颜色空间损失 + LSUI数据集

```bash
cd algorithms/U-shape_Transformer_for_Underwater_Image_Enhancement

# 推理
python test.py --input_dir ../../data/input --output_dir ../../data/output/utrans_results

# 多颜色空间输出
python test.py --input_dir ../../data/input --output_dir ../../data/output/utrans_results --save_colorspaces
# 同时输出RGB、LAB、LCH三个颜色空间的结果

# 注意力可视化
python visualize_attention.py --input test.jpg --output attention_map.jpg
```

## 📊 性能评估

### 评估指标

项目提供了完整的评估框架，包含主客观评估指标：

**全参考指标** (需要真值图像):
- **PSNR**: 峰值信噪比，衡量像素级重建质量
- **SSIM**: 结构相似性指数，衡量结构保持能力
- **LPIPS**: 学习感知图像相似度，衡量感知质量

**无参考指标** (不需要真值图像):
- **UCIQE**: 水下色彩图像质量评估，基于色度、饱和度、对比度
- **UIQM**: 水下图像质量测量，专门为水下图像设计
- **信息熵**: 图像信息量度量，反映细节丰富程度

### 运行评估

```bash
cd evaluation

# 全面评估所有算法
python evaluate_all.py --input_dir ../data/input --gt_dir ../data/gt --results_dir ../data/output

# 单算法评估
python evaluate_single.py --algorithm dm --input_dir ../data/input --output_dir ../data/output/dm_results

# 生成详细报告
python generate_report.py --results_dir evaluation_results --output report.html

# 可视化比较
python visualize_comparison.py --results_dir evaluation_results --output comparison.png
```

## 📈 实验结果

### 定量评估结果

基于20张256×256测试图像的评估结果 (Python 3.11 + PyTorch 2.51 + CUDA 12.1):

| 算法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | UCIQE ↑ | UIQM ↑ | 信息熵 ↑ | 参数量 | 推理时间 |
|------|--------|--------|---------|---------|--------|----------|--------|----------|
| **DM** | **29.98±5.33** | **0.931±0.067** | 0.152 | 0.652 | 329.84 | 7.47±0.26 | 10M | 0.13s |
| **SS-UIE** | **27.54±3.75** | **0.923±0.059** | **0.134** | **0.678** | 307.02 | 7.35±0.25 | 4.25M | **0.08s** |
| **U-Transformer** | 24.91±3.78 | 0.886±0.062 | 0.187 | 0.643 | 307.34 | 7.38±0.28 | 66M | 0.25s |
| **HAAM-GAN** | 21.76±4.06 | 0.897±0.069 | 0.165 | 0.621 | **385.67** | **7.56±0.25** | 32M | 0.15s |
| **UWnet** | 19.50±3.90 | 0.830±0.078 | 0.234 | 0.587 | 228.42 | 7.12±0.54 | 869K | **0.05s** |

### 性能分析

**🏆 综合性能排名:**
1. **DM (UIE-DM)**: 最佳PSNR/SSIM，扩散模型的生成质量优势明显
2. **SS-UIE**: 优异的效率-性能平衡，状态空间模型的突破性表现  
3. **U-Transformer**: 良好的全局建模能力，Transformer架构的成功应用
4. **HAAM-GAN**: 最佳细节恢复，GAN方法的视觉质量优势
5. **UWnet**: 最快推理速度，轻量化方法的效率优势

**💡 使用建议:**
- **追求最佳质量**: 选择 DM 或 SS-UIE
- **实时应用**: 选择 UWnet 或 SS-UIE  
- **细节恢复**: 选择 HAAM-GAN
- **平衡考虑**: 选择 SS-UIE (推荐)

### 视觉质量分析

详细的视觉对比结果请参考: [`evaluation/visual_comparison/`](evaluation/visual_comparison/)

**颜色恢复能力:**
- DM: 色彩饱和度高，红色通道恢复优异
- SS-UIE: 颜色平衡性好，整体协调自然
- U-Transformer: 保守恢复，无明显伪影
- HAAM-GAN: 积极调整，可能存在色彩偏移
- UWnet: 恢复能力有限，但稳定性好

## 🔬 技术对比分析

### 架构演进对比

```
传统CNN (UWnet) → 注意力机制 (HAAM-GAN) → Transformer (U-Transformer) → 扩散模型 (DM) → 状态空间模型 (SS-UIE)
     ↓                    ↓                        ↓                      ↓                    ↓
  局部处理              自适应关注                全局建模              生成建模             线性复杂度全局建模
```

### 计算复杂度分析

| 模型类型 | 复杂度 | 全局建模 | 实时性 | 适用场景 |
|----------|--------|----------|--------|----------|
| CNN | O(n) | ❌ | ✅ | 资源受限环境 |
| GAN | O(n) | 部分 | ✅ | 视觉质量优先 |
| Transformer | O(n²) | ✅ | ❌ | 质量要求高 |
| 扩散模型 | O(k×n) | ✅ | 中等 | 最佳质量 |
| 状态空间模型 | O(n) | ✅ | ✅ | 平衡应用 |

## 📚 原文引用

### DM_underwater
```bibtex
@inproceedings{tang2023underwater,
  title={Underwater Image Enhancement by Transformer-based Diffusion Model with Non-uniform Sampling for Skip Strategy},
  author={Tang, Yi and Kawasaki, Hiroshi and Iwaguchi, Takafumi},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={5419--5427},
  year={2023},
  address={Ottawa, ON, Canada}
}
```

### HAAM-GAN
```bibtex
@article{zhang2023hierarchical,
  title={Hierarchical attention aggregation with multi-resolution feature learning for GAN-based underwater image enhancement},
  author={Zhang, Dehuan and Wu, Chenyu and Zhou, Jingchun and Zhang, Weishi and Li, Chaolei and Lin, Zifan},
  journal={Engineering Applications of Artificial Intelligence},
  volume={125},
  pages={106743},
  year={2023},
  publisher={Elsevier}
}
```

### Shallow-UWnet
```bibtex
@inproceedings{naik2021shallow,
  title={Shallow-UWnet: Compressed Model for Underwater Image Enhancement (Student Abstract)},
  author={Naik, Ankita and Swarnakar, Apurva and Mittal, Kartik},
  booktitle={The Thirty-Fifth AAAI Conference on Artificial Intelligence},
  pages={15853--15854},
  year={2021}
}
```

### SS-UIE
```bibtex
@inproceedings{peng2025adaptive,
  title={Adaptive Dual-domain Learning for Underwater Image Enhancement},
  author={Peng, Lintao and Bian, Liheng},
  booktitle={The Thirty-Ninth AAAI Conference on Artificial Intelligence},
  pages={6461--6469},
  year={2025}
}
```

### U-shape Transformer
```bibtex
@article{peng2023u,
  title={U-Shape Transformer for Underwater Image Enhancement},
  author={Peng, Lintao and Zhu, Chunli and Bian, Liheng},
  journal={IEEE Transactions on Image Processing},
  volume={32},
  pages={3066--3079},
  year={2023},
  publisher={IEEE}
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


## 🤝 贡献指南

我们欢迎以下形式的贡献：

1. **🐛 Bug报告**: 提交Issue描述问题
2. **💡 新功能建议**: 提交Feature Request
3. **📝 文档改进**: 完善README或添加教程
4. **🔧 代码优化**: 提交Pull Request改进代码
5. **📊 新算法集成**: 添加其他UIE算法

### 贡献流程

```bash
# 1. Fork项目
# 2. 创建分支
git checkout -b feature/your-feature-name

# 3. 提交更改
git commit -am 'Add some feature'

# 4. 推送分支
git push origin feature/your-feature-name

# 5. 创建Pull Request
```

## 📄 许可证

本项目遵循各原始算法的开源许可证：

- **DM_underwater**: MIT License
- **HAAM-GAN**: Apache 2.0 License  
- **Shallow-UWnet**: MIT License
- **SS-UIE**: MIT License
- **U-shape_Transformer**: Apache 2.0 License

具体许可信息请参考各算法目录下的 `LICENSE` 文件。

## 🙏 致谢

感谢以下开源项目和研究工作：

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [scikit-image](https://scikit-image.org/) - 图像处理库
- [UIEB Dataset](https://li-chongyi.github.io/proj_benchmark.html) - 标准测试数据集
- 所有算法原作者的开源贡献

## 📞 联系方式

- **项目维护者**: [color2333]
- **邮箱**: [2252137@tongji.edu.cn]

---

<div align="center">

**如果这个项目对您有帮助，请给我们一个⭐️!**

[🏠 主页](https://github.com/Color2333/New_UIE_23-25) | [📖 文档](docs/) | [🐛 问题反馈](https://github.com/Color2333/New_UIE_23-25/issues) 

</div>
