#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UIE算法评估工具
作者: UIE评估系统
功能: 对多种UIE增强算法进行全面性能评估
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
from datetime import datetime
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class UIEEvaluator:
    def __init__(self, gt_folder, input_folder=None, output_folders=None, save_dir="evaluation_results"):
        """
        初始化UIE评估器

        Args:
            gt_folder: GT原图文件夹路径
            input_folder: 输入图像文件夹路径（可选）
            output_folders: 算法输出文件夹路径列表或字典
            save_dir: 结果保存目录
        """
        self.gt_folder = Path(gt_folder)
        self.input_folder = Path(input_folder) if input_folder else None
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # 处理输出文件夹
        if isinstance(output_folders, dict):
            self.output_folders = {name: Path(path) for name, path in output_folders.items()}
        elif isinstance(output_folders, list):
            self.output_folders = {}
            for folder in output_folders:
                folder_path = Path(folder)
                alg_name = folder_path.name
                self.output_folders[alg_name] = folder_path
        else:
            raise ValueError("output_folders应该是字典或列表")

        # 获取图像文件列表
        self.image_files = self._get_image_files()
        self.results = {}

        print(f"找到 {len(self.image_files)} 张图像")
        print(f"算法数量: {len(self.output_folders)}")
        print(f"算法列表: {list(self.output_folders.keys())}")

    def _get_image_files(self):
        """获取GT文件夹中的所有图像文件"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []

        for file_path in self.gt_folder.iterdir():
            if file_path.suffix.lower() in extensions:
                image_files.append(file_path.name)

        return sorted(image_files)

    def _load_image(self, image_path):
        """加载图像并转换为RGB格式"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def calculate_psnr(self, img1, img2):
        """计算PSNR"""
        if img1.shape != img2.shape:
            print(f"警告: 图像尺寸不匹配 {img1.shape} vs {img2.shape}")
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        if mse == 0:
            return 100
        return 20 * np.log10(255.0 / np.sqrt(mse))

    def calculate_ssim(self, img1, img2):
        """计算SSIM（简化版本）"""
        from skimage.metrics import structural_similarity as ssim

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        return ssim(gray1, gray2, data_range=255)

    def calculate_uciqe(self, img):
        """计算UCIQE（Underwater Color Image Quality Evaluation）"""
        # 转换到Lab颜色空间
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # 分离通道
        L, a, b = cv2.split(lab.astype(np.float64))

        # 计算色度
        chroma = np.sqrt(a ** 2 + b ** 2)

        # 计算饱和度
        saturation = chroma / (L + 1e-8)

        # 计算对比度
        contrast = np.std(L)

        # UCIQE公式（简化版本）
        c1, c2, c3 = 0.4680, 0.2745, 0.2576
        sigma_c = np.std(chroma)
        con_l = contrast
        mu_s = np.mean(saturation)

        uciqe = c1 * sigma_c + c2 * con_l + c3 * mu_s
        return uciqe

    def calculate_uiqm(self, img):
        """计算UIQM（Underwater Image Quality Measure）"""
        # RGB到LAB转换
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        L, a, b = cv2.split(lab.astype(np.float64))

        # 计算UICM (Underwater Image Colorfulness Measure)
        rg = img[:, :, 0].astype(np.float64) - img[:, :, 1].astype(np.float64)
        yb = (img[:, :, 0].astype(np.float64) + img[:, :, 1].astype(np.float64)) / 2 - img[:, :, 2].astype(np.float64)

        rg_std = np.std(rg)
        yb_std = np.std(yb)
        rg_mean = np.mean(rg)
        yb_mean = np.mean(yb)

        uicm = -0.0268 * np.sqrt(rg_std ** 2 + yb_std ** 2) + 0.1586 * np.sqrt(rg_mean ** 2 + yb_mean ** 2)

        # 计算UISM (Underwater Image Sharpness Measure)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        uism = np.mean(sobel ** 2)

        # 计算UIConM (Underwater Image Contrast Measure)
        uiconm = np.std(L)

        # UIQM组合
        c1, c2, c3 = 0.0282, 0.2953, 3.5753
        uiqm = c1 * uicm + c2 * uism + c3 * uiconm
        return uiqm

    def calculate_entropy(self, img):
        """计算图像熵"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        hist = hist[hist > 0]  # 移除零值
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    def calculate_color_distribution(self, img):
        """计算RGB三通道的颜色分布"""
        r_hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        g_hist = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten()
        b_hist = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten()

        return {
            'r_hist': r_hist,
            'g_hist': g_hist,
            'b_hist': b_hist
        }

    def evaluate_single_image(self, image_name):
        """评估单张图像"""
        gt_path = self.gt_folder / image_name
        gt_img = self._load_image(gt_path)

        results = {}

        for alg_name, output_folder in self.output_folders.items():
            output_path = output_folder / image_name

            if not output_path.exists():
                print(f"警告: 未找到 {alg_name} 的输出图像 {image_name}")
                continue

            try:
                output_img = self._load_image(output_path)

                # 计算各种指标
                psnr = self.calculate_psnr(gt_img, output_img)
                ssim = self.calculate_ssim(gt_img, output_img)
                uciqe = self.calculate_uciqe(output_img)
                uiqm = self.calculate_uiqm(output_img)
                entropy = self.calculate_entropy(output_img)

                results[alg_name] = {
                    'psnr': psnr,
                    'ssim': ssim,
                    'uciqe': uciqe,
                    'uiqm': uiqm,
                    'entropy': entropy
                }

            except Exception as e:
                print(f"处理 {alg_name} 的 {image_name} 时出错: {e}")
                continue

        return results

    def run_evaluation(self):
        """运行完整评估"""
        print("开始评估...")

        all_results = {}

        for image_name in tqdm(self.image_files, desc="处理图像"):
            image_results = self.evaluate_single_image(image_name)
            all_results[image_name] = image_results

        # 汇总结果
        self.results = self._aggregate_results(all_results)

        # 保存详细结果
        self._save_detailed_results(all_results)

        print("评估完成!")
        return self.results

    def _aggregate_results(self, all_results):
        """汇总所有结果"""
        aggregated = {}

        for alg_name in self.output_folders.keys():
            metrics = {'psnr': [], 'ssim': [], 'uciqe': [], 'uiqm': [], 'entropy': []}

            for image_name, image_results in all_results.items():
                if alg_name in image_results:
                    for metric, value in image_results[alg_name].items():
                        metrics[metric].append(value)

            # 计算统计信息
            aggregated[alg_name] = {}
            for metric, values in metrics.items():
                if values:
                    aggregated[alg_name][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'values': values
                    }

        return aggregated

    def _save_detailed_results(self, all_results):
        """保存详细结果到JSON和CSV"""

        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # 保存JSON格式
        json_path = self.save_dir / "detailed_results.json"
        converted_results = convert_numpy_types(all_results)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)

        # 保存CSV格式
        rows = []
        for image_name, image_results in all_results.items():
            for alg_name, metrics in image_results.items():
                row = {'image': image_name, 'algorithm': alg_name}
                row.update(metrics)
                rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = self.save_dir / "detailed_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"详细结果已保存到: {json_path} 和 {csv_path}")

    def generate_summary_table(self):
        """生成汇总表格"""
        if not self.results:
            print("请先运行评估")
            return None

        summary_data = []
        for alg_name, metrics in self.results.items():
            row = {'算法': alg_name}
            for metric_name, stats in metrics.items():
                metric_display = {
                    'psnr': 'PSNR',
                    'ssim': 'SSIM',
                    'uciqe': 'UCIQE',
                    'uiqm': 'UIQM',
                    'entropy': '信息熵'
                }.get(metric_name, metric_name.upper())

                row[f'{metric_display}_平均'] = f"{stats['mean']:.4f}"
                row[f'{metric_display}_标准差'] = f"{stats['std']:.4f}"
            summary_data.append(row)

        df = pd.DataFrame(summary_data)

        # 保存表格
        table_path = self.save_dir / "summary_table.csv"
        df.to_csv(table_path, index=False, encoding='utf-8-sig')

        print("汇总表格:")
        print(df.to_string(index=False))
        print(f"\n表格已保存到: {table_path}")

        return df

    def plot_metrics_comparison(self):
        """绘制指标对比图"""
        if not self.results:
            print("请先运行评估")
            return

        metrics = ['psnr', 'ssim', 'uciqe', 'uiqm', 'entropy']
        metric_names = ['PSNR', 'SSIM', 'UCIQE', 'UIQM', '信息熵']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            if i >= len(axes):
                break

            ax = axes[i]

            # 准备数据
            alg_names = []
            means = []
            stds = []

            for alg_name, alg_results in self.results.items():
                if metric in alg_results:
                    alg_names.append(alg_name)
                    means.append(alg_results[metric]['mean'])
                    stds.append(alg_results[metric]['std'])

            if not means:
                continue

            # 绘制柱状图
            bars = ax.bar(range(len(alg_names)), means, yerr=stds,
                          capsize=5, alpha=0.8, color=plt.cm.Set3(np.linspace(0, 1, len(alg_names))))

            ax.set_title(f'{name} 对比', fontsize=14, fontweight='bold')
            ax.set_xlabel('算法', fontsize=12)
            ax.set_ylabel(name, fontsize=12)
            ax.set_xticks(range(len(alg_names)))
            ax.set_xticklabels(alg_names, rotation=45, ha='right')

            # 添加数值标签
            for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + std,
                        f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plot_path = self.save_dir / "metrics_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"指标对比图已保存到: {plot_path}")

    def plot_radar_chart(self):
        """绘制雷达图"""
        if not self.results:
            print("请先运行评估")
            return

        # 选择要显示的指标
        metrics = ['psnr', 'ssim', 'uciqe', 'uiqm', 'entropy']
        metric_labels = ['PSNR', 'SSIM', 'UCIQE', 'UIQM', '信息熵']

        # 准备数据
        algorithms = list(self.results.keys())
        data_matrix = []

        for alg_name in algorithms:
            values = []
            for metric in metrics:
                if metric in self.results[alg_name]:
                    # 归一化到0-1范围
                    all_values = [self.results[alg][metric]['mean']
                                  for alg in algorithms if metric in self.results[alg]]
                    if all_values:
                        min_val, max_val = min(all_values), max(all_values)
                        if max_val > min_val:
                            normalized = (self.results[alg_name][metric]['mean'] - min_val) / (max_val - min_val)
                        else:
                            normalized = 1.0
                        values.append(normalized)
                    else:
                        values.append(0)
                else:
                    values.append(0)
            data_matrix.append(values)

        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 完成圆圈

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))

        for i, (alg_name, values) in enumerate(zip(algorithms, data_matrix)):
            values += values[:1]  # 完成圆圈
            ax.plot(angles, values, 'o-', linewidth=2, label=alg_name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('算法性能雷达图', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.tight_layout()
        radar_path = self.save_dir / "radar_chart.png"
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"雷达图已保存到: {radar_path}")

    def plot_color_distribution(self, sample_image=None):
        """绘制颜色分布对比"""
        if not sample_image:
            sample_image = self.image_files[0]

        gt_path = self.gt_folder / sample_image
        gt_img = self._load_image(gt_path)
        gt_dist = self.calculate_color_distribution(gt_img)

        fig, axes = plt.subplots(len(self.output_folders) + 1, 3, figsize=(15, 4 * (len(self.output_folders) + 1)))
        if len(self.output_folders) == 0:
            axes = axes.reshape(1, -1)

        channels = ['r_hist', 'g_hist', 'b_hist']
        channel_names = ['Red', 'Green', 'Blue']
        colors = ['red', 'green', 'blue']

        # 绘制GT分布
        for j, (channel, name, color) in enumerate(zip(channels, channel_names, colors)):
            axes[0, j].plot(gt_dist[channel], color=color, alpha=0.7, linewidth=2)
            axes[0, j].fill_between(range(256), gt_dist[channel], alpha=0.3, color=color)
            axes[0, j].set_title(f'GT - {name} Channel', fontsize=12, fontweight='bold')
            axes[0, j].set_xlabel('Pixel Value')
            axes[0, j].set_ylabel('Frequency')
            axes[0, j].grid(True, alpha=0.3)

        # 绘制各算法分布
        for i, (alg_name, output_folder) in enumerate(self.output_folders.items()):
            output_path = output_folder / sample_image
            if output_path.exists():
                output_img = self._load_image(output_path)
                output_dist = self.calculate_color_distribution(output_img)

                for j, (channel, name, color) in enumerate(zip(channels, channel_names, colors)):
                    axes[i + 1, j].plot(output_dist[channel], color=color, alpha=0.7, linewidth=2)
                    axes[i + 1, j].fill_between(range(256), output_dist[channel], alpha=0.3, color=color)
                    axes[i + 1, j].set_title(f'{alg_name} - {name} Channel', fontsize=12, fontweight='bold')
                    axes[i + 1, j].set_xlabel('Pixel Value')
                    axes[i + 1, j].set_ylabel('Frequency')
                    axes[i + 1, j].grid(True, alpha=0.3)

        plt.tight_layout()
        color_path = self.save_dir / f"color_distribution_{sample_image.split('.')[0]}.png"
        plt.savefig(color_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"颜色分布图已保存到: {color_path}")

    def create_image_comparison(self, sample_images=None, max_images=5):
        """创建图像对比展示"""
        if not sample_images:
            sample_images = self.image_files[:max_images]

        n_images = len(sample_images)
        n_algorithms = len(self.output_folders)

        # 包括GT和输入图像（如果有）
        n_cols = 2 + n_algorithms if self.input_folder else 1 + n_algorithms

        fig, axes = plt.subplots(n_images, n_cols, figsize=(4 * n_cols, 4 * n_images))
        if n_images == 1:
            axes = axes.reshape(1, -1)

        for i, image_name in enumerate(sample_images):
            col = 0

            # 显示GT
            gt_path = self.gt_folder / image_name
            if gt_path.exists():
                gt_img = self._load_image(gt_path)
                axes[i, col].imshow(gt_img)
                axes[i, col].set_title(f'GT\n{image_name}', fontweight='bold')
                axes[i, col].axis('off')
                col += 1

            # 显示输入图像（如果有）
            if self.input_folder:
                input_path = self.input_folder / image_name
                if input_path.exists():
                    input_img = self._load_image(input_path)
                    axes[i, col].imshow(input_img)
                    axes[i, col].set_title(f'Input\n{image_name}', fontweight='bold')
                    axes[i, col].axis('off')
                    col += 1

            # 显示各算法输出
            for alg_name, output_folder in self.output_folders.items():
                output_path = output_folder / image_name
                if output_path.exists():
                    output_img = self._load_image(output_path)
                    axes[i, col].imshow(output_img)

                    # 添加PSNR信息（如果有）
                    title = alg_name
                    if hasattr(self, 'results') and self.results and alg_name in self.results:
                        if 'psnr' in self.results[alg_name]:
                            # 找到这张图片的PSNR
                            image_idx = self.image_files.index(image_name)
                            if image_idx < len(self.results[alg_name]['psnr']['values']):
                                psnr_val = self.results[alg_name]['psnr']['values'][image_idx]
                                title += f'\nPSNR: {psnr_val:.2f}'

                    axes[i, col].set_title(title, fontweight='bold', fontsize=20)
                    axes[i, col].axis('off')
                    col += 1

        plt.tight_layout()
        comparison_path = self.save_dir / "image_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"图像对比已保存到: {comparison_path}")

    def generate_report(self):
        """生成完整的评估报告"""
        if not self.results:
            print("请先运行评估")
            return

        # 生成汇总表格
        summary_df = self.generate_summary_table()

        # 生成所有图表
        self.plot_metrics_comparison()
        self.plot_radar_chart()
        self.plot_color_distribution()
        self.create_image_comparison()

        # 生成Markdown报告
        report_path = self.save_dir / "evaluation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# UIE算法评估报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**评估图像数量**: {len(self.image_files)}\n\n")
            f.write(f"**参与算法**: {', '.join(self.output_folders.keys())}\n\n")

            f.write("## 评估指标说明\n\n")
            f.write("- **PSNR**: 峰值信噪比，数值越高表示图像质量越好\n")
            f.write("- **SSIM**: 结构相似性指数，数值越接近1表示结构保持越好\n")
            f.write("- **UCIQE**: 水下色彩图像质量评估，针对水下图像的专用指标\n")
            f.write("- **UIQM**: 水下图像质量测量，综合考虑色彩、清晰度和对比度\n")
            f.write("- **信息熵**: 图像信息量的度量，数值越高表示信息越丰富\n\n")

            f.write("## 算法性能汇总\n\n")
            if summary_df is not None:
                f.write(summary_df.to_markdown(index=False))
                f.write("\n\n")

            f.write("## 最佳算法推荐\n\n")

            # 找出各指标的最佳算法
            best_algorithms = {}
            for metric in ['psnr', 'ssim', 'uciqe', 'uiqm', 'entropy']:
                best_score = -float('inf') if metric != 'uiqm' else float('inf')
                best_alg = None

                for alg_name, alg_results in self.results.items():
                    if metric in alg_results:
                        score = alg_results[metric]['mean']
                        if (metric != 'uiqm' and score > best_score) or (metric == 'uiqm' and score < best_score):
                            best_score = score
                            best_alg = alg_name

                if best_alg:
                    best_algorithms[metric] = (best_alg, best_score)

            for metric, (alg, score) in best_algorithms.items():
                metric_name = {'psnr': 'PSNR', 'ssim': 'SSIM', 'uciqe': 'UCIQE',
                               'uiqm': 'UIQM', 'entropy': '信息熵'}[metric]
                f.write(f"- **{metric_name}最佳**: {alg} ({score:.4f})\n")

            f.write("\n## 文件说明\n\n")
            f.write("- `detailed_results.csv`: 每张图像的详细评估结果\n")
            f.write("- `summary_table.csv`: 算法性能汇总表\n")
            f.write("- `metrics_comparison.png`: 指标对比柱状图\n")
            f.write("- `radar_chart.png`: 算法性能雷达图\n")
            f.write("- `color_distribution_*.png`: 颜色分布对比图\n")
            f.write("- `image_comparison.png`: 图像效果对比图\n")

        print(f"\n完整评估报告已生成: {report_path}")
        return report_path


def main():
    """主函数"""

    # 创建评估器
    evaluator = UIEEvaluator(
        gt_folder="./GT",  # GT原图文件夹
        input_folder="./input",  # 输入文件夹（可选）
        output_folders={  # 算法输出文件夹字典
            "DM": "./output_DM",
            "HAAM-GAN": "./output_GAN",
            "UWnet": "./output_UWnet",
            "U-Transformer": "./output_Utrans",
            "SS-UIE": "./output_SSUIE"
        },
        save_dir="./evaluation_results"  # 结果保存目录
    )

    # 运行评估
    results = evaluator.run_evaluation()

    # 生成完整报告
    evaluator.generate_report()

    print("\n评估完成！请查看结果文件夹获取详细报告。")


if __name__ == "__main__":
    main()
