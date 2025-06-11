#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UIE算法图像对比可视化工具（增强版）
功能: 复现论文中的图像+直方图对比展示，支持ROI放大显示

🎯 主要功能:
- 多算法图像对比展示
- RGB直方图分析
- ROI区域自动放大显示
- 虚线连接原图ROI与放大区域
- 多层次对比图生成
- 批量处理支持

📊 生成的对比图包含:
- 上半部分: 各算法处理后的图像（带ROI标记框和放大区域）
- 中间部分: 对应的RGB三通道直方图
- 下半部分: ROI区域的独立放大对比（可选）

💡 ROI放大功能:
- 自动在图像右侧显示ROI放大区域
- 虚线连接原图ROI与放大区域
- 支持多个ROI同时放大
- 智能避免放大区域重叠

使用示例:
    # 基础用法
    visualizer = UIEComparisonVisualizer(algorithm_folders)
    visualizer.create_advanced_comparison_plot("image.png")

    # 自定义ROI
    visualizer.set_custom_roi([(x1,y1,x2,y2), (x3,y3,x4,y4)])

    # 批量处理
    visualizer.batch_compare_with_zoom(image_list, save_dir="./results")
"""

import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# 设置中文字体和高质量显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150  # 降低DPI避免内存问题
plt.rcParams['savefig.dpi'] = 300

# 检查matplotlib版本
import matplotlib

print(f"Matplotlib版本: {matplotlib.__version__}")


class UIEComparisonVisualizer:
    def __init__(self, image_folders, algorithm_names=None, roi_boxes=None):
        """
        初始化可视化器

        Args:
            image_folders: 字典，{算法名: 图像文件夹路径} 或 路径列表
            algorithm_names: 算法名称列表（如果image_folders是路径列表）
            roi_boxes: ROI区域坐标列表 [(x1,y1,x2,y2), ...]
        """
        if isinstance(image_folders, dict):
            self.algorithm_folders = {name: Path(path) for name, path in image_folders.items()}
        elif isinstance(image_folders, list) and algorithm_names:
            self.algorithm_folders = {}
            for i, folder in enumerate(image_folders):
                name = algorithm_names[i] if i < len(algorithm_names) else f"Algorithm_{i}"
                self.algorithm_folders[name] = Path(folder)
        else:
            raise ValueError("需要提供算法文件夹字典或路径列表+名称列表")

        # 默认ROI框位置（可以自定义）
        self.roi_boxes = roi_boxes or [
            (50, 50, 150, 150),  # 左上角区域
            (200, 100, 300, 200)  # 右侧区域
        ]

        print(f"找到 {len(self.algorithm_folders)} 个算法:")
        for name in self.algorithm_folders.keys():
            print(f"  - {name}")

    def load_image(self, image_path):
        """加载图像并转换为RGB格式"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def add_roi_boxes_with_zoom(self, img, boxes, zoom_positions=None, zoom_size=(80, 80), color=(255, 0, 0),
                                thickness=3):
        """
        在图像上添加ROI标记框并显示放大区域

        Args:
            img: 输入图像
            boxes: ROI区域坐标列表 [(x1,y1,x2,y2), ...]
            zoom_positions: 放大区域显示位置 [(x,y), ...] 如果为None则自动计算
            zoom_size: 放大区域显示大小 (width, height)
            color: 框的颜色
            thickness: 框的粗细
        """
        img_with_zoom = img.copy()
        h, w = img.shape[:2]

        if zoom_positions is None:
            # 自动计算放大区域的显示位置（避免重叠）
            zoom_positions = []
            for i, box in enumerate(boxes):
                if i == 0:
                    # 第一个放大框放在右上角
                    pos_x = w - zoom_size[0] - 20
                    pos_y = 20
                else:
                    # 后续放大框依次向下排列
                    pos_x = w - zoom_size[0] - 20
                    pos_y = 20 + i * (zoom_size[1] + 10)
                zoom_positions.append((pos_x, pos_y))

        for i, (box, zoom_pos) in enumerate(zip(boxes, zoom_positions)):
            x1, y1, x2, y2 = box
            zoom_x, zoom_y = zoom_pos

            # 确保ROI坐标在图像范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # 绘制ROI框
            cv2.rectangle(img_with_zoom, (x1, y1), (x2, y2), color, thickness)

            # 提取ROI区域
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # 调整放大区域尺寸
            roi_resized = cv2.resize(roi, zoom_size, interpolation=cv2.INTER_CUBIC)

            # 确保放大区域位置在图像范围内
            zoom_x = max(0, min(zoom_x, w - zoom_size[0]))
            zoom_y = max(0, min(zoom_y, h - zoom_size[1]))

            # 在图像上绘制放大区域
            img_with_zoom[zoom_y:zoom_y + zoom_size[1], zoom_x:zoom_x + zoom_size[0]] = roi_resized

            # 绘制放大区域的边框
            cv2.rectangle(img_with_zoom,
                          (zoom_x, zoom_y),
                          (zoom_x + zoom_size[0], zoom_y + zoom_size[1]),
                          color, thickness)

            # 绘制连接线（从ROI中心到放大区域）
            roi_center_x = (x1 + x2) // 2
            roi_center_y = (y1 + y2) // 2
            zoom_center_x = zoom_x + zoom_size[0] // 2
            zoom_center_y = zoom_y + zoom_size[1] // 2

            # 绘制虚线连接
            self._draw_dashed_line(img_with_zoom,
                                   (roi_center_x, roi_center_y),
                                   (zoom_center_x, zoom_center_y),
                                   color, thickness=2)

        return img_with_zoom

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=2, dash_length=10):
        """绘制虚线"""
        x1, y1 = pt1
        x2, y2 = pt2

        # 计算线段长度和方向
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx * dx + dy * dy)

        if distance == 0:
            return

        # 单位向量
        ux = dx / distance
        uy = dy / distance

        # 绘制虚线
        current_distance = 0
        while current_distance < distance:
            # 计算当前点
            start_x = int(x1 + current_distance * ux)
            start_y = int(y1 + current_distance * uy)

            # 计算下一个点
            end_distance = min(current_distance + dash_length, distance)
            end_x = int(x1 + end_distance * ux)
            end_y = int(y1 + end_distance * uy)

            # 绘制线段（每隔一段绘制）
            if int(current_distance / dash_length) % 2 == 0:
                cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)

            current_distance += dash_length

    def calculate_histogram(self, img, bins=256):
        """计算RGB三通道直方图"""
        hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256])

        # 归一化
        total_pixels = img.shape[0] * img.shape[1]
        hist_r = hist_r.flatten() / total_pixels * 1000  # 乘以1000便于显示
        hist_g = hist_g.flatten() / total_pixels * 1000
        hist_b = hist_b.flatten() / total_pixels * 1000

        return hist_r, hist_g, hist_b

    def create_comparison_plot(self, image_name, save_path=None, figsize=(20, 8), show_roi=True):
        """
        创建图像对比图

        Args:
            image_name: 要对比的图像文件名
            save_path: 保存路径
            figsize: 图像大小
            show_roi: 是否显示ROI框
        """
        n_algorithms = len(self.algorithm_folders)

        # 创建图像网格布局
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, n_algorithms, height_ratios=[2, 1], hspace=0.3, wspace=0.1)

        # 加载所有图像
        images = {}
        histograms = {}

        for i, (alg_name, folder_path) in enumerate(self.algorithm_folders.items()):
            image_path = folder_path / image_name

            if not image_path.exists():
                print(f"警告: 未找到 {alg_name} 的图像 {image_name}")
                continue

            # 加载图像
            img = self.load_image(image_path)

            # 添加ROI框和放大区域
            if show_roi:
                img_display = self.add_roi_boxes_with_zoom(img, self.roi_boxes,
                                                           zoom_size=(100, 100))
            else:
                img_display = img

            images[alg_name] = img_display

            # 计算直方图
            hist_r, hist_g, hist_b = self.calculate_histogram(img)
            histograms[alg_name] = (hist_r, hist_g, hist_b)

            # 显示图像
            ax_img = fig.add_subplot(gs[0, i])
            ax_img.imshow(img_display)
            ax_img.set_title(alg_name, fontsize=14, fontweight='bold', pad=10)
            ax_img.axis('off')

            # 绘制直方图
            ax_hist = fig.add_subplot(gs[1, i])

            x = np.arange(256)
            ax_hist.plot(x, hist_r, color='red', alpha=0.7, linewidth=1.5, label='R')
            ax_hist.plot(x, hist_g, color='green', alpha=0.7, linewidth=1.5, label='G')
            ax_hist.plot(x, hist_b, color='blue', alpha=0.7, linewidth=1.5, label='B')

            # 设置直方图样式
            ax_hist.set_xlim(0, 255)
            ax_hist.set_ylim(0, max(np.max(hist_r), np.max(hist_g), np.max(hist_b)) * 1.1)

            if i == 0:  # 只在第一个子图显示y轴标签
                ax_hist.set_ylabel('频次 ×10³', fontsize=10)

            ax_hist.set_xlabel('像素值', fontsize=10)
            ax_hist.grid(True, alpha=0.3)

            # 添加RGB柱状图（右上角小图）
            try:
                ax_bar = ax_hist.inset_axes([0.7, 0.6, 0.25, 0.35])
                colors = ['blue', 'green', 'red']
                values = [np.mean(hist_b), np.mean(hist_g), np.mean(hist_r)]
                x_pos = [0, 1, 2]  # 使用数字位置

                bars = ax_bar.bar(x_pos, values, color=colors, alpha=0.8, width=0.6)
                ax_bar.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
                ax_bar.set_xticks(x_pos)
                ax_bar.set_xticklabels(['B', 'G', 'R'])
                ax_bar.tick_params(labelsize=8)

                # 添加数值标签
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax_bar.text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.02,
                                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
            except Exception as e:
                print(f"警告: 无法创建RGB柱状图 - {e}")
                pass

        plt.suptitle(f'UIE算法对比 - {image_name}', fontsize=16, fontweight='bold', y=0.95)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"对比图已保存: {save_path}")

        plt.tight_layout()
        plt.show()

        return fig

    def create_advanced_comparison_plot(self, image_name, save_path=None, figsize=(20, 8),
                                        zoom_size=(100, 100), show_individual_rois=True):
        """
        创建高级对比图，支持多个ROI放大显示

        Args:
            image_name: 要对比的图像文件名
            save_path: 保存路径
            figsize: 图像大小
            zoom_size: 放大区域大小
            show_individual_rois: 是否在下方显示单独的ROI对比
        """
        n_algorithms = len(self.algorithm_folders)
        n_rows = 3 if show_individual_rois and len(self.roi_boxes) > 0 else 2

        # 创建复杂布局
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_rows, n_algorithms,
                      height_ratios=[3, 1, 1] if n_rows == 3 else [3, 1],
                      hspace=0.3, wspace=0.1)

        roi_data = {alg: [] for alg in self.algorithm_folders.keys()}

        for i, (alg_name, folder_path) in enumerate(self.algorithm_folders.items()):
            image_path = folder_path / image_name

            if not image_path.exists():
                print(f"警告: 未找到 {alg_name} 的图像 {image_name}")
                continue

            img = self.load_image(image_path)

            # 创建带放大框的图像
            img_display = self.add_roi_boxes_with_zoom(img, self.roi_boxes,
                                                       zoom_size=zoom_size)

            # 显示主图像
            ax_img = fig.add_subplot(gs[0, i])
            ax_img.imshow(img_display)
            ax_img.set_title(alg_name, fontsize=14, fontweight='bold', pad=10)
            ax_img.axis('off')

            # 计算和显示直方图
            hist_r, hist_g, hist_b = self.calculate_histogram(img)

            ax_hist = fig.add_subplot(gs[1, i])
            x = np.arange(256)
            ax_hist.plot(x, hist_r, color='red', alpha=0.7, linewidth=1.5, label='R')
            ax_hist.plot(x, hist_g, color='green', alpha=0.7, linewidth=1.5, label='G')
            ax_hist.plot(x, hist_b, color='blue', alpha=0.7, linewidth=1.5, label='B')

            ax_hist.set_xlim(0, 255)
            max_hist = max(np.max(hist_r), np.max(hist_g), np.max(hist_b))
            ax_hist.set_ylim(0, max_hist * 1.1 if max_hist > 0 else 1)

            if i == 0:
                ax_hist.set_ylabel('频次 ×10³', fontsize=10)
                ax_hist.legend(loc='upper right', fontsize=8)

            ax_hist.set_xlabel('像素值', fontsize=10)
            ax_hist.grid(True, alpha=0.3)

            # 提取ROI用于下方显示
            h, w = img.shape[:2]
            for box in self.roi_boxes:
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                roi = img[y1:y2, x1:x2]
                if roi.size > 0:
                    roi_data[alg_name].append(roi)

            # 显示第一个ROI的放大版本（如果启用）
            if show_individual_rois and n_rows == 3 and roi_data[alg_name]:
                ax_roi = fig.add_subplot(gs[2, i])
                first_roi = roi_data[alg_name][0]
                # 放大ROI到固定大小以便对比
                roi_enlarged = cv2.resize(first_roi, (120, 120), interpolation=cv2.INTER_CUBIC)
                ax_roi.imshow(roi_enlarged)
                ax_roi.set_title(f'ROI-1 放大', fontsize=10)
                ax_roi.axis('off')

        plt.suptitle(f'UIE算法详细对比 - {image_name}', fontsize=16, fontweight='bold', y=0.96)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"高级对比图已保存: {save_path}")

        plt.tight_layout()
        plt.show()

        return fig, roi_data

    def create_simple_comparison_plot(self, image_name, save_path=None, figsize=(20, 6)):
        """
        创建简化版图像对比图（兼容性更好）

        Args:
            image_name: 要对比的图像文件名
            save_path: 保存路径
            figsize: 图像大小
        """
        n_algorithms = len(self.algorithm_folders)

        # 创建简单布局：只有图像和直方图
        fig, axes = plt.subplots(2, n_algorithms, figsize=figsize,
                                 gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})

        if n_algorithms == 1:
            axes = axes.reshape(-1, 1)

        for i, (alg_name, folder_path) in enumerate(self.algorithm_folders.items()):
            image_path = folder_path / image_name

            if not image_path.exists():
                print(f"警告: 未找到 {alg_name} 的图像 {image_name}")
                # 创建空白图像
                axes[0, i].text(0.5, 0.5, f'{alg_name}\n图像未找到',
                                ha='center', va='center', transform=axes[0, i].transAxes)
                axes[0, i].set_title(alg_name, fontsize=20, fontweight='bold')
                axes[0, i].axis('off')

                axes[1, i].text(0.5, 0.5, '无数据', ha='center', va='center',
                                transform=axes[1, i].transAxes)
                axes[1, i].axis('off')
                continue

            try:
                # 加载和显示图像
                img = self.load_image(image_path)
                img_display = self.add_roi_boxes_with_zoom(img, self.roi_boxes,
                                                           zoom_size=(80, 80))

                axes[0, i].imshow(img_display)
                axes[0, i].set_title(alg_name, fontsize=14, fontweight='bold', pad=10)
                axes[0, i].axis('off')

                # 计算和绘制直方图
                hist_r, hist_g, hist_b = self.calculate_histogram(img)

                x = np.arange(256)
                axes[1, i].plot(x, hist_r, color='red', alpha=0.7, linewidth=1.5, label='R')
                axes[1, i].plot(x, hist_g, color='green', alpha=0.7, linewidth=1.5, label='G')
                axes[1, i].plot(x, hist_b, color='blue', alpha=0.7, linewidth=1.5, label='B')

                axes[1, i].set_xlim(0, 255)
                max_hist = max(np.max(hist_r), np.max(hist_g), np.max(hist_b))
                axes[1, i].set_ylim(0, max_hist * 1.1 if max_hist > 0 else 1)

                if i == 0:
                    axes[1, i].set_ylabel('频次 ×10³', fontsize=10)
                    axes[1, i].legend(loc='upper right', fontsize=8)

                axes[1, i].set_xlabel('像素值', fontsize=10)
                axes[1, i].grid(True, alpha=0.3)

                # 在直方图上添加RGB均值文本
                mean_r, mean_g, mean_b = np.mean(hist_r), np.mean(hist_g), np.mean(hist_b)
                axes[1, i].text(0.98, 0.95, f'R:{mean_r:.1f}\nG:{mean_g:.1f}\nB:{mean_b:.1f}',
                                transform=axes[1, i].transAxes, fontsize=8,
                                verticalalignment='top', horizontalalignment='right',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            except Exception as e:
                print(f"处理 {alg_name} 时出错: {e}")
                axes[0, i].text(0.5, 0.5, f'{alg_name}\n处理出错',
                                ha='center', va='center', transform=axes[0, i].transAxes)
                axes[0, i].axis('off')
                axes[1, i].axis('off')
                continue

        plt.suptitle(f'UIE算法对比 - {image_name}', fontsize=16, fontweight='bold', y=0.95)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"对比图已保存: {save_path}")

        plt.tight_layout()
        plt.show()

        return fig
        """
        创建详细对比图（包含ROI放大图）

        Args:
            image_name: 要对比的图像文件名
            save_path: 保存路径
            figsize: 图像大小
        """
        n_algorithms = len(self.algorithm_folders)
        n_roi = len(self.roi_boxes)

        # 创建复杂布局
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, n_algorithms, height_ratios=[2, 1, 1], hspace=0.4, wspace=0.1)

        roi_crops = {alg: [] for alg in self.algorithm_folders.keys()}

        for i, (alg_name, folder_path) in enumerate(self.algorithm_folders.items()):
            image_path = folder_path / image_name

            if not image_path.exists():
                continue

            img = self.load_image(image_path)
            img_with_boxes = self.add_roi_boxes(img, self.roi_boxes)

            # 主图像
            ax_main = fig.add_subplot(gs[0, i])
            ax_main.imshow(img_with_boxes)
            ax_main.set_title(alg_name, fontsize=14, fontweight='bold')
            ax_main.axis('off')

            # 直方图
            ax_hist = fig.add_subplot(gs[1, i])
            hist_r, hist_g, hist_b = self.calculate_histogram(img)

            x = np.arange(256)
            ax_hist.plot(x, hist_r, 'r-', alpha=0.7, linewidth=1.5)
            ax_hist.plot(x, hist_g, 'g-', alpha=0.7, linewidth=1.5)
            ax_hist.plot(x, hist_b, 'b-', alpha=0.7, linewidth=1.5)
            ax_hist.set_xlim(0, 255)
            ax_hist.grid(True, alpha=0.3)

            if i == 0:
                ax_hist.set_ylabel('频次', fontsize=10)

            # ROI放大图
            ax_roi = fig.add_subplot(gs[2, i])
            if self.roi_boxes:
                # 显示第一个ROI区域的放大图
                x1, y1, x2, y2 = self.roi_boxes[0]
                roi_crop = img[y1:y2, x1:x2]
                ax_roi.imshow(roi_crop)
                ax_roi.set_title(f'ROI放大', fontsize=10)
                ax_roi.axis('off')
                roi_crops[alg_name].append(roi_crop)

        plt.suptitle(f'详细对比分析 - {image_name}', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"详细对比图已保存: {save_path}")

        plt.tight_layout()
        plt.show()

        return fig, roi_crops

    def batch_compare(self, image_list=None, save_dir="./comparison_results"):
        """
        批量生成对比图

        Args:
            image_list: 要处理的图像文件名列表，None则处理第一个文件夹中的所有图像
            save_dir: 保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        if image_list is None:
            # 获取第一个算法文件夹中的所有图像
            first_folder = list(self.algorithm_folders.values())[0]
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_list = [f.name for f in first_folder.iterdir()
                          if f.suffix.lower() in extensions]

        print(f"开始批量处理 {len(image_list)} 张图像...")

        for i, image_name in enumerate(image_list):
            print(f"处理图像 {i + 1}/{len(image_list)}: {image_name}")

            try:
                save_path = save_dir / f"comparison_{image_name.split('.')[0]}.png"
                self.create_comparison_plot(image_name, save_path=save_path)

            except Exception as e:
                print(f"处理 {image_name} 时出错: {e}")
                continue

    def batch_compare_with_zoom(self, image_list=None, save_dir="./zoom_comparison_results",
                                zoom_size=(100, 100)):
        """
        批量生成带放大功能的对比图

        Args:
            image_list: 要处理的图像文件名列表
            save_dir: 保存目录
            zoom_size: 放大区域大小
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        if image_list is None:
            # 获取第一个算法文件夹中的所有图像
            first_folder = list(self.algorithm_folders.values())[0]
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_list = [f.name for f in first_folder.iterdir()
                          if f.suffix.lower() in extensions]

        print(f"开始批量处理 {len(image_list)} 张图像（带ROI放大）...")

        success_count = 0
        for i, image_name in enumerate(image_list):
            print(f"处理图像 {i + 1}/{len(image_list)}: {image_name}")

            try:
                save_path = save_dir / f"zoom_comparison_{image_name.split('.')[0]}.png"
                self.create_advanced_comparison_plot(
                    image_name=image_name,
                    save_path=save_path,
                    zoom_size=zoom_size,
                    show_individual_rois=True
                )
                success_count += 1

            except Exception as e:
                print(f"处理 {image_name} 时出错: {e}")
                # 尝试简化版本
                try:
                    save_path = save_dir / f"simple_comparison_{image_name.split('.')[0]}.png"
                    self.create_simple_comparison_plot(image_name, save_path=save_path)
                    success_count += 1
                except:
                    print(f"简化版本也失败")
                    continue

        print(f"批量处理完成！成功处理 {success_count}/{len(image_list)} 张图像")
        print(f"结果保存在: {save_dir}")

    def set_custom_roi(self, roi_boxes):
        """
        设置自定义ROI区域

        Args:
            roi_boxes: ROI区域坐标列表 [(x1,y1,x2,y2), ...]
        """
        self.roi_boxes = roi_boxes
        print(f"已设置 {len(roi_boxes)} 个ROI区域:")
        for i, box in enumerate(roi_boxes):
            print(f"  ROI-{i + 1}: {box}")

    def interactive_roi_selector(self, sample_image_name):
        """
        交互式ROI选择器（简化版本）
        提供建议的ROI位置
        """
        print(f"\n=== ROI区域选择建议 ===")
        print(f"针对图像: {sample_image_name}")

        # 尝试加载第一个算法的样本图像来获取尺寸
        first_folder = list(self.algorithm_folders.values())[0]
        sample_path = first_folder / sample_image_name

        if sample_path.exists():
            try:
                img = self.load_image(sample_path)
                h, w = img.shape[:2]
                print(f"图像尺寸: {w} x {h}")

                # 提供几种预设的ROI选择
                presets = {
                    "小目标": [(50, 50, 150, 150), (w - 150, 50, w - 50, 150)],
                    "中等目标": [(80, 80, 200, 200), (w - 200, h - 200, w - 80, h - 80)],
                    "细节区域": [(100, 100, 180, 180), (w - 250, 100, w - 170, 180)],
                    "当前默认": self.roi_boxes
                }

                print("\n建议的ROI配置:")
                for name, boxes in presets.items():
                    print(f"{name}: {boxes}")

                print(f"\n当前使用: {self.roi_boxes}")
                print("你可以通过 visualizer.set_custom_roi([(x1,y1,x2,y2), ...]) 来自定义ROI区域")

            except Exception as e:
                print(f"无法分析样本图像: {e}")
        else:
            print(f"样本图像不存在: {sample_path}")


def main():
    """主函数 - 使用示例"""

    # 方式1: 使用字典指定算法和路径
    algorithm_folders = {
        "GT": "./GT",  # Ground Truth图像
        "Input": "./input",  # 原图或输入图像
        "DM": "./output_DM",
        "HAAM-GAN": "./output_GAN",
        "UWnet": "./output_UWnet",
        "U-Transformer": "./output_Utrans",
        "SS-UIE": "./output_SSUIE"
    }

    # 定义ROI区域（可以根据你的图像调整）
    roi_boxes = [
        (80, 80, 180, 180),  # ROI-1: 左上区域
    ]

    # 创建可视化器
    visualizer = UIEComparisonVisualizer(
        image_folders=algorithm_folders,
        roi_boxes=roi_boxes
    )

    # 显示ROI选择建议
    image_name = "00003.png"  # 替换为你要分析的图像文件名
    visualizer.interactive_roi_selector(image_name)

    # 可选：自定义ROI区域
    # 例如：针对鱼类图像的ROI
    custom_rois = [
        (120, 100, 220, 200),  # 鱼头部区域
        (250, 150, 350, 250),  # 鱼身体区域
        (400, 300, 500, 400)  # 背景珊瑚区域
    ]
    # visualizer.set_custom_roi(custom_rois)  # 取消注释来使用自定义ROI

    print("生成对比图...")
    try:
        # 首先尝试创建高级版本（带ROI放大）
        visualizer.create_advanced_comparison_plot(
            image_name=image_name,
            save_path=f"./advanced_comparison_{image_name.split('.')[0]}.png",
            zoom_size=(120, 120),
            show_individual_rois=False
        )
        print("高级版本（带ROI放大）生成成功！")
    except Exception as e:
        print(f"高级版本出错: {e}")
        print("尝试基础版本...")
        try:
            # 回退到基础版本
            visualizer.create_comparison_plot(
                image_name=image_name,
                save_path=f"./comparison_{image_name.split('.')[0]}.png",
                show_roi=True
            )
            print("基础版本生成成功！")
        except Exception as e2:
            print(f"基础版本出错: {e2}")
            print("尝试简化版本...")
            try:
                # 最后回退到简化版本
                visualizer.create_simple_comparison_plot(
                    image_name=image_name,
                    save_path=f"./simple_comparison_{image_name.split('.')[0]}.png"
                )
                print("简化版本生成成功！")
            except Exception as e3:
                print(f"所有版本都失败: {e3}")

    # 可选：批量处理（取消注释来启用）
    # print("开始批量处理...")
    # visualizer.batch_compare_with_zoom(
    #     image_list=["00001.png", "00002.png", "00003.png"],  # 指定要处理的图像
    #     save_dir="./batch_zoom_comparison_results",
    #     zoom_size=(120, 120)  # 放大区域大小
    # )


if __name__ == "__main__":
    # 快速使用示例
    print("=== UIE算法图像对比可视化工具（增强版）===")
    print("\n🎯 新功能:")
    print("✅ ROI区域自动放大显示")
    print("✅ 虚线连接ROI与放大区域")
    print("✅ 多层次对比图生成")
    print("✅ 智能ROI位置建议")
    print("✅ 批量处理支持")
    print("\n📋 使用方法:")
    print("1. 修改 algorithm_folders 字典中的路径")
    print("2. 根据ROI建议调整 roi_boxes 坐标")
    print("3. 设置要分析的图像文件名")
    print("4. 运行脚本获得带放大功能的对比图\n")

    # 如果直接运行，执行示例
    try:
        main()
    except Exception as e:
        print(f"运行示例时出错: {e}")
        print("\n💡 故障排除:")
        print("1. 检查文件路径是否正确")
        print("2. 确保图像文件存在")
        print("3. 根据ROI建议调整坐标")
        print("4. 尝试使用更小的ROI区域")

        # 提供备用的最简单版本
        print("\n🔄 尝试最简版本...")
        try:
            # 最简单的测试
            test_folders = {
                "GT": "./GT",
                "DM": "./output_DM"
            }

            simple_viz = UIEComparisonVisualizer(test_folders, roi_boxes=[(50, 50, 150, 150)])
            simple_viz.create_simple_comparison_plot("00001.png")

        except Exception as e2:
            print(f"最简版本也失败: {e2}")
            print("请检查文件路径和图像文件是否存在。")
