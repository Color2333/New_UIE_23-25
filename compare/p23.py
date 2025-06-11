#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文风格的UIE算法对比图生成器
复现类似您提供的论文图片效果，包含PSNR值标注和多行对比布局

主要功能:
- 生成论文风格的多行多列对比图
- 自动计算并显示PSNR值
- 支持突出显示"Ours"算法
- 自动调整图像尺寸和布局
- 支持批量生成多组对比图
"""

import json
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

# 设置中文字体和高质量显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class PaperStyleComparison:
    def __init__(self, image_folders, algorithm_names=None, gt_folder=None, input_folder=None):
        """
        初始化论文风格对比图生成器

        Args:
            image_folders: 算法结果文件夹字典或列表
            algorithm_names: 算法名称列表
            gt_folder: Ground Truth文件夹路径
            input_folder: 输入图像文件夹路径
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

        self.gt_folder = Path(gt_folder) if gt_folder else None
        self.input_folder = Path(input_folder) if input_folder else None

        # 加载评估结果（如果存在）
        self.evaluation_results = {}
        self.load_evaluation_results()

        print(f"初始化完成，找到 {len(self.algorithm_folders)} 个算法")

    def load_evaluation_results(self):
        """加载评估结果数据"""
        try:
            # 尝试加载详细结果
            if Path("detailed_results.json").exists():
                with open("detailed_results.json", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.evaluation_results = data
                print("已加载评估结果数据")
            else:
                print("未找到评估结果文件，将使用图像计算PSNR")
        except Exception as e:
            print(f"加载评估结果时出错: {e}")

    def load_image(self, image_path):
        """加载图像并转换为RGB格式"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def calculate_psnr(self, img1, img2):
        """计算两张图像之间的PSNR"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')

        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    def get_psnr_value(self, image_name, algorithm_name):
        """获取PSNR值（优先使用加载的结果，否则计算）"""
        # 移除文件扩展名
        base_name = image_name.split('.')[0] + '.png'

        # 尝试从加载的结果中获取
        if (base_name in self.evaluation_results and
                algorithm_name in self.evaluation_results[base_name]):
            return self.evaluation_results[base_name][algorithm_name]['psnr']

        # 如果没有预加载的结果，尝试计算
        if self.gt_folder:
            try:
                # 加载算法结果图像
                alg_path = self.algorithm_folders[algorithm_name] / image_name
                gt_path = self.gt_folder / image_name

                if alg_path.exists() and gt_path.exists():
                    alg_img = self.load_image(alg_path)
                    gt_img = self.load_image(gt_path)
                    return self.calculate_psnr(alg_img, gt_img)
            except Exception as e:
                print(f"计算PSNR时出错: {e}")

        return None

    def create_paper_style_comparison(self, image_list, save_path=None,
                                      figsize_per_image=(2, 2), highlight_ours=True,
                                      show_input=True, show_gt=True):
        """
        创建论文风格的对比图

        Args:
            image_list: 要对比的图像文件名列表
            save_path: 保存路径
            figsize_per_image: 每个图像的显示大小
            highlight_ours: 是否突出显示"Ours"算法
            show_input: 是否显示输入图像
            show_gt: 是否显示Ground Truth
        """
        n_images = len(image_list)
        n_algorithms = len(self.algorithm_folders)

        # 计算总列数
        n_cols = n_algorithms
        if show_input and self.input_folder:
            n_cols += 1
        if show_gt and self.gt_folder:
            n_cols += 1

        # 创建图像网格
        fig_width = n_cols * figsize_per_image[0]
        fig_height = n_images * figsize_per_image[1] + 1  # 额外空间用于算法名称

        fig, axes = plt.subplots(n_images, n_cols, figsize=(fig_width, fig_height))

        # 确保axes是2D数组
        if n_images == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        # 准备列标题
        col_names = []
        if show_input and self.input_folder:
            col_names.append("Input")

        for alg_name in self.algorithm_folders.keys():
            col_names.append(alg_name)

        if show_gt and self.gt_folder:
            col_names.append("Ground Truth")

        # 为每行图像生成对比
        for row, image_name in enumerate(image_list):
            col = 0

            # 显示输入图像
            if show_input and self.input_folder:
                input_path = self.input_folder / image_name
                if input_path.exists():
                    try:
                        img = self.load_image(input_path)
                        axes[row, col].imshow(img)

                        # 为输入图像也计算PSNR（相对于GT）
                        if self.gt_folder:
                            psnr = self.get_psnr_value(image_name, "Input")
                            if psnr and not np.isinf(psnr):
                                axes[row, col].text(0.05, 0.95, f'PSNR {psnr:.2f}',
                                                    transform=axes[row, col].transAxes,
                                                    bbox=dict(boxstyle='round,pad=0.3',
                                                              facecolor='green', alpha=0.8),
                                                    fontsize=10, fontweight='bold', color='white')
                    except Exception as e:
                        print(f"处理输入图像 {image_name} 时出错: {e}")
                        axes[row, col].text(0.5, 0.5, 'Input\nNot Found',
                                            ha='center', va='center',
                                            transform=axes[row, col].transAxes)
                else:
                    axes[row, col].text(0.5, 0.5, 'Input\nNot Found',
                                        ha='center', va='center',
                                        transform=axes[row, col].transAxes)

                axes[row, col].axis('off')
                col += 1

            # 显示各算法结果
            for alg_name in self.algorithm_folders.keys():
                image_path = self.algorithm_folders[alg_name] / image_name

                if image_path.exists():
                    try:
                        img = self.load_image(image_path)
                        axes[row, col].imshow(img)

                        # 获取PSNR值
                        psnr = self.get_psnr_value(image_name, alg_name)

                        if psnr and not np.isinf(psnr):
                            # 选择颜色（突出显示"Ours"）
                            if highlight_ours and ("ours" in alg_name.lower() or "ss-uie" in alg_name.lower()):
                                bbox_color = 'gold'
                                text_color = 'black'
                                fontweight = 'bold'
                            else:
                                bbox_color = 'gray'
                                text_color = 'white'
                                fontweight = 'normal'

                            axes[row, col].text(0.05, 0.95, f'PSNR {psnr:.2f}',
                                                transform=axes[row, col].transAxes,
                                                bbox=dict(boxstyle='round,pad=0.3',
                                                          facecolor=bbox_color, alpha=0.9),
                                                fontsize=10, fontweight=fontweight,
                                                color=text_color)

                    except Exception as e:
                        print(f"处理 {alg_name} 图像 {image_name} 时出错: {e}")
                        axes[row, col].text(0.5, 0.5, f'{alg_name}\nError',
                                            ha='center', va='center',
                                            transform=axes[row, col].transAxes)
                else:
                    axes[row, col].text(0.5, 0.5, f'{alg_name}\nNot Found',
                                        ha='center', va='center',
                                        transform=axes[row, col].transAxes)

                axes[row, col].axis('off')
                col += 1

            # 显示Ground Truth
            if show_gt and self.gt_folder:
                gt_path = self.gt_folder / image_name
                if gt_path.exists():
                    try:
                        img = self.load_image(gt_path)
                        axes[row, col].imshow(img)
                    except Exception as e:
                        print(f"处理GT图像 {image_name} 时出错: {e}")
                        axes[row, col].text(0.5, 0.5, 'GT\nError',
                                            ha='center', va='center',
                                            transform=axes[row, col].transAxes)
                else:
                    axes[row, col].text(0.5, 0.5, 'GT\nNot Found',
                                        ha='center', va='center',
                                        transform=axes[row, col].transAxes)

                axes[row, col].axis('off')

        # 设置列标题（只在第一行的底部显示）
        for col, col_name in enumerate(col_names):
            if highlight_ours and ("ours" in col_name.lower() or "ss-uie" in col_name.lower()):
                fontweight = 'bold'
                color = 'red'
            else:
                fontweight = 'normal'
                color = 'black'

            axes[0, col].text(0.5, -0.1, col_name,
                              transform=axes[0, col].transAxes,
                              ha='center', va='top', fontsize=12,
                              fontweight=fontweight, color=color)

        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # 为底部标签留出空间

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"论文风格对比图已保存: {save_path}")

        plt.show()
        return fig

    def create_single_row_comparison(self, image_name, save_path=None,
                                     figsize_per_image=(2.5, 2.5), highlight_best=True):
        """
        创建单行对比图（类似您提供的图片样式）

        Args:
            image_name: 图像文件名
            save_path: 保存路径
            figsize_per_image: 每个图像的大小
            highlight_best: 是否突出显示最佳结果
        """
        n_algorithms = len(self.algorithm_folders)

        # 计算总列数
        n_cols = n_algorithms
        if self.input_folder:
            n_cols += 1
        if self.gt_folder:
            n_cols += 1

        # 创建图像
        fig_width = n_cols * figsize_per_image[0]
        fig_height = figsize_per_image[1] + 0.8  # 额外空间用于标签

        fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height))

        if n_cols == 1:
            axes = [axes]

        col = 0
        psnr_values = {}

        # 显示输入图像
        if self.input_folder:
            input_path = self.input_folder / image_name
            if input_path.exists():
                try:
                    img = self.load_image(input_path)
                    axes[col].imshow(img)
                    axes[col].set_title("Input", fontsize=14, fontweight='bold', pad=20)
                except Exception as e:
                    print(f"处理输入图像时出错: {e}")
                    axes[col].text(0.5, 0.5, 'Input\nError', ha='center', va='center',
                                   transform=axes[col].transAxes)
            axes[col].axis('off')
            col += 1

        # 显示各算法结果并收集PSNR值
        for alg_name in self.algorithm_folders.keys():
            image_path = self.algorithm_folders[alg_name] / image_name

            if image_path.exists():
                try:
                    img = self.load_image(image_path)
                    axes[col].imshow(img)

                    # 获取PSNR值
                    psnr = self.get_psnr_value(image_name, alg_name)
                    if psnr and not np.isinf(psnr):
                        psnr_values[alg_name] = psnr

                        # 在图像上显示PSNR
                        axes[col].text(0.05, 0.95, f'PSNR {psnr:.2f}',
                                       transform=axes[col].transAxes,
                                       bbox=dict(boxstyle='round,pad=0.3',
                                                 facecolor='black', alpha=0.7),
                                       fontsize=12, fontweight='bold', color='white')

                    # 设置标题
                    title_color = 'black'
                    title_weight = 'normal'

                    if "ours" in alg_name.lower() or "ss-uie" in alg_name.lower():
                        title_color = 'red'
                        title_weight = 'bold'

                    axes[col].set_title(alg_name, fontsize=14,
                                        fontweight=title_weight, color=title_color, pad=20)

                except Exception as e:
                    print(f"处理 {alg_name} 时出错: {e}")
                    axes[col].text(0.5, 0.5, f'{alg_name}\nError',
                                   ha='center', va='center',
                                   transform=axes[col].transAxes)
            else:
                axes[col].text(0.5, 0.5, f'{alg_name}\nNot Found',
                               ha='center', va='center',
                               transform=axes[col].transAxes)

            axes[col].axis('off')
            col += 1

        # 显示Ground Truth
        if self.gt_folder:
            gt_path = self.gt_folder / image_name
            if gt_path.exists():
                try:
                    img = self.load_image(gt_path)
                    axes[col].imshow(img)
                    axes[col].set_title("Ground Truth", fontsize=14, fontweight='bold', pad=20)
                except Exception as e:
                    print(f"处理GT时出错: {e}")
                    axes[col].text(0.5, 0.5, 'GT\nError', ha='center', va='center',
                                   transform=axes[col].transAxes)
            axes[col].axis('off')

        # 如果需要，突出显示最佳结果
        if highlight_best and psnr_values:
            best_alg = max(psnr_values.keys(), key=lambda k: psnr_values[k])
            print(f"最佳算法: {best_alg} (PSNR: {psnr_values[best_alg]:.2f})")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"单行对比图已保存: {save_path}")

        plt.show()
        return fig, psnr_values

    def batch_create_paper_comparisons(self, image_list=None, save_dir="./paper_comparisons",
                                       images_per_figure=3):
        """
        批量生成论文风格对比图

        Args:
            image_list: 图像列表，None则自动获取
            save_dir: 保存目录
            images_per_figure: 每个图中显示多少张图像
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        if image_list is None:
            # 获取第一个算法文件夹中的所有图像
            first_folder = list(self.algorithm_folders.values())[0]
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_list = [f.name for f in first_folder.iterdir()
                          if f.suffix.lower() in extensions]

        print(f"开始批量生成论文风格对比图，共 {len(image_list)} 张图像...")

        # 分组处理
        for i in range(0, len(image_list), images_per_figure):
            group_images = image_list[i:i + images_per_figure]
            group_name = f"comparison_group_{i // images_per_figure + 1}"

            try:
                save_path = save_dir / f"{group_name}.png"
                self.create_paper_style_comparison(
                    image_list=group_images,
                    save_path=save_path,
                    figsize_per_image=(2, 2)
                )
                print(f"已生成: {group_name}")

            except Exception as e:
                print(f"生成 {group_name} 时出错: {e}")
                continue

        print(f"批量生成完成！结果保存在: {save_dir}")


def main():
    """主函数示例"""

    # 配置算法文件夹（根据您的实际路径修改）
    algorithm_folders = {
        "GT": "./GT",  # Ground Truth图像
        "DM": "./output_DM",
        "HAAM-GAN": "./output_GAN",
        "UWnet": "./output_UWnet",
        "U-Transformer": "./output_Utrans",
        "SS-UIE": "./output_SSUIE"
    }

    # 创建对比图生成器
    comparator = PaperStyleComparison(
        image_folders=algorithm_folders,
        gt_folder="./GT",  # Ground Truth文件夹
        input_folder="./Input"  # 输入图像文件夹
    )

    # 生成单行对比图（类似您提供的图片）
    image_name = "00003.png"  # 修改为您要处理的图像

    print(f"生成单行对比图: {image_name}")
    fig, psnr_values = comparator.create_single_row_comparison(
        image_name=image_name,
        save_path=f"./paper_style_single_{image_name.split('.')[0]}.png",
        figsize_per_image=(2.5, 2.5),
        highlight_best=True
    )

    # 生成多行对比图
    image_list = ["00001.png", "00002.png", "00003.png"]  # 修改为您要对比的图像列表

    print(f"生成多行对比图...")
    fig = comparator.create_paper_style_comparison(
        image_list=image_list,
        save_path="./paper_style_multi_comparison.png",
        figsize_per_image=(2, 2),
        highlight_ours=True
    )

    # 批量生成（可选）
    # print("开始批量生成...")
    # comparator.batch_create_paper_comparisons(
    #     image_list=None,  # None表示处理所有图像
    #     save_dir="./paper_style_batch",
    #     images_per_figure=3
    # )


if __name__ == "__main__":
    print("=== 论文风格UIE算法对比图生成器 ===")
    print("\n📋 功能特点:")
    print("✅ 复现论文风格的对比图布局")
    print("✅ 自动显示PSNR值")
    print("✅ 突出显示'Ours'算法结果")
    print("✅ 支持单行和多行布局")
    print("✅ 批量处理功能")
    print("\n🔧 使用步骤:")
    print("1. 修改 algorithm_folders 中的文件夹路径")
    print("2. 设置 gt_folder 和 input_folder 路径")
    print("3. 指定要处理的图像文件名")
    print("4. 运行生成对比图")
    print("\n💡 注意事项:")
    print("- 确保所有算法的结果图像文件名一致")
    print("- PSNR值会自动从evaluation results加载或计算")
    print("- 'Ours'算法会被自动高亮显示")

    try:
        main()
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("\n🔧 检查清单:")
        print("1. 文件夹路径是否正确")
        print("2. 图像文件是否存在")
        print("3. 文件名是否匹配")
        print("4. evaluation results文件是否存在")
