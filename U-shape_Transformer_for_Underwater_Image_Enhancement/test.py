#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
U-shape Transformer for Underwater Image Enhancement - 测试脚本
转换自 test.ipynb
"""

import argparse

import cv2
from torch.autograd import Variable
from torchvision.utils import save_image

from loss.LCH import *
# 导入网络和工具
from net.Ushape_Trans import *
from net.utils import *


def split(img):
    """分割图像为不同尺度"""
    output = []
    output.append(F.interpolate(img, scale_factor=0.125))
    output.append(F.interpolate(img, scale_factor=0.25))
    output.append(F.interpolate(img, scale_factor=0.5))
    output.append(img)
    return output


def compute_psnr(img1, img2):
    """计算PSNR值"""
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compute_mse(img1, img2):
    """计算MSE值"""
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    return mse


def setup_environment():
    """设置运行环境"""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.set_default_dtype(torch.float32)


def load_generator(model_path, device='cuda'):
    """加载生成器模型"""
    print(f"正在加载模型: {model_path}")

    # 初始化生成器
    generator = Generator().to(device)

    # 加载模型权重
    try:
        generator.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"安全加载失败: {e}")
        print("尝试传统方式加载...")
        generator.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))
        print("✅ 传统方式加载成功")

    # 设置为评估模式
    generator.eval()
    return generator


def process_image(img_path, generator, device='cuda', target_size=(256, 256)):
    """处理单张图像"""
    # 读取并预处理图像
    imgx = cv2.imread(img_path)
    if imgx is None:
        raise ValueError(f"无法读取图像: {img_path}")

    imgx = cv2.resize(imgx, target_size)
    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx = np.array(imgx).astype('float32')

    # 转换为张量
    imgx = torch.from_numpy(imgx)
    imgx = imgx.permute(2, 0, 1).unsqueeze(0)
    imgx = imgx / 255.0
    imgx = Variable(imgx).to(device)

    # 模型推理
    with torch.no_grad():
        output = generator(imgx)

    return output[3].data


def test_images(input_dir, output_dir, model_path, device='cuda'):
    """测试所有图像"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ 输出目录已创建: {output_dir}")

    # 加载模型
    generator = load_generator(model_path, device)

    # 获取图像列表
    if not os.path.exists(input_dir):
        raise ValueError(f"输入目录不存在: {input_dir}")

    path_list = os.listdir(input_dir)
    path_list.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)

    print(f"找到 {len(path_list)} 张图像")

    # 处理每张图像
    for i, item in enumerate(path_list, 1):
        try:
            print(f"处理第 {i}/{len(path_list)} 张图像: {item}")

            img_path = os.path.join(input_dir, item)
            output_tensor = process_image(img_path, generator, device)

            # 保存结果
            output_path = os.path.join(output_dir, item)
            save_image(output_tensor, output_path, nrow=5, normalize=True)
            print(f"✅ 保存成功: {output_path}")

        except Exception as e:
            print(f"❌ 处理失败 {item}: {e}")

    print(f"🎉 图像处理完成！共处理 {len(path_list)} 张图像")


def evaluate_results(gt_dir, output_dir):
    """评估结果质量"""
    if not os.path.exists(gt_dir):
        print("⚠️ 未找到真值图像目录，跳过评估")
        return

    if not os.path.exists(output_dir):
        print("❌ 输出目录不存在，无法评估")
        return

    path_list = os.listdir(gt_dir)
    path_list.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)

    PSNR = []

    print("开始评估结果质量...")

    for item in path_list:
        gt_path = os.path.join(gt_dir, item)
        output_path = os.path.join(output_dir, item)

        if not os.path.exists(output_path):
            print(f"⚠️ 输出图像不存在: {output_path}")
            continue

        try:
            # 读取图像
            imgx = cv2.imread(gt_path)
            imgy = cv2.imread(output_path)

            if imgx is None or imgy is None:
                print(f"⚠️ 无法读取图像: {item}")
                continue

            # 调整尺寸
            imgx = cv2.resize(imgx, (256, 256))
            imgy = cv2.resize(imgy, (256, 256))

            # 计算PSNR
            psnr1 = compute_psnr(imgx[:, :, 0], imgy[:, :, 0])
            psnr2 = compute_psnr(imgx[:, :, 1], imgy[:, :, 1])
            psnr3 = compute_psnr(imgx[:, :, 2], imgy[:, :, 2])

            psnr = (psnr1 + psnr2 + psnr3) / 3.0
            PSNR.append(psnr)

        except Exception as e:
            print(f"❌ 评估失败 {item}: {e}")

    if PSNR:
        PSNR = np.array(PSNR)
        mean_psnr = PSNR.mean()
        print(f"📊 平均PSNR: {mean_psnr:.2f} dB")
        print(f"📊 PSNR范围: {PSNR.min():.2f} - {PSNR.max():.2f} dB")
    else:
        print("❌ 没有成功评估的图像")


def main():
    parser = argparse.ArgumentParser(description='U-shape Transformer 水下图像增强测试')
    parser.add_argument('--input_dir', type=str, default='./test/input/',
                        help='输入图像目录')
    parser.add_argument('--output_dir', type=str, default='./test/output/',
                        help='输出图像目录')
    parser.add_argument('--gt_dir', type=str, default='./test/GT/',
                        help='真值图像目录（用于评估）')
    parser.add_argument('--model_path', type=str, default='./saved_models/G/generator_795.pth',
                        help='模型权重文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--evaluate', action='store_true',
                        help='是否进行结果评估')

    args = parser.parse_args()

    # 设置环境
    setup_environment()

    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，切换到CPU")
        args.device = 'cpu'

    print(f"使用设备: {args.device}")

    try:
        # 测试图像
        test_images(args.input_dir, args.output_dir, args.model_path, args.device)

        # 评估结果（如果需要）
        if args.evaluate:
            evaluate_results(args.gt_dir, args.output_dir)

    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        return 1

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())
