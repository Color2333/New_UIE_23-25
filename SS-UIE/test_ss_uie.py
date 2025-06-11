#!/usr/bin/env python3
"""
SS-UIE 水下图像增强测试脚本
将Jupyter notebook转换为可执行的Python脚本

使用方法:
python test_ss_uie.py --input_dir ./data/Test_400/input --output_dir ./data/Test_400/output --model_path ./saved_models/SS_UIE.pth

要求:
- 确保已安装所有依赖包（特别是mamba_ssm）
- 有可用的GPU和CUDA环境
- 模型权重文件存在
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
from torchvision.utils import save_image

# 尝试导入所有依赖
try:
    import pytorch_ssim

    PYTORCH_SSIM_AVAILABLE = True
except ImportError:
    PYTORCH_SSIM_AVAILABLE = False

try:
    import mamba_ssm

    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False

try:
    from net.model import SS_UIE_model

    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

try:
    from utils.utils import *
    from utils.LAB import *
    from utils.LCH import *
    from utils.FDL import *

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


def check_dependencies():
    """检查必要的依赖包"""
    missing_deps = []

    if not PYTORCH_SSIM_AVAILABLE:
        missing_deps.append("pytorch_ssim")

    if not MAMBA_SSM_AVAILABLE:
        missing_deps.append("mamba_ssm")

    if not MODEL_AVAILABLE:
        missing_deps.append("SS_UIE_model (检查net/model.py是否存在)")

    if not UTILS_AVAILABLE:
        missing_deps.append("utils modules (检查utils文件夹)")

    if missing_deps:
        print("❌ 缺少以下依赖:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n请先安装缺失的依赖包")
        return False

    print("✅ 所有依赖检查通过")
    return True


def setup_device():
    """设置GPU设备"""
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，请确保安装了GPU版本的PyTorch")
        return None

    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    torch.set_default_tensor_type(torch.FloatTensor)

    print(f"✅ 使用设备: {device}")
    print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"   显存容量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    return device


def load_model(model_path, device):
    """加载SS-UIE模型"""
    try:
        if not MODEL_AVAILABLE:
            raise ImportError("SS_UIE_model 不可用")

        print("🔄 正在加载模型...")

        # 初始化模型
        model = SS_UIE_model(in_channels=3, channels=16, num_resblock=4, num_memblock=4)
        model = model.to(device)

        # 使用DataParallel（如果有多个GPU）
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print(f"   使用 {torch.cuda.device_count()} 个GPU")

        # 加载预训练权重
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()

        print("✅ 模型加载成功")
        return model

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None


def preprocess_image(image_path, target_size=(256, 256)):
    """预处理输入图像"""
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 调整大小
        img = cv2.resize(img, target_size)

        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 转换为tensor
        img = np.array(img).astype(np.float32)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        img = img / 255.0  # 归一化到[0,1]

        return img

    except Exception as e:
        print(f"❌ 图像预处理失败 {image_path}: {e}")
        return None


def process_single_image(model, image_path, output_path, device):
    """处理单张图像"""
    try:
        # 预处理
        input_tensor = preprocess_image(image_path)
        if input_tensor is None:
            return False

        input_tensor = input_tensor.to(device)

        # 推理
        with torch.no_grad():
            output = model(input_tensor)

        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_image(output, output_path, nrow=1, normalize=True)

        return True

    except Exception as e:
        print(f"❌ 处理图像失败 {image_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='SS-UIE 水下图像增强测试')
    parser.add_argument('--input_dir', type=str, default='./data/Test_400/input/',
                        help='输入图像文件夹路径')
    parser.add_argument('--output_dir', type=str, default='./data/Test_400/output/',
                        help='输出图像文件夹路径')
    parser.add_argument('--model_path', type=str, default='./saved_models/SS_UIE.pth',
                        help='模型权重文件路径')
    parser.add_argument('--image_size', type=int, default=256,
                        help='输入图像尺寸 (默认: 256x256)')
    parser.add_argument('--batch_process', action='store_true',
                        help='批量处理模式')

    args = parser.parse_args()

    print("🚀 SS-UIE 水下图像增强测试脚本")
    print("=" * 50)

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 设置设备
    device = setup_device()
    if device is None:
        sys.exit(1)

    # 加载模型
    model = load_model(args.model_path, device)
    if model is None:
        sys.exit(1)

    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取图像列表
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(args.input_dir).glob(f'*{ext}'))
        image_files.extend(Path(args.input_dir).glob(f'*{ext.upper()}'))

    if not image_files:
        print(f"❌ 在 {args.input_dir} 中未找到图像文件")
        sys.exit(1)

    # 按文件名排序
    image_files.sort(key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)

    print(f"📁 找到 {len(image_files)} 张图像")
    print(f"📤 输出目录: {args.output_dir}")
    print("=" * 50)

    # 处理图像
    start_time = time.time()
    success_count = 0

    for i, image_file in enumerate(image_files, 1):
        print(f"🔄 [{i}/{len(image_files)}] 处理: {image_file.name}")

        output_path = os.path.join(args.output_dir, image_file.name)

        if process_single_image(model, str(image_file), output_path, device):
            success_count += 1
            print(f"✅ 完成: {output_path}")
        else:
            print(f"❌ 失败: {image_file.name}")

    # 统计结果
    total_time = time.time() - start_time
    print("=" * 50)
    print(f"✅ 处理完成!")
    print(f"   成功: {success_count}/{len(image_files)} 张图像")
    print(f"   用时: {total_time:.2f} 秒")
    print(f"   平均: {total_time / len(image_files):.2f} 秒/张")

    if success_count < len(image_files):
        print(f"⚠️  有 {len(image_files) - success_count} 张图像处理失败")


if __name__ == "__main__":
    main()
