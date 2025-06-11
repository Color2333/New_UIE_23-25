import sys

import cv2
import numpy as np
import torch

print("Python版本:", sys.version)
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda)
print("GPU数量:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU型号:", torch.cuda.get_device_name(0))
print("NumPy版本:", np.__version__)
print("OpenCV版本:", cv2.__version__)
