import torch

print(torch.backends.cudnn.version())  # cuDNN version
print(torch.backends.cudnn.is_available())  # Check if cuDNN is available

import torch
# import torch_tensorrt

# print(torch.__version__)  # Check PyTorch version
# print(torch_tensorrt.__version__)  # Check Torch-TensorRT version
