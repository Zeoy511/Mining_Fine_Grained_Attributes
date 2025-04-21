import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional, List
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torch import Tensor
from matplotlib import cm
import numpy as np
import wrn_model
from model.res12 import Res12
from torchvision.transforms.functional import to_pil_image
img_path = '../test/n07613480/n0761348000000002.jpg'     # 输入图片的路径
save_path = '../cam/CAM1.png'    # 类激活图保存路径
preprocess = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 您的预训练 .pth 文件的路径
path_to_pth_file = '../cam/MiniImageNet-WRN28.pth'

model = wrn_model.wrn28_10(num_classes=200)

# 将模型移动到 GPU 上
model = model.cuda()

# 从您的 .pth 文件中加载状态字典
state_dict = torch.load(path_to_pth_file)

#model.load_state_dict(state_dict)
if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
else:
    state_dict = state_dict

# 打印状态字典的键名
print("Keys in the state_dict:")
for key in state_dict.keys():
    print(key)
print(model)
# 将状态字典加载到模型中
net =model

#net = models.resnet18(pretrained=True).cuda()   # 导入模型
feature_map = []     # 建立列表容器，用于盛放输出特征图

def forward_hook(module, inp, outp):     # 定义hook
    feature_map.append(outp)    # 把输出装入字典feature_map

net.block3.register_forward_hook(forward_hook)    # 对net.layer4这一层注册前向传播
orign_img = Image.open(img_path).convert('RGB')    # 打开图片并转换为RGB模型
img = preprocess(orign_img)     # 图片预处理
img = torch.unsqueeze(img, 0)     # 增加batch维度 [1, 3, 224, 224]

with torch.no_grad():
    out = net(img.cuda())     # 前向传播

cls = torch.argmax(out[1]).item()    # 获取预测类别编码
weights =  net.linear.weight.data[cls,:]    # 获取类别对应的权重
cam = (weights.view(*weights.shape, 1, 1) * feature_map[0].squeeze(0)).sum(0)


def _normalize(cams: Tensor) -> Tensor:
    """CAM normalization"""
    cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
    cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))

    return cams
cam = _normalize(F.relu(cam, inplace=True)).cpu()
mask = to_pil_image(cam.detach().numpy(), mode='F')

def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.6) -> Image.Image:
    """Overlay a colormapped mask on a background image

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError('img and mask arguments need to be PIL.Image')

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError('alpha argument is expected to be of type float between 0 and 1')

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, 1:]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img
result = overlay_mask(orign_img, mask)
result.show()
result.save(save_path)