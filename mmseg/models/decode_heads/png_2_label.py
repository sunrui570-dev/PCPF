import torch
import numpy as np
from PIL import Image

# 定义 Cityscapes 的调色板（GTA 使用相同的风格）
cityspallete = [
    128, 64, 128,     # road
    244, 35, 232,     # sidewalk
    70, 70, 70,       # building
    102, 102, 156,    # wall
    190, 153, 153,    # fence
    153, 153, 153,    # pole
    250, 170, 30,     # traffic light
    220, 220, 0,      # traffic sign
    107, 142, 35,     # vegetation
    152, 251, 152,    # terrain
    0, 130, 180,      # sky
    220, 20, 60,      # person
    255, 0, 0,        # rider
    0, 0, 142,        # car
    0, 0, 70,         # truck
    0, 60, 100,       # bus
    0, 80, 100,       # train
    0, 0, 230,        # motorcycle
    119, 11, 32,      # bicycle
    255, 255, 255     # unknown (可忽略或映射为 ignore_index)
]

# 生成调色板颜色到类别ID的映射字典
palette = np.array(cityspallete).reshape(-1, 3)
color2label = {tuple(color): i for i, color in enumerate(palette)}

def convert_png_to_tensor(png_path):
    # 读取图片并转为 numpy 数组
    img = Image.open(png_path).convert('RGB')
    img_np = np.array(img)

    # 创建标签图（每个像素值为类别ID）
    label = np.full((img_np.shape[0], img_np.shape[1]), fill_value=255, dtype=np.uint8)  # 默认填充为 ignore_index

    # 映射每种颜色到标签ID
    for color, class_id in color2label.items():
        mask = np.all(img_np == color, axis=-1)
        label[mask] = class_id

    # 转为 PyTorch 张量
    return torch.from_numpy(label).long()  # [H, W] 类型为 long


def get_corresponding_label_tensor(meta, label_root='/data1/hl/sr/code/SeCo-main/PSA'):
    """
    从 img_meta 中构造标签路径，并读取为 label tensor
    """
    ori_filename = meta['ori_filename']  # e.g. 'erfurt/erfurt_000033_000019_leftImg8bit.png'
    basename = os.path.basename(ori_filename)  # 'erfurt_000033_000019_leftImg8bit.png'

    # 构造标签文件名
    label_filename = basename.replace('.png', '_sam_vit_h.png')
    label_path = os.path.join(label_root, label_filename)

    # 转为类别张量（使用 convert_png_to_tensor）
    label_tensor = convert_png_to_tensor(label_path)
    return label_tensor