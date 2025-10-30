import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Cityscapes 的 RGB palette（19 类）
city_palette = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (0, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32)
]

palette_to_trainid = {rgb: i for i, rgb in enumerate(city_palette)}

def convert_color_mask_to_trainid(mask):
    h, w, _ = mask.shape
    label_map = np.full((h, w), 255, dtype=np.uint8)  # 默认填充 ignore_index = 255

    for rgb, train_id in palette_to_trainid.items():
        match = np.all(mask == rgb, axis=2)
        label_map[match] = train_id

    return label_map

def convert_all_images(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for fname in tqdm(files, desc=root):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            in_path = os.path.join(root, fname)
            rel_path = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            img = Image.open(in_path).convert('RGB')
            img_np = np.array(img)
            label_np = convert_color_mask_to_trainid(img_np)
            Image.fromarray(label_np).save(out_path)

if __name__ == '__main__':
    input_dir = '/data1/hl/sr/data/cityscapes/pl/train'
    output_dir = '/data1/hl/sr/data/cityscapes/pl2/train'
    convert_all_images(input_dir, output_dir)
    print("转换完成 ✅")
