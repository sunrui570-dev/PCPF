import cv2
import numpy as np
import os


def extract_connected_components(label_img, output_dir):
    """
    从标签图像中提取连通域并保存为单独的图像。

    参数：
    label_img: 输入标签图像 (numpy array)，每种颜色代表一个类别。
    output_dir: 保存结果图像的目录。
    """
    # 将输入图像转换为灰度图
    gray_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)

    # 二值化处理（忽略黑色部分）
    _, binary_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

    # 找到连通域
    num_labels, labels = cv2.connectedComponents(binary_img, connectivity=8)

    print(f"检测到的连通域数量（不含背景）：{num_labels - 1}")

    # 为每个连通域生成单独的图像
    for i in range(1, num_labels):  # 从1开始，跳过背景
        component_mask = (labels == i).astype(np.uint8) * 255
        output_img = np.full_like(binary_img, 255)  # 初始化为全白
        output_img[component_mask == 255] = 0  # 连通域位置设置为黑色

        # 保存结果图像
        output_path = os.path.join(output_dir, f"component_{i}.png")
        cv2.imwrite(output_path, output_img)
        print(f"连通域{i}保存为: {output_path}")


# 示例使用
if __name__ == "__main__":
    # 输入标签图像路径
    input_image_path = "test/gta_label/24966.png"

    # 输出目录
    output_directory = "./test"
    os.makedirs(output_directory, exist_ok=True)

    # 读取标签图像
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        print("无法读取输入图像，请检查路径！")
    else:
        extract_connected_components(input_image, output_directory)
